#!/usr/bin/env python3
"""Smoke-test or train the released paper-era standalone BiasNet on KLT3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pymap3d as p3d
import torch

try:
    from .core import (
        instantiate_released_biasnet,
        parameter_gradient_norm,
        solve_paper_bias_position,
    )
except ImportError:
    from core import (
        instantiate_released_biasnet,
        parameter_gradient_norm,
        solve_paper_bias_position,
    )


REFERENCE_COMMIT = "dd5eac669676ba0a922102047e58c2dfc9be9267"
EXPECTED_EPOCHS = 405
EXPECTED_MEASUREMENTS = 8857
DEFAULT_SEED = 20_260_929
DEFAULT_TRAINING_EPOCHS = 500
RELEASED_LEARNING_RATE = 0.01


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repository = here.parents[1]
    shared = repository / "validation/paper_weightnet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=shared / "klt3_features.npz")
    parser.add_argument(
        "--manifest", type=Path, default=shared / "klt3_feature_manifest.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repository / "checkpoints/paper_biasnet/biasnet_3d.pth",
    )
    parser.add_argument(
        "--metrics", type=Path, default=here / "training_metrics.json"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one full-dataset forward/backward/Adam step and validation.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(
    path: Path, manifest_path: Path, device: torch.device
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = path.resolve()
    manifest_path = manifest_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"feature cache does not exist: {path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"feature manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    actual_hash = sha256(path)
    if actual_hash != manifest["cache"]["sha256"]:
        raise RuntimeError(
            f"feature cache hash mismatch: expected {manifest['cache']['sha256']}, "
            f"got {actual_hash}"
        )
    with np.load(path, allow_pickle=False) as cache:
        dataset = {name: cache[name].copy() for name in cache.files}
    features = dataset["features"]
    offsets = dataset["epoch_offsets"]
    if features.shape != (EXPECTED_MEASUREMENTS, 3):
        raise RuntimeError(f"unexpected feature shape: {features.shape}")
    if offsets.shape != (EXPECTED_EPOCHS + 1,) or int(offsets[-1]) != EXPECTED_MEASUREMENTS:
        raise RuntimeError(f"unexpected epoch offsets: {offsets.shape}/{offsets[-1]}")
    if not np.all(np.diff(offsets) >= 4):
        raise RuntimeError("an epoch has fewer than four retained observations")
    for name, values in dataset.items():
        if values.dtype.kind in "f" and not np.all(np.isfinite(values)):
            raise RuntimeError(f"cached array {name} contains a non-finite value")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return dataset, manifest


def epoch_position_loss(
    predicted_state: torch.Tensor, ground_truth_geodetic: np.ndarray
) -> torch.Tensor:
    east, north, up = p3d.ecef2enu(
        predicted_state[0],
        predicted_state[1],
        predicted_state[2],
        float(ground_truth_geodetic[0]),
        float(ground_truth_geodetic[1]),
        float(ground_truth_geodetic[2]),
    )
    return torch.linalg.vector_norm(torch.stack((east, north, up)))


def released_training_ground_truth_index(epoch_index: int) -> int:
    """Reproduce the duplicate-append bug in ``bias_network_train.py``."""

    if not 0 <= epoch_index < EXPECTED_EPOCHS:
        raise IndexError(epoch_index)
    return epoch_index // 2


def full_dataset_loss(
    model: torch.nn.Module,
    dataset: dict[str, np.ndarray],
    device: torch.device,
    *,
    require_bias_gradients: bool = True,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    features = dataset["features"]
    offsets = dataset["epoch_offsets"]
    ground_truth = dataset["ground_truth_geodetic_deg_m"]
    loss: torch.Tensor | int = 0
    minimum_bias = float("inf")
    maximum_bias = float("-inf")
    maximum_wls_iterations = 0
    nonfinite_wls_states = 0

    context = torch.enable_grad() if require_bias_gradients else torch.no_grad()
    with context:
        for epoch_index in range(EXPECTED_EPOCHS):
            start = int(offsets[epoch_index])
            stop = int(offsets[epoch_index + 1])
            feature_tensor = torch.as_tensor(
                features[start:stop], dtype=torch.float32, device=device
            )
            predicted_bias = model(feature_tensor).squeeze(-1)
            if not bool(torch.all(torch.isfinite(predicted_bias)).detach().cpu()):
                raise RuntimeError(f"non-finite BiasNet output at epoch {epoch_index}")
            minimum_bias = min(
                minimum_bias, float(torch.min(predicted_bias).detach().cpu())
            )
            maximum_bias = max(
                maximum_bias, float(torch.max(predicted_bias).detach().cpu())
            )
            solution = solve_paper_bias_position(
                dataset["satellite_positions_ecef_m"][start:stop],
                dataset["satellite_clock_bias_s"][start:stop],
                dataset["corrected_pseudorange_m"][start:stop],
                dataset["system_clock_indices"][start:stop],
                predicted_bias,
                dataset["initial_states"][epoch_index],
                return_trace=True,
            )
            maximum_wls_iterations = max(
                maximum_wls_iterations, len(solution.iterations)
            )
            if not bool(torch.all(torch.isfinite(solution.state)).detach().cpu()):
                nonfinite_wls_states += 1
            # This intentionally does not use epoch_index.  The released bias
            # trainer appended every successful epoch's GT twice, then indexed
            # the resulting list with i=0..404.
            gt_index = released_training_ground_truth_index(epoch_index)
            epoch_loss = epoch_position_loss(solution.state, ground_truth[gt_index])
            if not bool(torch.isfinite(epoch_loss).detach().cpu()):
                raise RuntimeError(f"non-finite position loss at epoch {epoch_index}")
            loss = loss + epoch_loss
    assert isinstance(loss, torch.Tensor)
    if nonfinite_wls_states:
        raise RuntimeError(f"{nonfinite_wls_states} WLS states were non-finite")
    return loss, {
        "minimum_bias_m": minimum_bias,
        "maximum_bias_m": maximum_bias,
        "maximum_wls_iterations": maximum_wls_iterations,
    }


def gradient_audit(model: torch.nn.Module) -> dict[str, object]:
    per_parameter: dict[str, dict[str, object]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        present = gradient is not None
        finite = bool(present and torch.all(torch.isfinite(gradient)).detach().cpu())
        norm = float(torch.linalg.vector_norm(gradient).detach().cpu()) if present else None
        per_parameter[name] = {
            "present": present,
            "finite": finite,
            "nonzero": bool(norm is not None and norm > 0.0),
            "l2_norm": norm,
        }
    return {
        "per_parameter": per_parameter,
        "every_trainable_parameter_has_gradient": all(
            item["present"] for item in per_parameter.values()
        ),
        "every_trainable_parameter_gradient_finite": all(
            item["finite"] for item in per_parameter.values()
        ),
        "every_trainable_tensor_has_nonzero_gradient": all(
            item["nonzero"] for item in per_parameter.values()
        ),
        "global_l2_norm": parameter_gradient_norm(model),
    }


def parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def parameter_changes(
    before: dict[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, float]:
    return {
        name: float(torch.max(torch.abs(parameter.detach() - before[name])).cpu())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def bias_statistics(values: np.ndarray) -> dict[str, float | bool]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum_m": float(values.min()),
        "maximum_m": float(values.max()),
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "population_std_m": float(values.std()),
        "p5_m": float(np.percentile(values, 5.0)),
        "p95_m": float(np.percentile(values, 95.0)),
        "all_finite": bool(np.all(np.isfinite(values))),
    }


def training_bias_diagnostics(
    model: torch.nn.Module,
    dataset: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, object]:
    with torch.no_grad():
        values = (
            model(torch.as_tensor(dataset["features"], dtype=torch.float32, device=device))
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
    stored_mean = model.seq[0].mean.detach().cpu().numpy()
    stored_std = model.seq[0].std.detach().cpu().numpy()
    normalized = (
        dataset["features"].astype(np.float32).astype(np.float64) - stored_mean
    ) / stored_std
    order = np.argsort(np.abs(values))[::-1][:10]
    return {
        "statistics": bias_statistics(values),
        "largest_absolute_outputs": [
            {
                "measurement_index": int(index),
                "satellite": str(dataset["satellite_ids"][index]),
                "features": dataset["features"][index].tolist(),
                "normalized_features": normalized[index].tolist(),
                "maximum_absolute_feature_z_score": float(
                    np.max(np.abs(normalized[index]))
                ),
                "has_feature_beyond_3_population_std": bool(
                    np.max(np.abs(normalized[index])) > 3.0
                ),
                "predicted_bias_m": float(values[index]),
            }
            for index in order
        ],
    }


def runtime_versions(device: torch.device) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pymap3d": importlib.metadata.version("pymap3d"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    device = torch.device(args.device)
    dataset, feature_manifest = load_dataset(args.features, args.manifest, device)
    set_seed(args.seed)

    feature_mean = dataset["features"].mean(axis=0)
    feature_std = dataset["features"].std(axis=0)
    model = instantiate_released_biasnet(feature_mean, feature_std, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RELEASED_LEARNING_RATE)
    requested_epochs = 1 if args.smoke else args.epochs
    metrics_path = (
        Path(__file__).resolve().parent / "smoke_metrics.json"
        if args.smoke
        else args.metrics.resolve()
    )
    checkpoint_path = args.checkpoint.resolve()
    history: list[dict[str, object]] = []
    first_gradient: dict[str, object] | None = None
    first_changes: dict[str, float] | None = None
    start_time = time.perf_counter()

    for training_epoch in range(requested_epochs):
        before = parameter_snapshot(model) if training_epoch == 0 else None
        optimizer.zero_grad()
        loss, diagnostics = full_dataset_loss(model, dataset, device)
        loss.backward()
        audit = gradient_audit(model)
        if not audit["every_trainable_parameter_has_gradient"]:
            raise RuntimeError("a trainable tensor did not receive a gradient")
        if not audit["every_trainable_parameter_gradient_finite"]:
            raise RuntimeError("a trainable tensor received a non-finite gradient")
        optimizer.step()
        if training_epoch == 0:
            assert before is not None
            first_gradient = audit
            first_changes = parameter_changes(before, model)
            if not all(change > 0.0 for change in first_changes.values()):
                raise RuntimeError("the first optimizer step did not change every tensor")
        summed = float(loss.detach().cpu())
        row = {
            "epoch": training_epoch + 1,
            "loss_sum_3d_m": summed,
            "loss_divided_by_405_like_released_print": summed / EXPECTED_EPOCHS,
            "gradient_l2_norm": audit["global_l2_norm"],
            **diagnostics,
        }
        history.append(row)
        print(
            f"epoch {training_epoch + 1:03d}/{requested_epochs}: "
            f"mean_like_released={summed / EXPECTED_EPOCHS:.12g} m "
            f"sum={summed:.12g} m"
        )

    duration = time.perf_counter() - start_time
    metrics: dict[str, object] = {
        "status": "passed",
        "reference_commit": REFERENCE_COMMIT,
        "mode": "one_epoch_full_dataset_smoke" if args.smoke else "full_training",
        "configuration": {
            "seed": args.seed,
            "local_seed_behavior": (
                "Python, NumPy, and Torch seeded before default Linear initialization"
            ),
            "upstream_seed_behavior": "no seed set",
            "optimizer": "Adam",
            "learning_rate": RELEASED_LEARNING_RATE,
            "training_epochs": requested_epochs,
            "shuffle": False,
            "batch_config_value": 128,
            "batch_config_effect": "unused; one optimizer step after all epochs-in-dataset",
            "loss": "sum of per-epoch 3D ENU Euclidean position-error norms",
            "ground_truth_alignment": (
                "released duplicate-append defect: training epoch i uses KLT3 GT row floor(i/2)"
            ),
        },
        "runtime": runtime_versions(device),
        "duration_seconds": duration,
        "dataset": {
            "epoch_count": EXPECTED_EPOCHS,
            "measurement_count": EXPECTED_MEASUREMENTS,
            "feature_cache_sha256": sha256(args.features.resolve()),
            "feature_mean_float64": feature_mean.tolist(),
            "feature_population_std_float64": feature_std.tolist(),
            "normalization_matches_weightnet_shared_population_exactly": True,
            "manifest_reference": str(args.manifest.resolve()),
            "manifest_status": feature_manifest["status"],
        },
        "first_gradient_audit": first_gradient,
        "first_parameter_max_abs_changes": first_changes,
        "epochs": history,
    }
    if not args.smoke:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        metrics["checkpoint"] = {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
        }
        metrics["bias_output_diagnostics_klt3"] = training_bias_diagnostics(
            model, dataset, device
        )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"metrics: {metrics_path}")
    if not args.smoke:
        print(f"checkpoint: {checkpoint_path}")
        print(f"checkpoint SHA-256: {metrics['checkpoint']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
