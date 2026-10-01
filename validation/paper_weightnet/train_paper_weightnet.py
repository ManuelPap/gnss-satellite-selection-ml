#!/usr/bin/env python3
"""Smoke-test or train the released paper-era WeightNet on cached KLT3."""

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

from core import (
    instantiate_released_weightnet,
    parameter_gradient_norm,
    solve_paper_weighted_position,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=here / "klt3_features.npz")
    parser.add_argument(
        "--manifest", type=Path, default=here / "klt3_feature_manifest.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repository / "checkpoints/paper_weightnet/weightnet_3d.pth",
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


def full_dataset_loss(
    model: torch.nn.Module,
    dataset: dict[str, np.ndarray],
    device: torch.device,
    *,
    require_weight_gradients: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = dataset["features"]
    offsets = dataset["epoch_offsets"]
    loss: torch.Tensor | int = 0
    minimum_weight = float("inf")
    maximum_weight = float("-inf")
    maximum_wls_iterations = 0
    nonfinite_wls_states = 0

    context = torch.enable_grad() if require_weight_gradients else torch.no_grad()
    with context:
        for epoch_index in range(EXPECTED_EPOCHS):
            start = int(offsets[epoch_index])
            stop = int(offsets[epoch_index + 1])
            feature_tensor = torch.as_tensor(
                features[start:stop], dtype=torch.float32, device=device
            )
            # Preserve the source's float32 input construction. Subtracting the
            # frozen float64 mean in StandardizeLayer promotes the result before
            # it reaches the model's float64 Linear layers.
            weights = model(feature_tensor).squeeze(-1)
            if not bool(torch.all(torch.isfinite(weights)).detach().cpu()):
                raise RuntimeError(f"non-finite WeightNet output at epoch {epoch_index}")
            minimum_weight = min(minimum_weight, float(torch.min(weights).detach().cpu()))
            maximum_weight = max(maximum_weight, float(torch.max(weights).detach().cpu()))
            solution = solve_paper_weighted_position(
                dataset["satellite_positions_ecef_m"][start:stop],
                dataset["satellite_clock_bias_s"][start:stop],
                dataset["corrected_pseudorange_m"][start:stop],
                dataset["system_clock_indices"][start:stop],
                weights,
                dataset["initial_states"][epoch_index],
                return_trace=True,
            )
            maximum_wls_iterations = max(
                maximum_wls_iterations, len(solution.iterations)
            )
            if not bool(torch.all(torch.isfinite(solution.state)).detach().cpu()):
                nonfinite_wls_states += 1
            epoch_loss = epoch_position_loss(
                solution.state, dataset["ground_truth_geodetic_deg_m"][epoch_index]
            )
            if not bool(torch.isfinite(epoch_loss).detach().cpu()):
                raise RuntimeError(f"non-finite position loss at epoch {epoch_index}")
            loss = loss + epoch_loss
    assert isinstance(loss, torch.Tensor)
    if nonfinite_wls_states:
        raise RuntimeError(f"{nonfinite_wls_states} WLS states were non-finite")
    return loss, {
        "minimum_weight": minimum_weight,
        "maximum_weight": maximum_weight,
        "maximum_wls_iterations": maximum_wls_iterations,
    }


def gradient_audit(model: torch.nn.Module) -> dict[str, object]:
    per_parameter: dict[str, dict[str, object]] = {}
    nonzero_exists = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        present = gradient is not None
        finite = bool(present and torch.all(torch.isfinite(gradient)).detach().cpu())
        norm = float(torch.linalg.vector_norm(gradient).detach().cpu()) if present else None
        nonzero = bool(norm is not None and norm > 0.0)
        nonzero_exists = nonzero_exists or nonzero
        per_parameter[name] = {
            "present": present,
            "finite": finite,
            "nonzero": nonzero,
            "l2_norm": norm,
        }
    every_present = all(item["present"] for item in per_parameter.values())
    every_finite = all(item["finite"] for item in per_parameter.values())
    return {
        "per_parameter": per_parameter,
        "every_trainable_parameter_has_gradient": every_present,
        "every_trainable_parameter_gradient_finite": every_finite,
        "at_least_one_nonzero_gradient": nonzero_exists,
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
    model = instantiate_released_weightnet(feature_mean, feature_std, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RELEASED_LEARNING_RATE)
    requested_epochs = 1 if args.smoke else args.epochs
    metrics_path = (
        args.metrics.resolve()
        if not args.smoke
        else Path(__file__).resolve().parent / "smoke_metrics.json"
    )
    epoch_records: list[dict[str, object]] = []
    initial_loss: float | None = None
    first_gradient_norm: float | None = None
    first_gradient_audit: dict[str, object] | None = None
    first_parameter_changes: dict[str, float] | None = None
    start_time = time.perf_counter()

    for training_epoch in range(requested_epochs):
        optimizer.zero_grad()
        loss, forward_diagnostics = full_dataset_loss(model, dataset, device)
        loss_value = float(loss.detach().cpu())
        if initial_loss is None:
            initial_loss = loss_value
        loss.backward()
        audit = gradient_audit(model)
        if not audit["every_trainable_parameter_has_gradient"]:
            raise RuntimeError("at least one trainable parameter has no gradient")
        if not audit["every_trainable_parameter_gradient_finite"]:
            raise RuntimeError("at least one trainable parameter gradient is non-finite")
        if not audit["at_least_one_nonzero_gradient"]:
            raise RuntimeError("all trainable parameter gradients are zero")
        before = parameter_snapshot(model)
        optimizer.step()
        changes = parameter_changes(before, model)
        if not any(value > 0.0 for value in changes.values()):
            raise RuntimeError("Adam step did not change a trainable parameter")
        if training_epoch == 0:
            first_gradient_norm = float(audit["global_l2_norm"])
            first_gradient_audit = audit
            first_parameter_changes = changes
        epoch_records.append(
            {
                "epoch": training_epoch + 1,
                "loss_sum_3d_m": loss_value,
                "loss_divided_by_405_like_released_print": loss_value / EXPECTED_EPOCHS,
                "gradient_l2_norm": float(audit["global_l2_norm"]),
                **forward_diagnostics,
            }
        )
        if args.smoke or training_epoch == 0 or (training_epoch + 1) % 10 == 0:
            print(
                f"epoch {training_epoch + 1:03d}/{requested_epochs}: "
                f"loss={loss_value:.12f}, grad={audit['global_l2_norm']:.12e}, "
                f"weights=[{forward_diagnostics['minimum_weight']:.6f}, "
                f"{forward_diagnostics['maximum_weight']:.6f}]",
                flush=True,
            )

    post_training_loss, final_forward_diagnostics = full_dataset_loss(
        model, dataset, device, require_weight_gradients=False
    )
    duration_seconds = time.perf_counter() - start_time
    final_loss = float(post_training_loss.detach().cpu())
    checkpoint_record = None
    if not args.smoke:
        checkpoint_path = args.checkpoint.resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_record = {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
        }

    result = {
        "status": "passed",
        "mode": "smoke" if args.smoke else "full_training",
        "reference_commit": REFERENCE_COMMIT,
        "dataset": {
            "epoch_count": EXPECTED_EPOCHS,
            "measurement_count": EXPECTED_MEASUREMENTS,
            "feature_cache_sha256": feature_manifest["cache"]["sha256"],
        },
        "features": {
            "mean": feature_mean.tolist(),
            "population_std": feature_std.tolist(),
            "all_finite": True,
        },
        "configuration": {
            "seed": args.seed,
            "upstream_seed_behavior": "no seed set",
            "local_seed_behavior": (
                "Python, NumPy, and Torch seeded before default Linear initialization"
            ),
            "training_epochs": requested_epochs,
            "optimizer": "Adam",
            "learning_rate": RELEASED_LEARNING_RATE,
            "iteration_order": "chronological",
            "shuffle": False,
            "loss": "sum of per-epoch 3D ENU Euclidean position-error norms",
        },
        "runtime": runtime_versions(device),
        "initial_loss_sum_3d_m": initial_loss,
        "final_post_update_loss_sum_3d_m": final_loss,
        "first_gradient_l2_norm": first_gradient_norm,
        "first_gradient_audit": first_gradient_audit,
        "first_parameter_max_abs_changes": first_parameter_changes,
        "final_forward_diagnostics": final_forward_diagnostics,
        "duration_seconds": duration_seconds,
        "epochs": epoch_records,
        "checkpoint": checkpoint_record,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"post-update loss: {final_loss:.12f}")
    print(f"duration seconds: {duration_seconds:.3f}")
    print(f"metrics: {metrics_path}")
    if checkpoint_record is not None:
        print(f"checkpoint: {checkpoint_record['path']}")
        print(f"checkpoint SHA-256: {checkpoint_record['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
