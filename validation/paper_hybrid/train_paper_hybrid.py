#!/usr/bin/env python3
"""Smoke-test or train released paper-era HybridShareNet on cached KLT3."""

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
        instantiate_released_hybrid,
        parameter_gradient_norm,
        solve_paper_hybrid_position,
    )
except ImportError:
    from core import (
        instantiate_released_hybrid,
        parameter_gradient_norm,
        solve_paper_hybrid_position,
    )

from validation.paper_biasnet.train_paper_biasnet import load_dataset


REFERENCE_COMMIT = "dd5eac669676ba0a922102047e58c2dfc9be9267"
EXPECTED_EPOCHS = 405
EXPECTED_MEASUREMENTS = 8857
DEFAULT_SEED = 20_260_929
DEFAULT_TRAINING_EPOCHS = 100
RELEASED_LEARNING_RATE = 0.01
MAXIMUM_WLS_ITERATIONS = 10
WLS_TOLERANCE = 1.0e-4


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repository = here.parents[1]
    shared = here.parent / "paper_weightnet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=shared / "klt3_features.npz")
    parser.add_argument(
        "--manifest", type=Path, default=shared / "klt3_feature_manifest.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repository / "checkpoints/paper_hybrid/hybrid_share_3d.pth",
    )
    parser.add_argument("--metrics", type=Path, default=here / "training_metrics.json")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one full-dataset forward/backward/Adam step without checkpointing.",
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
    require_gradients: bool = True,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    features = dataset["features"]
    offsets = dataset["epoch_offsets"]
    loss: torch.Tensor | int = 0
    minimum_bias = float("inf")
    maximum_bias = float("-inf")
    minimum_weight = float("inf")
    maximum_weight = float("-inf")
    maximum_wls_iterations = 0

    context = torch.enable_grad() if require_gradients else torch.no_grad()
    with context:
        for epoch_index in range(EXPECTED_EPOCHS):
            start = int(offsets[epoch_index])
            stop = int(offsets[epoch_index + 1])
            feature_tensor = torch.as_tensor(
                features[start:stop], dtype=torch.float32, device=device
            )
            weight, bias = model(feature_tensor)
            if not bool(torch.all(torch.isfinite(weight)).detach().cpu()):
                raise RuntimeError(f"non-finite weight at epoch {epoch_index}")
            if not bool(torch.all(torch.isfinite(bias)).detach().cpu()):
                raise RuntimeError(f"non-finite bias at epoch {epoch_index}")
            minimum_bias = min(minimum_bias, float(torch.min(bias).detach().cpu()))
            maximum_bias = max(maximum_bias, float(torch.max(bias).detach().cpu()))
            minimum_weight = min(
                minimum_weight, float(torch.min(weight).detach().cpu())
            )
            maximum_weight = max(
                maximum_weight, float(torch.max(weight).detach().cpu())
            )
            solution = solve_paper_hybrid_position(
                dataset["satellite_positions_ecef_m"][start:stop],
                dataset["satellite_clock_bias_s"][start:stop],
                dataset["corrected_pseudorange_m"][start:stop],
                dataset["system_clock_indices"][start:stop],
                weight,
                bias,
                dataset["initial_states"][epoch_index],
                return_trace=True,
            )
            maximum_wls_iterations = max(
                maximum_wls_iterations, len(solution.iterations)
            )
            if not bool(torch.all(torch.isfinite(solution.state)).detach().cpu()):
                raise RuntimeError(f"non-finite WLS state at epoch {epoch_index}")
            epoch_loss = epoch_position_loss(
                solution.state, dataset["ground_truth_geodetic_deg_m"][epoch_index]
            )
            if not bool(torch.isfinite(epoch_loss).detach().cpu()):
                raise RuntimeError(f"non-finite position loss at epoch {epoch_index}")
            loss = loss + epoch_loss
    assert isinstance(loss, torch.Tensor)
    return loss, {
        "minimum_bias_m": minimum_bias,
        "maximum_bias_m": maximum_bias,
        "minimum_weight": minimum_weight,
        "maximum_weight": maximum_weight,
        "maximum_wls_iterations": maximum_wls_iterations,
    }


def gradient_audit(model: torch.nn.Module) -> dict[str, object]:
    records: dict[str, dict[str, object]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        present = gradient is not None
        finite = bool(present and torch.all(torch.isfinite(gradient)).detach().cpu())
        norm = float(torch.linalg.vector_norm(gradient).detach().cpu()) if present else None
        records[name] = {
            "present": present,
            "finite": finite,
            "nonzero": bool(norm is not None and norm > 0.0),
            "l2_norm": norm,
        }
    return {
        "per_parameter": records,
        "every_trainable_parameter_has_gradient": all(
            item["present"] for item in records.values()
        ),
        "every_trainable_parameter_gradient_finite": all(
            item["finite"] for item in records.values()
        ),
        "every_trainable_tensor_gradient_nonzero": all(
            item["nonzero"] for item in records.values()
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


def output_statistics(values: np.ndarray, unit: str) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "unit": unit,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "population_std": float(values.std()),
        "p5": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
        "all_finite": bool(np.all(np.isfinite(values))),
    }


def frozen_training_outputs(
    model: torch.nn.Module, dataset: dict[str, np.ndarray], device: torch.device
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        feature_tensor = torch.as_tensor(
            dataset["features"], dtype=torch.float32, device=device
        )
        weight, bias = model(feature_tensor)
    weights = weight.detach().cpu().numpy()
    biases = bias.detach().cpu().numpy()
    result = {
        "bias": output_statistics(biases, "metre"),
        "weight": output_statistics(
            weights, "dimensionless relative WLS coefficient"
        ),
        "weight_threshold_fractions": {
            "weight_lt_1e-5": float(np.mean(weights < 1.0e-5)),
            "weight_lt_0.01": float(np.mean(weights < 0.01)),
            "weight_gt_0.5": float(np.mean(weights > 0.5)),
            "weight_gt_0.99": float(np.mean(weights > 0.99)),
        },
        "clipping_imposed_beyond_released_code": False,
    }
    model.train()
    return result


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    device = torch.device(args.device)
    dataset, manifest = load_dataset(args.features, args.manifest, device)
    set_seed(args.seed)
    feature_mean = dataset["features"].mean(axis=0)
    feature_std = dataset["features"].std(axis=0)
    model = instantiate_released_hybrid(feature_mean, feature_std, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RELEASED_LEARNING_RATE)
    requested_epochs = 1 if args.smoke else args.epochs
    metrics_path = (
        Path(__file__).resolve().parent / "smoke_metrics.json"
        if args.smoke
        else args.metrics.resolve()
    )
    epoch_records: list[dict[str, object]] = []
    first_audit = None
    first_changes = None
    total_start = time.perf_counter()

    for training_epoch in range(requested_epochs):
        epoch_start = time.perf_counter()
        optimizer.zero_grad()
        loss, forward = full_dataset_loss(model, dataset, device)
        loss_value = float(loss.detach().cpu())
        loss.backward()
        audit = gradient_audit(model)
        if not audit["every_trainable_parameter_has_gradient"]:
            raise RuntimeError("a trainable tensor has no gradient")
        if not audit["every_trainable_parameter_gradient_finite"]:
            raise RuntimeError("a trainable tensor has a non-finite gradient")
        if not audit["every_trainable_tensor_gradient_nonzero"]:
            raise RuntimeError("a trainable tensor has an all-zero gradient")
        before = parameter_snapshot(model)
        optimizer.step()
        changes = parameter_changes(before, model)
        if not all(value > 0.0 for value in changes.values()):
            raise RuntimeError("Adam did not change every expected trainable tensor")
        if training_epoch == 0:
            first_audit = audit
            first_changes = changes
        record = {
            "epoch": training_epoch + 1,
            "loss_sum_3d_m": loss_value,
            "loss_divided_by_405_like_released_print": loss_value / EXPECTED_EPOCHS,
            "gradient_l2_norm": float(audit["global_l2_norm"]),
            "duration_seconds": time.perf_counter() - epoch_start,
            **forward,
        }
        epoch_records.append(record)
        if args.smoke or training_epoch == 0 or (training_epoch + 1) % 10 == 0:
            print(
                f"epoch {training_epoch + 1:03d}/{requested_epochs}: "
                f"loss={loss_value:.12f}, mean-like={loss_value / EXPECTED_EPOCHS:.12f}, "
                f"grad={audit['global_l2_norm']:.12e}, "
                f"bias=[{forward['minimum_bias_m']:.6g},{forward['maximum_bias_m']:.6g}], "
                f"weight=[{forward['minimum_weight']:.6g},{forward['maximum_weight']:.6g}]",
                flush=True,
            )

    post_loss, final_forward = full_dataset_loss(
        model, dataset, device, require_gradients=False
    )
    final_loss = float(post_loss.detach().cpu())
    checkpoint_record = None
    if not args.smoke:
        checkpoint_path = args.checkpoint.resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_record = {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
            "all_tensors_finite_after_reload": all(
                bool(torch.all(torch.isfinite(value))) for value in reloaded.values()
            ),
        }
    minimum_record = min(epoch_records, key=lambda row: row["loss_sum_3d_m"])
    result = {
        "status": "passed",
        "mode": "smoke" if args.smoke else "full_training",
        "reference_commit": REFERENCE_COMMIT,
        "dataset": {
            "training_dataset": "KLT3",
            "held_out_datasets": ["KLT1", "KLT2"],
            "held_out_data_used_for_training": False,
            "epoch_count": EXPECTED_EPOCHS,
            "gt_target_count": EXPECTED_EPOCHS,
            "measurement_count": EXPECTED_MEASUREMENTS,
            "feature_cache_sha256": manifest["cache"]["sha256"],
        },
        "features": {
            "mean_float64": feature_mean.tolist(),
            "population_std_float64": feature_std.tolist(),
            "checkpoint_mean_float32_then_double": feature_mean.astype(np.float32).astype(np.float64).tolist(),
            "checkpoint_std_float32_then_double": feature_std.astype(np.float32).astype(np.float64).tolist(),
        },
        "configuration": {
            "seed": args.seed,
            "upstream_seed_behavior": "no seed set",
            "local_seed_behavior": "Python, NumPy, and Torch seeded before default Linear initialization",
            "training_epochs": requested_epochs,
            "optimizer": "Adam",
            "learning_rate": RELEASED_LEARNING_RATE,
            "adam_defaults_other_than_learning_rate": True,
            "iteration_order": "chronological",
            "shuffle": False,
            "batch_configuration_read_but_unused": 128,
            "optimizer_updates_per_training_epoch": 1,
            "loss": "sum of per-epoch 3D ENU Euclidean position-error norms",
            "imported_mse_loss_used": False,
            "maximum_wls_iterations": MAXIMUM_WLS_ITERATIONS,
            "wls_tolerance": WLS_TOLERANCE,
            "initialization": "from scratch; no standalone BiasNet or WeightNet weights loaded",
            "bias_output": "ReLU, nonnegative, metres",
            "weight_output": "sigmoid then clamp [0,1], dimensionless",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pymap3d": importlib.metadata.version("pymap3d"),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "initial_loss_sum_3d_m": epoch_records[0]["loss_sum_3d_m"],
        "initial_mean_like_loss_m": epoch_records[0]["loss_divided_by_405_like_released_print"],
        "final_pre_update_loss_sum_3d_m": epoch_records[-1]["loss_sum_3d_m"],
        "final_post_update_loss_sum_3d_m": final_loss,
        "minimum_pre_update_loss": {
            "epoch": minimum_record["epoch"],
            "loss_sum_3d_m": minimum_record["loss_sum_3d_m"],
            "mean_like_m": minimum_record["loss_divided_by_405_like_released_print"],
        },
        "first_gradient_audit": first_audit,
        "first_parameter_max_abs_changes": first_changes,
        "final_forward_diagnostics": final_forward,
        "final_training_output_diagnostics": frozen_training_outputs(
            model, dataset, device
        ),
        "duration_seconds": time.perf_counter() - total_start,
        "epochs": epoch_records,
        "checkpoint": checkpoint_record,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"post-update loss: {final_loss:.12f}")
    print(f"metrics: {metrics_path}")
    if checkpoint_record:
        print(f"checkpoint SHA-256: {checkpoint_record['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
