#!/usr/bin/env python3
"""Train BiasNet with the sole correction KLT3 epoch i -> matched GT i."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from validation.paper_biasnet.train_paper_biasnet import (
    DEFAULT_SEED,
    DEFAULT_TRAINING_EPOCHS,
    EXPECTED_EPOCHS,
    RELEASED_LEARNING_RATE,
    bias_statistics,
    gradient_audit,
    load_dataset,
    parameter_changes,
    parameter_snapshot,
    runtime_versions,
    sha256,
    training_bias_diagnostics,
)
from validation.paper_biasnet_corrected_gt.experiment import (
    MAXIMUM_WLS_ITERATIONS,
    WLS_TOLERANCE,
    ab_control_record,
    corrected_full_dataset_loss,
    initialize_controlled_ab,
    validate_training_dataset,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    shared = root / "validation/paper_weightnet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=shared / "klt3_features.npz")
    parser.add_argument(
        "--manifest", type=Path, default=shared / "klt3_feature_manifest.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "checkpoints/paper_biasnet_corrected_gt/biasnet_3d.pth",
    )
    parser.add_argument("--metrics", type=Path, default=here / "training_metrics.json")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one corrected full-dataset forward/backward/Adam step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    device = torch.device(args.device)
    dataset, feature_manifest = load_dataset(args.features, args.manifest, device)
    validate_training_dataset(dataset)
    feature_mean = dataset["features"].mean(axis=0)
    feature_std = dataset["features"].std(axis=0)
    model, optimizer, initialization = initialize_controlled_ab(
        feature_mean, feature_std, seed=args.seed, device=device
    )
    controls = ab_control_record(dataset, initialization)
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
    training_start = time.perf_counter()

    for training_epoch in range(requested_epochs):
        epoch_start = time.perf_counter()
        before = parameter_snapshot(model) if training_epoch == 0 else None
        optimizer.zero_grad()
        loss, diagnostics = corrected_full_dataset_loss(model, dataset, device)
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
        epoch_duration = time.perf_counter() - epoch_start
        row = {
            "epoch": training_epoch + 1,
            "loss_sum_3d_m": summed,
            "loss_divided_by_405_like_released_print": summed / EXPECTED_EPOCHS,
            "gradient_l2_norm": audit["global_l2_norm"],
            "duration_seconds": epoch_duration,
            **diagnostics,
        }
        history.append(row)
        print(
            f"epoch {training_epoch + 1:03d}/{requested_epochs}: "
            f"mean_like_released={summed / EXPECTED_EPOCHS:.12g} m "
            f"sum={summed:.12g} m duration={epoch_duration:.3f} s",
            flush=True,
        )

    duration = time.perf_counter() - training_start
    metrics: dict[str, object] = {
        "status": "passed",
        "experiment": "Corrected-GT BiasNet controlled experiment",
        "reference_commit": "dd5eac669676ba0a922102047e58c2dfc9be9267",
        "mode": "one_epoch_full_dataset_smoke" if args.smoke else "full_training",
        "configuration": {
            "seed": args.seed,
            "local_seed_behavior": (
                "Python, NumPy, and Torch seeded before default Linear initialization"
            ),
            "optimizer": "Adam",
            "learning_rate": RELEASED_LEARNING_RATE,
            "training_epochs": requested_epochs,
            "shuffle": False,
            "batch_config_value": 128,
            "batch_config_effect": "unused; one update after all dataset epochs",
            "loss": "sum of per-epoch 3D ENU Euclidean position-error norms",
            "ground_truth_alignment": (
                "corrected one-to-one mapping: training epoch i uses its matched KLT3 GT row i"
            ),
            "bias_output": "linear, unbounded, metres",
            "bias_sign": "corrected_pseudorange_m = pseudorange_m - predicted_bias_m",
            "maximum_wls_iterations": MAXIMUM_WLS_ITERATIONS,
            "wls_tolerance": WLS_TOLERANCE,
        },
        "runtime": runtime_versions(device),
        "duration_seconds": duration,
        "dataset": {
            "name": "KLT3",
            "epoch_count": EXPECTED_EPOCHS,
            "gt_target_count": int(dataset["ground_truth_geodetic_deg_m"].shape[0]),
            "measurement_count": int(dataset["features"].shape[0]),
            "feature_cache_sha256": sha256(args.features.resolve()),
            "feature_mean_float64": feature_mean.tolist(),
            "feature_population_std_float64": feature_std.tolist(),
            "manifest_reference": str(args.manifest.resolve()),
            "manifest_status": feature_manifest["status"],
        },
        "ab_control": controls,
        "first_gradient_audit": first_gradient,
        "first_parameter_max_abs_changes": first_changes,
        "epochs": history,
    }
    if not args.smoke:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not all(torch.all(torch.isfinite(value)) for value in reloaded.values()):
            raise RuntimeError("saved corrected checkpoint contains a non-finite value")
        metrics["checkpoint"] = {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
            "all_tensors_finite_after_reload": True,
        }
        diagnostics = training_bias_diagnostics(model, dataset, device)
        diagnostics["statistics_recomputed_independently"] = bias_statistics(
            model(
                torch.as_tensor(
                    dataset["features"], dtype=torch.float32, device=device
                )
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        metrics["bias_output_diagnostics_klt3"] = diagnostics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"metrics: {metrics_path}")
    if not args.smoke:
        print(f"checkpoint: {checkpoint_path}")
        print(f"checkpoint SHA-256: {metrics['checkpoint']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
