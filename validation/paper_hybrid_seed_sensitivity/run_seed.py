#!/usr/bin/env python3
"""Run one independent, controlled TDL-BW initialization-seed experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from validation.ibiza_generalization.runtime_cache import (  # noqa: E402
    DEFAULT_RUNTIME_DIR,
)
from validation.paper_hybrid.held_out import (  # noqa: E402
    DATASET_SPECS,
    evaluate_prepared_dataset,
    prepare_dataset,
    resolve_input_paths,
)
from validation.paper_hybrid_seed_sensitivity.experiment import (  # noqa: E402
    TRAINING_EPOCHS,
    build_initial_snapshot,
    canonical_json_hash,
    dataset_output_diagnostics,
    file_sha256,
    full_dataset_loss_with_diagnostics,
    gradient_diagnostics,
    load_dataset,
    model_tensors_finite,
    normalization_identity,
    scientific_configuration,
    training_data_identity,
    validate_training_dataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    shared = REPOSITORY / "validation/paper_weightnet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--features", type=Path, default=shared / "klt3_features.npz")
    parser.add_argument(
        "--manifest", type=Path, default=shared / "klt3_feature_manifest.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY / "results/paper_hybrid_seed_sensitivity",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPOSITORY / "checkpoints/paper_hybrid_seed_sensitivity",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument(
        "--ephemeris-glob", action="append", dest="ephemeris_patterns"
    )
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run one KLT3 optimization epoch and skip checkpoint/KLT1/KLT2; "
            "this is not a completed seed result."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing result (and checkpoint for a complete run).",
    )
    return parser.parse_args(argv)


def held_out_metrics(
    model: torch.nn.Module, args: argparse.Namespace
) -> dict[str, dict[str, object]]:
    model.to("cpu")
    model.eval()
    model.requires_grad_(False)
    result: dict[str, dict[str, object]] = {}
    for dataset_name in ("KLT1", "KLT2"):
        spec = DATASET_SPECS[dataset_name]
        inputs = resolve_input_paths(
            spec,
            data_root=args.data_root,
            observation=args.observation,
            ephemeris_patterns=args.ephemeris_patterns,
            ground_truth=args.ground_truth,
            runtime_dir=args.runtime_dir,
        )
        prepared = prepare_dataset(spec, inputs)
        evaluations = evaluate_prepared_dataset(model, prepared)
        errors_2d = np.asarray([item.error_2d_m for item in evaluations])
        errors_3d = np.asarray([item.error_3d_m for item in evaluations])
        if not np.all(np.isfinite(errors_2d)) or not np.all(np.isfinite(errors_3d)):
            raise RuntimeError(f"{dataset_name} evaluation produced a non-finite error")
        result[dataset_name] = {
            "mean_2d_error_m": float(errors_2d.mean()),
            "mean_3d_error_m": float(errors_3d.mean()),
            "valid_epoch_count": len(evaluations),
            "retained_measurement_count": prepared.measurement_count,
            "all_finite": True,
        }
    return result


def _target_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    prefix = "smoke_seed" if args.smoke else "seed"
    result_path = args.output_dir.resolve() / f"{prefix}_{args.seed}.json"
    checkpoint_path = None
    if not args.smoke:
        checkpoint_path = (
            args.checkpoint_dir.resolve() / f"seed_{args.seed}_hybrid_share_3d.pth"
        )
    return result_path, checkpoint_path


def _ensure_targets_available(
    result_path: Path, checkpoint_path: Path | None, overwrite: bool
) -> None:
    occupied = [path for path in (result_path, checkpoint_path) if path and path.exists()]
    if occupied and not overwrite:
        rendered = ", ".join(str(path) for path in occupied)
        raise FileExistsError(f"output already exists (use --overwrite): {rendered}")


def run(args: argparse.Namespace) -> dict[str, object]:
    result_path, checkpoint_path = _target_paths(args)
    _ensure_targets_available(result_path, checkpoint_path, args.overwrite)
    device = torch.device(args.device)

    # All data/control identities are established before seeded construction.
    dataset, manifest = load_dataset(args.features, args.manifest, device)
    validate_training_dataset(dataset)
    data_identity = training_data_identity(dataset, manifest)
    normalization = normalization_identity(dataset)

    model, optimizer, initial_snapshot = build_initial_snapshot(
        dataset, args.seed, device
    )
    configuration = scientific_configuration(model, optimizer)
    configuration_sha256 = canonical_json_hash(configuration)
    if configuration_sha256 != initial_snapshot["configuration_sha256"]:
        raise RuntimeError("configuration hash changed during initialization")
    initial_outputs = dataset_output_diagnostics(model, dataset, device)
    requested_epochs = 1 if args.smoke else TRAINING_EPOCHS
    epoch_records: list[dict[str, object]] = []
    total_start = time.perf_counter()

    for epoch_index in range(requested_epochs):
        epoch_start = time.perf_counter()
        optimizer.zero_grad()
        loss, outputs = full_dataset_loss_with_diagnostics(model, dataset, device)
        loss_value = float(loss.detach().cpu())
        loss.backward()
        gradients = gradient_diagnostics(model)
        if not gradients["all_trainable_gradients_finite"]:
            raise RuntimeError(f"non-finite gradient at training epoch {epoch_index + 1}")
        optimizer.step()
        if not model_tensors_finite(model):
            raise RuntimeError(f"non-finite model tensor at training epoch {epoch_index + 1}")
        epoch_records.append(
            {
                "epoch": epoch_index + 1,
                "loss_sum_3d_m": loss_value,
                "loss_divided_by_405_m": loss_value / 405.0,
                "positive_bias_preactivation_fraction": outputs[
                    "bias_preactivation"
                ]["fraction_gt_zero"],
                "outputs": outputs,
                "gradients": gradients,
                "duration_seconds": time.perf_counter() - epoch_start,
            }
        )
        if args.smoke or epoch_index == 0 or (epoch_index + 1) % 10 == 0:
            print(
                f"seed {args.seed} epoch {epoch_index + 1:03d}/{requested_epochs}: "
                f"loss={loss_value:.12f}, "
                f"positive-preactivation="
                f"{outputs['bias_preactivation']['fraction_gt_zero']:.6f}, "
                f"bias-grad={gradients['bias_output_row_l2_norm']:.6e}",
                flush=True,
            )

    final_loss_tensor, final_outputs = full_dataset_loss_with_diagnostics(
        model, dataset, device, require_gradients=False
    )
    final_loss = float(final_loss_tensor.detach().cpu())
    losses = [float(row["loss_sum_3d_m"]) for row in epoch_records]
    minimum_index = int(np.argmin(losses))

    checkpoint_record: dict[str, object] | None = None
    evaluation: dict[str, dict[str, object]] | None = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_model = model.to("cpu")
        torch.save(cpu_model.state_dict(), checkpoint_path)
        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_record = {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "all_tensors_finite_after_reload": all(
                bool(torch.all(torch.isfinite(value))) for value in reloaded.values()
            ),
        }
        evaluation = held_out_metrics(cpu_model, args)

    completed = not args.smoke
    finite_checks = {
        "epoch_losses": all(math.isfinite(value) for value in losses),
        "final_loss": math.isfinite(final_loss),
        "initial_outputs": all(
            bool(initial_outputs[name]["all_finite"])
            for name in ("bias_preactivation", "bias", "weight")
        ),
        "final_outputs": all(
            bool(final_outputs[name]["all_finite"])
            for name in ("bias_preactivation", "bias", "weight")
        ),
        "gradient_history": all(
            bool(row["gradients"]["all_trainable_gradients_finite"])
            for row in epoch_records
        ),
        "checkpoint": bool(
            args.smoke
            or (
                checkpoint_record
                and checkpoint_record["all_tensors_finite_after_reload"]
            )
        ),
        "held_out": bool(
            args.smoke
            or (
                evaluation
                and all(item["all_finite"] for item in evaluation.values())
            )
        ),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "completed" if completed else "smoke_completed",
        "mode": "complete_100_epoch_seed_run" if completed else "one_epoch_smoke",
        "seed": args.seed,
        "executed_training_epochs": requested_epochs,
        "initial_parameter_sha256": initial_snapshot["initial_parameter_sha256"],
        "initial_parameter_tensor_sha256": initial_snapshot[
            "initial_parameter_tensor_sha256"
        ],
        "initial_output_sha256": initial_snapshot["initial_output_sha256"],
        "configuration_sha256": configuration_sha256,
        "normalization_sha256": normalization["sha256"],
        "training_data_identity_sha256": data_identity["identity_sha256"],
        "controls": {
            "configuration": configuration,
            "normalization": normalization,
            "training_data": data_identity,
        },
        "initial_positive_bias_preactivation_fraction": initial_outputs[
            "bias_preactivation"
        ]["fraction_gt_zero"],
        "initial_output_statistics": initial_outputs,
        "per_epoch_positive_bias_preactivation_fraction": [
            row["positive_bias_preactivation_fraction"] for row in epoch_records
        ],
        "bias_head_gradient_l2_history": [
            row["gradients"]["bias_output_row_l2_norm"] for row in epoch_records
        ],
        "weight_head_gradient_l2_history": [
            row["gradients"]["weight_output_row_l2_norm"] for row in epoch_records
        ],
        "shared_layer_gradient_l2_history": [
            row["gradients"]["shared_layers_l2_norm"] for row in epoch_records
        ],
        "global_gradient_l2_history": [
            row["gradients"]["global_l2_norm"] for row in epoch_records
        ],
        "training_loss": {
            "epoch_1_pre_update_sum_3d_m": losses[0],
            "epoch_100_pre_update_sum_3d_m": losses[-1] if completed else None,
            "minimum_pre_update": {
                "epoch": minimum_index + 1,
                "sum_3d_m": losses[minimum_index],
            },
            "final_post_update_sum_3d_m": final_loss,
        },
        "final_positive_bias_fraction": final_outputs["bias"]["fraction_gt_zero"],
        "final_bias_statistics": final_outputs["bias"],
        "final_weight_statistics": final_outputs["weight"],
        "final_bias_preactivation_statistics": final_outputs[
            "bias_preactivation"
        ],
        "epochs": epoch_records,
        "checkpoint": checkpoint_record,
        "held_out": evaluation,
        "finite": {
            **finite_checks,
            "all_finite": all(finite_checks.values()),
        },
        "duration_seconds": time.perf_counter() - total_start,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"result: {result_path}")
    if checkpoint_record:
        print(f"checkpoint SHA-256: {checkpoint_record['sha256']}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    if not result["finite"]["all_finite"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
