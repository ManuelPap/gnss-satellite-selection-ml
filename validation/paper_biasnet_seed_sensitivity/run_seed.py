#!/usr/bin/env python3
"""Run one corrected-GT BiasNet initialization-seed experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from validation.paper_biasnet.check_exported_metrics import aggregate_csv  # noqa: E402
from validation.ibiza_generalization.runtime_cache import (  # noqa: E402
    DEFAULT_RUNTIME_DIR,
)
from validation.paper_biasnet.held_out import (  # noqa: E402
    DATASET_SPECS,
    evaluate_prepared_dataset,
    prepare_dataset,
    resolve_input_paths,
    write_results_csv,
)
from validation.paper_biasnet_seed_sensitivity.experiment import (  # noqa: E402
    SELECTED_EPOCHS,
    TRAINING_EPOCHS,
    array_sha256,
    build_initial_snapshot,
    canonical_json_hash,
    dataset_output_diagnostics,
    file_sha256,
    full_dataset_loss_with_diagnostics,
    gradient_diagnostics,
    load_dataset,
    model_tensors_finite,
    normalization_identity,
    parameter_change_diagnostics,
    parameter_snapshot,
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
        default=REPOSITORY / "results/paper_biasnet_seed_sensitivity",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPOSITORY / "checkpoints/paper_biasnet_seed_sensitivity",
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
            "Run one KLT3 update and skip checkpoint/KLT1/KLT2; this output is "
            "incomplete and is rejected by scientific aggregation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated result/checkpoint.",
    )
    return parser.parse_args(argv)


def _descriptive_errors(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.all(np.isfinite(values)):
        raise RuntimeError("held-out errors must be finite and nonempty")
    return {
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "rms_m": float(np.sqrt(np.mean(values**2))),
        "p68_m": float(np.percentile(values, 68.0)),
        "p95_m": float(np.percentile(values, 95.0)),
    }


def prepared_dataset_identity(prepared: object) -> dict[str, object]:
    """Hash all retained held-out rows used by the frozen evaluator."""

    epochs = prepared.epochs
    offsets = np.zeros(len(epochs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([epoch.features.shape[0] for epoch in epochs])
    arrays = {
        "epoch_offsets": offsets,
        "epoch_times": np.asarray([epoch.epoch_time for epoch in epochs]),
        "gt_times": np.asarray([epoch.gt_time for epoch in epochs]),
        "features": np.concatenate([epoch.features for epoch in epochs]),
        "satellite_ids": np.concatenate([epoch.satellite_ids for epoch in epochs]),
        "satellite_positions": np.concatenate(
            [epoch.satellite_positions_ecef_m for epoch in epochs]
        ),
        "satellite_clock_bias": np.concatenate(
            [epoch.satellite_clock_bias_s for epoch in epochs]
        ),
        "corrected_pseudorange": np.concatenate(
            [epoch.corrected_pseudorange_m for epoch in epochs]
        ),
        "system_clock_indices": np.concatenate(
            [epoch.system_clock_indices for epoch in epochs]
        ),
        "initial_states": np.stack([epoch.initial_ols_state for epoch in epochs]),
        "ground_truth": np.stack(
            [epoch.ground_truth_geodetic_deg_m for epoch in epochs]
        ),
    }
    array_hashes = {name: array_sha256(value) for name, value in arrays.items()}
    record: dict[str, object] = {
        "dataset": prepared.spec.name,
        "valid_epoch_count": len(epochs),
        "retained_measurement_count": prepared.measurement_count,
        "array_sha256": array_hashes,
    }
    record["identity_sha256"] = canonical_json_hash(record)
    return record


def prepare_held_out(args: argparse.Namespace) -> dict[str, object]:
    prepared: dict[str, object] = {}
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
        prepared[dataset_name] = prepare_dataset(spec, inputs)
    return prepared


def held_out_metrics(
    model: torch.nn.Module, prepared_datasets: dict[str, object]
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Evaluate once and independently re-aggregate temporary evaluator CSVs."""

    model.to("cpu")
    model.eval()
    model.requires_grad_(False)
    metrics: dict[str, dict[str, object]] = {}
    identities: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="biasnet-seed-check-") as directory:
        temporary = Path(directory)
        for dataset_name in ("KLT1", "KLT2"):
            prepared = prepared_datasets[dataset_name]
            evaluations = evaluate_prepared_dataset(model, prepared)
            errors_2d = np.asarray([item.error_2d_m for item in evaluations])
            errors_3d = np.asarray([item.error_3d_m for item in evaluations])
            csv_path = temporary / f"{dataset_name.lower()}_per_epoch.csv"
            write_results_csv(csv_path, dataset_name, evaluations)
            independent = aggregate_csv(csv_path)
            two_dimensional = _descriptive_errors(errors_2d)
            three_dimensional = _descriptive_errors(errors_3d)
            differences = {
                "mean_2d_error_m": abs(
                    float(independent["mean_2d_error_m"])
                    - two_dimensional["mean_m"]
                ),
                "mean_3d_error_m": abs(
                    float(independent["mean_3d_error_m"])
                    - three_dimensional["mean_m"]
                ),
            }
            tolerance = 1.0e-12
            if int(independent["rows"]) != len(evaluations) or any(
                value > tolerance for value in differences.values()
            ):
                raise RuntimeError(
                    f"{dataset_name} independent metric checker disagreement"
                )
            metrics[dataset_name] = {
                "mean_2d_error_m": two_dimensional["mean_m"],
                "mean_3d_error_m": three_dimensional["mean_m"],
                "error_2d": two_dimensional,
                "error_3d": three_dimensional,
                "valid_epoch_count": len(evaluations),
                "retained_measurement_count": prepared.measurement_count,
                "independent_checker": {
                    **independent,
                    "absolute_mean_differences_m": differences,
                    "tolerance_m": tolerance,
                    "agreed": True,
                    "temporary_csv_retained": False,
                },
                "all_finite": bool(
                    np.all(np.isfinite(errors_2d))
                    and np.all(np.isfinite(errors_3d))
                ),
            }
            identities[dataset_name] = prepared_dataset_identity(prepared)
    identity_hashes = {
        name: value["identity_sha256"] for name, value in identities.items()
    }
    held_out_identity = {
        "datasets": identities,
        "identity_sha256": canonical_json_hash(identity_hashes),
    }
    return metrics, held_out_identity


def _target_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    prefix = "smoke_seed" if args.smoke else "seed"
    result_path = args.output_dir.resolve() / f"{prefix}_{args.seed}.json"
    checkpoint_path = None
    if not args.smoke:
        checkpoint_path = (
            args.checkpoint_dir.resolve() / f"seed_{args.seed}_biasnet_3d.pth"
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
        epoch_number = epoch_index + 1
        detailed = epoch_number in SELECTED_EPOCHS
        epoch_start = time.perf_counter()
        before = parameter_snapshot(model)
        optimizer.zero_grad()
        loss, outputs = full_dataset_loss_with_diagnostics(model, dataset, device)
        loss_value = float(loss.detach().cpu())
        loss.backward()
        gradients = gradient_diagnostics(model, detailed=detailed)
        if not gradients["all_trainable_gradients_present"]:
            raise RuntimeError(f"missing gradient at training epoch {epoch_number}")
        if not gradients["all_trainable_gradients_finite"]:
            raise RuntimeError(f"non-finite gradient at training epoch {epoch_number}")
        optimizer.step()
        changes = parameter_change_diagnostics(before, model, detailed=detailed)
        if not model_tensors_finite(model):
            raise RuntimeError(f"non-finite model tensor at epoch {epoch_number}")
        epoch_records.append(
            {
                "epoch": epoch_number,
                "selected_detailed_epoch": detailed,
                "loss_sum_3d_m": loss_value,
                "loss_divided_by_405_m": loss_value / 405.0,
                "activations_and_output": outputs,
                "gradients": gradients,
                "optimization": changes,
                "duration_seconds": time.perf_counter() - epoch_start,
            }
        )
        if args.smoke or epoch_number == 1 or epoch_number % 25 == 0:
            print(
                f"seed {args.seed} epoch {epoch_number:03d}/{requested_epochs}: "
                f"loss={loss_value:.12f}, "
                f"dead-h1={outputs['hidden_layer_1']['dead_neuron_count']}, "
                f"dead-h2={outputs['hidden_layer_2']['dead_neuron_count']}, "
                f"grad={gradients['global_l2_norm']:.6e}",
                flush=True,
            )

    final_loss_tensor, final_outputs = full_dataset_loss_with_diagnostics(
        model, dataset, device, require_gradients=False
    )
    final_loss = float(final_loss_tensor.detach().cpu())
    losses = [float(row["loss_sum_3d_m"]) for row in epoch_records]
    minimum_pre_index = int(np.argmin(losses))
    minimum_value = min(losses[minimum_pre_index], final_loss)
    minimum_stage = (
        "final_post_update"
        if final_loss < losses[minimum_pre_index]
        else "epoch_pre_update"
    )
    minimum_epoch = (
        requested_epochs
        if minimum_stage == "final_post_update"
        else minimum_pre_index + 1
    )

    checkpoint_record: dict[str, object] | None = None
    evaluation: dict[str, dict[str, object]] | None = None
    held_out_identity: dict[str, object] | None = None
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
        prepared = prepare_held_out(args)
        evaluation, held_out_identity = held_out_metrics(cpu_model, prepared)

    completed = not args.smoke
    finite_checks = {
        "epoch_losses": all(math.isfinite(value) for value in losses),
        "final_loss": math.isfinite(final_loss),
        "initial_outputs": all(
            bool(initial_outputs[name]["all_finite"])
            for name in ("hidden_layer_1", "hidden_layer_2", "bias_output")
        ),
        "per_epoch_outputs": all(
            all(
                bool(row["activations_and_output"][name]["all_finite"])
                for name in ("hidden_layer_1", "hidden_layer_2", "bias_output")
            )
            for row in epoch_records
        ),
        "final_outputs": all(
            bool(final_outputs[name]["all_finite"])
            for name in ("hidden_layer_1", "hidden_layer_2", "bias_output")
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
                and all(
                    item["independent_checker"]["agreed"]
                    for item in evaluation.values()
                )
            )
        ),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "completed" if completed else "smoke_completed_incomplete",
        "mode": (
            "complete_500_epoch_seed_run" if completed else "one_epoch_smoke_incomplete"
        ),
        "scientific_aggregate_eligible": completed,
        "seed": args.seed,
        "executed_training_epochs": requested_epochs,
        "initial_parameter_sha256": initial_snapshot["initial_parameter_sha256"],
        "initial_parameter_tensor_sha256": initial_snapshot[
            "initial_parameter_tensor_sha256"
        ],
        "initial_output_sha256": initial_snapshot["initial_output_sha256"],
        "configuration_sha256": configuration_sha256,
        "training_data_identity_sha256": data_identity["identity_sha256"],
        "normalization_sha256": normalization["sha256"],
        "held_out_data_identity_sha256": (
            held_out_identity["identity_sha256"] if held_out_identity else None
        ),
        "controls": {
            "configuration": configuration,
            "training_data": data_identity,
            "normalization": normalization,
            "held_out_data": held_out_identity,
        },
        "initial_diagnostics": {
            "parameter_sha256": initial_snapshot["initial_parameter_sha256"],
            "output_sha256": initial_snapshot["initial_output_sha256"],
            "activations_and_output": initial_outputs,
            "training_objective_sum_3d_m": losses[0],
        },
        "training_loss": {
            "epoch_1_pre_update_sum_3d_m": losses[0],
            "epoch_500_pre_update_sum_3d_m": losses[-1] if completed else None,
            "minimum_pre_update": {
                "epoch": minimum_pre_index + 1,
                "sum_3d_m": losses[minimum_pre_index],
            },
            "final_post_update_sum_3d_m": final_loss,
            "minimum_evaluated": {
                "epoch": minimum_epoch,
                "stage": minimum_stage,
                "sum_3d_m": minimum_value,
            },
            "pre_update_history_sum_3d_m": losses,
        },
        "hidden_layer_1_history": [
            row["activations_and_output"]["hidden_layer_1"] for row in epoch_records
        ],
        "hidden_layer_2_history": [
            row["activations_and_output"]["hidden_layer_2"] for row in epoch_records
        ],
        "bias_output_history": [
            row["activations_and_output"]["bias_output"] for row in epoch_records
        ],
        "gradient_history": [row["gradients"] for row in epoch_records],
        "parameter_change_history": [
            row["optimization"] for row in epoch_records
        ],
        "selected_detailed_epochs": [
            epoch for epoch in SELECTED_EPOCHS if epoch <= requested_epochs
        ],
        "selected_epoch_diagnostics": [
            row for row in epoch_records if row["selected_detailed_epoch"]
        ],
        "epoch_duration_seconds": [
            row["duration_seconds"] for row in epoch_records
        ],
        "all_epochs_every_trainable_tensor_received_gradient": all(
            row["gradients"]["all_trainable_gradients_present"]
            for row in epoch_records
        ),
        "all_epochs_every_expected_trainable_tensor_changed": all(
            row["optimization"]["all_expected_trainable_tensors_changed"]
            for row in epoch_records
        ),
        "final_diagnostics": final_outputs,
        "checkpoint": checkpoint_record,
        "held_out": evaluation,
        "finite": {**finite_checks, "all_finite": all(finite_checks.values())},
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
    return 0 if result["finite"]["all_finite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
