"""Held-out evaluation helpers for the released paper-era standalone BiasNet."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pymap3d as p3d
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from validation.ibiza_generalization.runtime_cache import (  # noqa: E402
    DEFAULT_RUNTIME_DIR,
)

try:
    from .core import BiasNet, solve_paper_bias_position
except ImportError:
    from core import BiasNet, solve_paper_bias_position

from validation.paper_weightnet.held_out import (  # noqa: E402
    KLT3_FEATURE_MEAN,
    KLT3_FEATURE_POPULATION_STD,
    DatasetSpec,
    InputPaths,
    PreparedDataset,
    PreparedEpoch,
    historical_position_error,
    prepare_dataset,
    resolve_input_paths,
)


TDL_COMMIT = "dd5eac669676ba0a922102047e58c2dfc9be9267"
FROZEN_MODEL_MEAN = KLT3_FEATURE_MEAN.astype(np.float32).astype(np.float64)
FROZEN_MODEL_STD = KLT3_FEATURE_POPULATION_STD.astype(np.float32).astype(np.float64)

DATASET_SPECS = {
    "KLT1": DatasetSpec(
        name="KLT1",
        start_time=1623296154.0,
        end_time=1623296357.0,
        observation_relative="0610_KLT/COM38_210610_025603.obs",
        ephemeris_relative=("0610_KLT/sta/hksc161d.21*",),
        ground_truth_relative="0610_KLT/20210610_100.txt",
        published_epochs=203,
        published_measurements=4676,
        paper_2d_mean_m=2.24,
        paper_3d_mean_m=5.30,
    ),
    "KLT2": DatasetSpec(
        name="KLT2",
        start_time=1623296917.0,
        end_time=1623297126.0,
        observation_relative="0610_KLT/COM38_210610_025603.obs",
        ephemeris_relative=("0610_KLT/sta/hksc161d.21*",),
        ground_truth_relative="0610_KLT/20210610_100.txt",
        published_epochs=209,
        published_measurements=4914,
        paper_2d_mean_m=2.35,
        paper_3d_mean_m=5.89,
    ),
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_training_metrics() -> Path:
    return Path(__file__).resolve().parent / "training_metrics.json"


def verify_checkpoint(
    checkpoint: Path, training_metrics: Path | None = None
) -> str:
    checkpoint = checkpoint.resolve()
    metrics_path = (training_metrics or default_training_metrics()).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen checkpoint not found: {checkpoint}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"training metrics not found: {metrics_path}")
    record = json.loads(metrics_path.read_text())
    expected = record["checkpoint"]["sha256"]
    actual = sha256(checkpoint)
    if actual != expected:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_frozen_biasnet(
    checkpoint: Path, training_metrics: Path | None = None
) -> BiasNet:
    """Run the released default-model/double/load sequence, then freeze it."""

    verify_checkpoint(checkpoint, training_metrics)
    model = BiasNet()
    model.double()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    np.testing.assert_array_equal(
        model.seq[0].mean.detach().numpy(), FROZEN_MODEL_MEAN
    )
    np.testing.assert_array_equal(model.seq[0].std.detach().numpy(), FROZEN_MODEL_STD)
    return model


def normalize_with_frozen_klt3(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).astype(np.float64)
    return (values - FROZEN_MODEL_MEAN) / FROZEN_MODEL_STD


def infer_biases(model: BiasNet, features: np.ndarray) -> torch.Tensor:
    """Infer aligned metre-valued corrections without an autograd graph."""

    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32)
    with torch.no_grad():
        biases = model(tensor).squeeze(-1)
    if biases.shape != (features.shape[0],):
        raise RuntimeError("BiasNet output rows do not match satellite rows")
    if biases.requires_grad or biases.grad_fn is not None:
        raise RuntimeError("test-time biases unexpectedly require gradients")
    if not bool(torch.all(torch.isfinite(biases))):
        raise RuntimeError("BiasNet emitted a non-finite correction")
    return biases


@dataclass(frozen=True)
class EpochEvaluation:
    prepared: PreparedEpoch
    normalized_features: np.ndarray
    predicted_bias_m: np.ndarray
    bias_corrected_pseudorange_m: np.ndarray
    solution: object
    estimated_ecef_m: np.ndarray
    estimated_geodetic_deg_m: np.ndarray
    ground_truth_ecef_m: np.ndarray
    enu_error_m: np.ndarray
    error_2d_m: float
    error_3d_m: float
    ols_enu_error_m: np.ndarray
    ols_error_2d_m: float
    ols_error_3d_m: float
    historical_wls_status: str
    historical_wls_residual_norm_m: float


def evaluate_epoch(model: BiasNet, epoch: PreparedEpoch) -> EpochEvaluation:
    normalized = normalize_with_frozen_klt3(epoch.features)
    biases = infer_biases(model, epoch.features)
    solution = solve_paper_bias_position(
        epoch.satellite_positions_ecef_m,
        epoch.satellite_clock_bias_s,
        epoch.corrected_pseudorange_m,
        epoch.system_clock_indices,
        biases,
        epoch.initial_ols_state,
        return_trace=True,
    )
    estimated_ecef = solution.state[:3].detach().numpy().copy()
    if not np.all(np.isfinite(estimated_ecef)):
        raise RuntimeError(f"non-finite WLS state at {epoch.epoch_time:.9f}")
    estimated_geodetic, enu, error_2d, error_3d = historical_position_error(
        estimated_ecef, epoch.ground_truth_geodetic_deg_m
    )
    _, ols_enu, ols_2d, ols_3d = historical_position_error(
        epoch.initial_ols_state[:3], epoch.ground_truth_geodetic_deg_m
    )
    ground_truth_ecef = np.asarray(
        p3d.geodetic2ecef(*epoch.ground_truth_geodetic_deg_m), dtype=np.float64
    )
    residual_norm = (
        float(torch.linalg.vector_norm(solution.iterations[-1].residual_v_m))
        if solution.iterations
        else float("nan")
    )
    if not solution.converged:
        status = "failed_maximum_iterations_included_by_historical_predictor"
    elif residual_norm > 1000.0:
        status = "failed_residual_norm_included_by_historical_predictor"
    else:
        status = "converged"
    return EpochEvaluation(
        prepared=epoch,
        normalized_features=normalized,
        predicted_bias_m=biases.numpy().copy(),
        bias_corrected_pseudorange_m=(
            solution.bias_corrected_pseudorange_m.detach().numpy().copy()
        ),
        solution=solution,
        estimated_ecef_m=estimated_ecef,
        estimated_geodetic_deg_m=estimated_geodetic,
        ground_truth_ecef_m=ground_truth_ecef,
        enu_error_m=enu,
        error_2d_m=error_2d,
        error_3d_m=error_3d,
        ols_enu_error_m=ols_enu,
        ols_error_2d_m=ols_2d,
        ols_error_3d_m=ols_3d,
        historical_wls_status=status,
        historical_wls_residual_norm_m=residual_norm,
    )


def evaluate_prepared_dataset(
    model: BiasNet, prepared: PreparedDataset
) -> list[EpochEvaluation]:
    return [evaluate_epoch(model, epoch) for epoch in prepared.epochs]


CSV_FIELDS = (
    "dataset",
    "epoch_index",
    "candidate_epoch_index",
    "split_epoch_index",
    "gnss_timestamp",
    "gt_timestamp",
    "gt_time_difference_s",
    "number_of_satellites",
    "estimated_ecef_x_m",
    "estimated_ecef_y_m",
    "estimated_ecef_z_m",
    "ground_truth_ecef_x_m",
    "ground_truth_ecef_y_m",
    "ground_truth_ecef_z_m",
    "east_error_m",
    "north_error_m",
    "up_error_m",
    "error_2d_m",
    "error_3d_m",
    "ols_error_2d_m",
    "ols_error_3d_m",
    "wls_convergence_status",
    "wls_iterations",
    "wls_last_residual_norm_m",
)


def evaluation_csv_row(item: EpochEvaluation) -> dict[str, object]:
    epoch = item.prepared
    return {
        "dataset": "pending",
        "epoch_index": epoch.valid_epoch_index,
        "candidate_epoch_index": epoch.candidate_epoch_index,
        "split_epoch_index": epoch.split_epoch_index,
        "gnss_timestamp": f"{epoch.epoch_time:.9f}",
        "gt_timestamp": f"{epoch.gt_time:.9f}",
        "gt_time_difference_s": f"{epoch.gt_time_difference_s:.12g}",
        "number_of_satellites": epoch.features.shape[0],
        "estimated_ecef_x_m": f"{item.estimated_ecef_m[0]:.15g}",
        "estimated_ecef_y_m": f"{item.estimated_ecef_m[1]:.15g}",
        "estimated_ecef_z_m": f"{item.estimated_ecef_m[2]:.15g}",
        "ground_truth_ecef_x_m": f"{item.ground_truth_ecef_m[0]:.15g}",
        "ground_truth_ecef_y_m": f"{item.ground_truth_ecef_m[1]:.15g}",
        "ground_truth_ecef_z_m": f"{item.ground_truth_ecef_m[2]:.15g}",
        "east_error_m": f"{item.enu_error_m[0]:.15g}",
        "north_error_m": f"{item.enu_error_m[1]:.15g}",
        "up_error_m": f"{item.enu_error_m[2]:.15g}",
        "error_2d_m": f"{item.error_2d_m:.15g}",
        "error_3d_m": f"{item.error_3d_m:.15g}",
        "ols_error_2d_m": f"{item.ols_error_2d_m:.15g}",
        "ols_error_3d_m": f"{item.ols_error_3d_m:.15g}",
        "wls_convergence_status": item.historical_wls_status,
        "wls_iterations": len(item.solution.iterations),
        "wls_last_residual_norm_m": f"{item.historical_wls_residual_norm_m:.15g}",
    }


def write_results_csv(
    path: Path, dataset_name: str, evaluations: Sequence[EpochEvaluation]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in evaluations:
            row = evaluation_csv_row(item)
            row["dataset"] = dataset_name
            writer.writerow(row)


def _difference(reproduction: float, paper: float) -> dict[str, float]:
    difference = reproduction - paper
    return {
        "reproduction_minus_paper_m": difference,
        "absolute_difference_m": abs(difference),
        "relative_difference_percent": difference / paper * 100.0,
        "absolute_relative_difference_percent": abs(difference) / paper * 100.0,
    }


def bias_diagnostics(evaluations: Sequence[EpochEvaluation]) -> dict[str, object]:
    biases = np.concatenate([item.predicted_bias_m for item in evaluations])
    records: list[dict[str, object]] = []
    for item in evaluations:
        for row, value in enumerate(item.predicted_bias_m):
            maximum_abs_z = float(np.max(np.abs(item.normalized_features[row])))
            records.append(
                {
                    "dataset_epoch_index": item.prepared.valid_epoch_index,
                    "gnss_timestamp": item.prepared.epoch_time,
                    "satellite": str(item.prepared.satellite_ids[row]),
                    "features": item.prepared.features[row].tolist(),
                    "normalized_features": item.normalized_features[row].tolist(),
                    "maximum_absolute_feature_z_score": maximum_abs_z,
                    "has_feature_beyond_3_population_std": maximum_abs_z > 3.0,
                    "predicted_bias_m": float(value),
                }
            )
    records.sort(key=lambda item: abs(float(item["predicted_bias_m"])), reverse=True)
    return {
        "minimum_m": float(biases.min()),
        "maximum_m": float(biases.max()),
        "mean_m": float(biases.mean()),
        "median_m": float(np.median(biases)),
        "population_std_m": float(biases.std()),
        "p5_m": float(np.percentile(biases, 5.0)),
        "p95_m": float(np.percentile(biases, 95.0)),
        "all_finite": bool(np.all(np.isfinite(biases))),
        "clipping_applied": False,
        "largest_absolute_outputs": records[:10],
    }


def summarize_results(
    prepared: PreparedDataset,
    evaluations: Sequence[EpochEvaluation],
    *,
    csv_path: Path,
    checkpoint: Path,
    training_metrics: Path | None = None,
) -> dict[str, object]:
    errors_2d = np.asarray([item.error_2d_m for item in evaluations])
    errors_3d = np.asarray([item.error_3d_m for item in evaluations])
    ols_2d = np.asarray([item.ols_error_2d_m for item in evaluations])
    ols_3d = np.asarray([item.ols_error_3d_m for item in evaluations])
    mean_2d = float(errors_2d.mean())
    mean_3d = float(errors_3d.mean())
    failures = [
        {
            "epoch_index": item.prepared.valid_epoch_index,
            "timestamp": item.prepared.epoch_time,
            "status": item.historical_wls_status,
            "iterations": len(item.solution.iterations),
            "last_residual_norm_m": item.historical_wls_residual_norm_m,
        }
        for item in evaluations
        if item.historical_wls_status != "converged"
    ]
    return {
        "status": "passed",
        "dataset": prepared.spec.name,
        "reference_commit": TDL_COMMIT,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": verify_checkpoint(checkpoint, training_metrics),
        },
        "cardinality": {
            "raw_candidate_epochs_in_strict_interval": prepared.candidate_epoch_count,
            "released_code_valid_evaluation_epochs": len(prepared.epochs),
            "retained_satellite_measurements": prepared.measurement_count,
            "first_retained_timestamp": prepared.epochs[0].epoch_time,
            "last_retained_timestamp": prepared.epochs[-1].epoch_time,
            "published_epochs": prepared.spec.published_epochs,
            "published_satellite_measurements": prepared.spec.published_measurements,
            "invalid_ols_epochs": list(prepared.invalid_epochs),
        },
        "normalization": {
            "klt3_float64_source_mean": KLT3_FEATURE_MEAN.tolist(),
            "klt3_float64_source_population_std": KLT3_FEATURE_POPULATION_STD.tolist(),
            "checkpoint_stored_mean_after_float32_then_double": FROZEN_MODEL_MEAN.tolist(),
            "checkpoint_stored_std_after_float32_then_double": FROZEN_MODEL_STD.tolist(),
            "test_dataset_statistics_used": False,
        },
        "bias_outputs": bias_diagnostics(evaluations),
        "tdl_b": {
            "mean_2d_error_m": mean_2d,
            "mean_3d_error_m": mean_3d,
            "paper_mean_2d_error_m": prepared.spec.paper_2d_mean_m,
            "paper_mean_3d_error_m": prepared.spec.paper_3d_mean_m,
            "difference_2d": _difference(mean_2d, prepared.spec.paper_2d_mean_m),
            "difference_3d": _difference(mean_3d, prepared.spec.paper_3d_mean_m),
        },
        "equal_weight_ols_sanity_baseline": {
            "mean_2d_error_m": float(ols_2d.mean()),
            "mean_3d_error_m": float(ols_3d.mean()),
        },
        "wls_failures_included_by_released_predictor": failures,
        "aggregation": {
            "formula_2d": "mean_i(sqrt(E_i^2 + N_i^2))",
            "formula_3d": "mean_i(sqrt(E_i^2 + N_i^2 + U_i^2))",
        },
        "per_epoch_csv": {
            "path": str(csv_path.resolve()),
            "sha256": sha256(csv_path),
            "rows": len(evaluations),
        },
        "input_provenance": prepared.input_provenance,
    }


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def common_input_arguments(parser: object) -> None:
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_SPECS))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--ephemeris-glob", action="append", dest="ephemeris_patterns")
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)


def inputs_from_args(args: object) -> tuple[DatasetSpec, InputPaths]:
    spec = DATASET_SPECS[args.dataset]
    return spec, resolve_input_paths(
        spec,
        data_root=args.data_root,
        observation=args.observation,
        ephemeris_patterns=args.ephemeris_patterns,
        ground_truth=args.ground_truth,
        runtime_dir=args.runtime_dir,
    )


__all__ = [
    "CSV_FIELDS",
    "DATASET_SPECS",
    "EpochEvaluation",
    "FROZEN_MODEL_MEAN",
    "FROZEN_MODEL_STD",
    "KLT3_FEATURE_MEAN",
    "KLT3_FEATURE_POPULATION_STD",
    "bias_diagnostics",
    "common_input_arguments",
    "evaluate_epoch",
    "evaluate_prepared_dataset",
    "historical_position_error",
    "infer_biases",
    "inputs_from_args",
    "load_frozen_biasnet",
    "normalize_with_frozen_klt3",
    "prepare_dataset",
    "repository_root",
    "resolve_input_paths",
    "sha256",
    "summarize_results",
    "verify_checkpoint",
    "write_results_csv",
    "write_summary",
]
