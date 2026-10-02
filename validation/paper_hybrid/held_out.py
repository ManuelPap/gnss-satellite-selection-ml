"""Frozen KLT1/KLT2 evaluation for the released shared TDL-BW model."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pymap3d as p3d
import torch

try:
    from .core import HybridShareNet, solve_paper_hybrid_position
except ImportError:
    from core import HybridShareNet, solve_paper_hybrid_position

from validation.paper_weightnet.held_out import (
    DATASET_SPECS,
    FROZEN_MODEL_MEAN,
    FROZEN_MODEL_STD,
    KLT3_FEATURE_MEAN,
    KLT3_FEATURE_POPULATION_STD,
    PreparedDataset,
    PreparedEpoch,
    common_input_arguments,
    historical_position_error,
    inputs_from_args,
    prepare_dataset,
    repository_root,
    resolve_input_paths,
    sha256,
)


TDL_COMMIT = "dd5eac669676ba0a922102047e58c2dfc9be9267"
PAPER_HYBRID_METRICS = {
    "KLT1": {"mean_2d_error_m": 1.84, "mean_3d_error_m": 4.72},
    "KLT2": {"mean_2d_error_m": 1.86, "mean_3d_error_m": 3.92},
}


def checkpoint_sha256_from_metrics(metrics_path: Path) -> str:
    metrics = json.loads(metrics_path.resolve().read_text())
    checkpoint = metrics.get("checkpoint")
    if not checkpoint or not checkpoint.get("sha256"):
        raise RuntimeError("training metrics do not record a full-training checkpoint")
    return str(checkpoint["sha256"])


def verify_checkpoint(path: Path, metrics_path: Path) -> str:
    if not path.resolve().is_file():
        raise FileNotFoundError(f"frozen hybrid checkpoint not found: {path.resolve()}")
    expected = checkpoint_sha256_from_metrics(metrics_path)
    actual = sha256(path.resolve())
    if actual != expected:
        raise RuntimeError(
            f"hybrid checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_frozen_hybrid(checkpoint: Path, metrics_path: Path) -> HybridShareNet:
    verify_checkpoint(checkpoint, metrics_path)
    model = HybridShareNet()
    model.double()
    model.load_state_dict(
        torch.load(checkpoint.resolve(), map_location="cpu", weights_only=True)
    )
    model.eval()
    np.testing.assert_array_equal(
        model.seq[0].mean.detach().numpy(), FROZEN_MODEL_MEAN
    )
    np.testing.assert_array_equal(
        model.seq[0].std.detach().numpy(), FROZEN_MODEL_STD
    )
    return model


def normalize_with_frozen_klt3(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).astype(np.float64)
    return (values - FROZEN_MODEL_MEAN) / FROZEN_MODEL_STD


def infer_hybrid(
    model: HybridShareNet, features: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        weight, bias = model(torch.as_tensor(features, dtype=torch.float32))
    expected = (features.shape[0],)
    if tuple(weight.shape) != expected or tuple(bias.shape) != expected:
        raise RuntimeError("hybrid output rows do not match satellite rows")
    if weight.requires_grad or bias.requires_grad:
        raise RuntimeError("test-time hybrid outputs unexpectedly require gradients")
    if not bool(torch.all(torch.isfinite(weight)) and torch.all(torch.isfinite(bias))):
        raise RuntimeError("hybrid model emitted a non-finite output")
    if not bool(torch.all((weight > 0.0) & (weight < 1.0))):
        raise RuntimeError("finite sigmoid hybrid weights must be in (0,1)")
    if not bool(torch.all(bias >= 0.0)):
        raise RuntimeError("released ReLU hybrid bias must be nonnegative")
    return weight, bias


@dataclass(frozen=True)
class EpochEvaluation:
    prepared: PreparedEpoch
    normalized_features: np.ndarray
    predicted_bias_m: np.ndarray
    bias_corrected_pseudorange_m: np.ndarray
    weights: np.ndarray
    solution: object
    estimated_ecef_m: np.ndarray
    estimated_geodetic_deg_m: np.ndarray
    ground_truth_ecef_m: np.ndarray
    enu_error_m: np.ndarray
    error_2d_m: float
    error_3d_m: float
    ols_error_2d_m: float
    ols_error_3d_m: float
    historical_wls_status: str
    historical_wls_residual_norm_m: float
    normal_matrix_rank: int
    normal_matrix_condition_number: float
    material_constellation_counts: dict[str, int]


def evaluate_epoch(model: HybridShareNet, epoch: PreparedEpoch) -> EpochEvaluation:
    normalized = normalize_with_frozen_klt3(epoch.features)
    weight_t, bias_t = infer_hybrid(model, epoch.features)
    solution = solve_paper_hybrid_position(
        epoch.satellite_positions_ecef_m,
        epoch.satellite_clock_bias_s,
        epoch.corrected_pseudorange_m,
        epoch.system_clock_indices,
        weight_t,
        bias_t,
        epoch.initial_ols_state,
        return_trace=True,
    )
    estimated_ecef = solution.state[:3].detach().numpy().copy()
    if not np.all(np.isfinite(estimated_ecef)):
        raise RuntimeError(f"non-finite WLS state at {epoch.epoch_time:.9f}")
    estimated_geodetic, enu, error_2d, error_3d = historical_position_error(
        estimated_ecef, epoch.ground_truth_geodetic_deg_m
    )
    _, _ols_enu, ols_2d, ols_3d = historical_position_error(
        epoch.initial_ols_state[:3], epoch.ground_truth_geodetic_deg_m
    )
    ground_truth_ecef = np.asarray(
        p3d.geodetic2ecef(*epoch.ground_truth_geodetic_deg_m), dtype=np.float64
    )
    if solution.iterations:
        final_iteration = solution.iterations[-1]
        residual_norm = float(torch.linalg.vector_norm(final_iteration.residual_v_m))
        normal = final_iteration.normal_matrix_HTWH.detach().numpy()
        rank = int(np.linalg.matrix_rank(normal))
        condition = float(np.linalg.cond(normal))
    else:
        residual_norm = float("nan")
        rank = 0
        condition = float("inf")
    if not solution.converged:
        status = "failed_maximum_iterations_included_by_historical_predictor"
    elif residual_norm > 1000.0:
        status = "failed_residual_norm_included_by_historical_predictor"
    else:
        status = "converged"
    weights = weight_t.numpy().copy()
    constellations: dict[str, int] = {}
    for satellite, weight in zip(epoch.satellite_ids, weights, strict=True):
        if weight > 0.01:
            system = str(satellite)[0]
            constellations[system] = constellations.get(system, 0) + 1
    return EpochEvaluation(
        prepared=epoch,
        normalized_features=normalized,
        predicted_bias_m=bias_t.numpy().copy(),
        bias_corrected_pseudorange_m=solution.bias_corrected_pseudorange_m.detach().numpy().copy(),
        weights=weights,
        solution=solution,
        estimated_ecef_m=estimated_ecef,
        estimated_geodetic_deg_m=estimated_geodetic,
        ground_truth_ecef_m=ground_truth_ecef,
        enu_error_m=enu,
        error_2d_m=error_2d,
        error_3d_m=error_3d,
        ols_error_2d_m=ols_2d,
        ols_error_3d_m=ols_3d,
        historical_wls_status=status,
        historical_wls_residual_norm_m=residual_norm,
        normal_matrix_rank=rank,
        normal_matrix_condition_number=condition,
        material_constellation_counts=constellations,
    )


def evaluate_prepared_dataset(
    model: HybridShareNet, prepared: PreparedDataset
) -> list[EpochEvaluation]:
    return [evaluate_epoch(model, epoch) for epoch in prepared.epochs]


CSV_FIELDS = (
    "dataset", "epoch_index", "candidate_epoch_index", "split_epoch_index",
    "gnss_timestamp", "gt_timestamp", "gt_time_difference_s",
    "number_of_satellites", "number_weight_gt_0.01", "number_weight_gt_0.1",
    "number_weight_gt_0.5", "material_constellation_counts_json",
    "normal_matrix_rank", "normal_matrix_condition_number",
    "estimated_ecef_x_m", "estimated_ecef_y_m", "estimated_ecef_z_m",
    "ground_truth_ecef_x_m", "ground_truth_ecef_y_m", "ground_truth_ecef_z_m",
    "east_error_m", "north_error_m", "up_error_m", "error_2d_m", "error_3d_m",
    "ols_error_2d_m", "ols_error_3d_m", "wls_convergence_status",
    "wls_iterations", "wls_last_residual_norm_m",
)


def evaluation_csv_row(item: EpochEvaluation, dataset_name: str) -> dict[str, object]:
    epoch = item.prepared
    return {
        "dataset": dataset_name,
        "epoch_index": epoch.valid_epoch_index,
        "candidate_epoch_index": epoch.candidate_epoch_index,
        "split_epoch_index": epoch.split_epoch_index,
        "gnss_timestamp": f"{epoch.epoch_time:.9f}",
        "gt_timestamp": f"{epoch.gt_time:.9f}",
        "gt_time_difference_s": f"{epoch.gt_time_difference_s:.12g}",
        "number_of_satellites": epoch.features.shape[0],
        "number_weight_gt_0.01": int(np.count_nonzero(item.weights > 0.01)),
        "number_weight_gt_0.1": int(np.count_nonzero(item.weights > 0.1)),
        "number_weight_gt_0.5": int(np.count_nonzero(item.weights > 0.5)),
        "material_constellation_counts_json": json.dumps(item.material_constellation_counts, sort_keys=True),
        "normal_matrix_rank": item.normal_matrix_rank,
        "normal_matrix_condition_number": f"{item.normal_matrix_condition_number:.15g}",
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
            writer.writerow(evaluation_csv_row(item, dataset_name))


def distribution(values: np.ndarray, unit: str) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "unit": unit,
        "minimum": float(values.min()), "maximum": float(values.max()),
        "mean": float(values.mean()), "median": float(np.median(values)),
        "population_std": float(values.std()),
        "p5": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
        "all_finite": bool(np.all(np.isfinite(values))),
    }


def error_diagnostics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "rms_m": float(np.sqrt(np.mean(values**2))),
        "p68_m": float(np.percentile(values, 68.0)),
        "p95_m": float(np.percentile(values, 95.0)),
    }


def difference(reproduction: float, paper: float) -> dict[str, float]:
    delta = reproduction - paper
    return {
        "reproduction_minus_paper_m": delta,
        "absolute_difference_m": abs(delta),
        "relative_difference_percent": delta / paper * 100.0,
        "absolute_relative_difference_percent": abs(delta) / paper * 100.0,
    }


def representative_epoch(item: EpochEvaluation) -> dict[str, object]:
    epoch = item.prepared
    rows = []
    for index, satellite in enumerate(epoch.satellite_ids):
        rows.append(
            {
                "PRN": str(satellite),
                "features": epoch.features[index].tolist(),
                "normalized_features": item.normalized_features[index].tolist(),
                "bias_m": float(item.predicted_bias_m[index]),
                "rtklib_corrected_pseudorange_m": float(epoch.corrected_pseudorange_m[index]),
                "bias_corrected_pseudorange_m": float(item.bias_corrected_pseudorange_m[index]),
                "weight": float(item.weights[index]),
            }
        )
    return {
        "epoch_index": epoch.valid_epoch_index,
        "gnss_timestamp": epoch.epoch_time,
        "rows": rows,
    }


def summarize_results(
    prepared: PreparedDataset,
    evaluations: Sequence[EpochEvaluation],
    *,
    csv_path: Path,
    checkpoint: Path,
    training_metrics: Path,
) -> dict[str, object]:
    errors_2d = np.asarray([item.error_2d_m for item in evaluations])
    errors_3d = np.asarray([item.error_3d_m for item in evaluations])
    ols_2d = np.asarray([item.ols_error_2d_m for item in evaluations])
    ols_3d = np.asarray([item.ols_error_3d_m for item in evaluations])
    biases = np.concatenate([item.predicted_bias_m for item in evaluations])
    weights = np.concatenate([item.weights for item in evaluations])
    paper = PAPER_HYBRID_METRICS[prepared.spec.name]
    mean_2d = float(errors_2d.mean())
    mean_3d = float(errors_3d.mean())
    failures = [
        {"epoch_index": item.prepared.valid_epoch_index, "status": item.historical_wls_status}
        for item in evaluations if item.historical_wls_status != "converged"
    ]
    geometry = [
        {
            "epoch_index": item.prepared.valid_epoch_index,
            "gnss_timestamp": item.prepared.epoch_time,
            "total_valid_satellites": int(item.weights.size),
            "number_weight_gt_0.01": int(np.count_nonzero(item.weights > 0.01)),
            "number_weight_gt_0.1": int(np.count_nonzero(item.weights > 0.1)),
            "number_weight_gt_0.5": int(np.count_nonzero(item.weights > 0.5)),
            "material_constellation_counts_weight_gt_0.01": item.material_constellation_counts,
            "normal_matrix_rank": item.normal_matrix_rank,
            "normal_matrix_condition_number": item.normal_matrix_condition_number,
        }
        for item in evaluations
    ]
    representative_indices = sorted({0, len(evaluations) // 2, len(evaluations) - 1})
    return {
        "status": "passed",
        "dataset": prepared.spec.name,
        "reference_commit": TDL_COMMIT,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": verify_checkpoint(checkpoint, training_metrics),
        },
        "model_eval": True,
        "torch_no_grad": True,
        "model_parameters_frozen_during_inference": True,
        "cardinality": {
            "raw_candidate_epochs_in_strict_interval": prepared.candidate_epoch_count,
            "released_code_valid_evaluation_epochs": len(prepared.epochs),
            "retained_satellite_measurements": prepared.measurement_count,
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
        "bias_outputs": distribution(biases, "metre"),
        "weight_outputs": {
            **distribution(weights, "dimensionless relative WLS coefficient"),
            "fraction_weight_lt_1e-5": float(np.mean(weights < 1.0e-5)),
            "fraction_weight_lt_0.01": float(np.mean(weights < 0.01)),
            "fraction_weight_gt_0.5": float(np.mean(weights > 0.5)),
            "fraction_weight_gt_0.99": float(np.mean(weights > 0.99)),
        },
        "tdl_bw": {
            "mean_2d_error_m": mean_2d,
            "mean_3d_error_m": mean_3d,
            "paper_mean_2d_error_m": paper["mean_2d_error_m"],
            "paper_mean_3d_error_m": paper["mean_3d_error_m"],
            "difference_2d": difference(mean_2d, paper["mean_2d_error_m"]),
            "difference_3d": difference(mean_3d, paper["mean_3d_error_m"]),
            "error_2d_diagnostics": error_diagnostics(errors_2d),
            "error_3d_diagnostics": error_diagnostics(errors_3d),
        },
        "equal_weight_ols_sanity_baseline": {
            "mean_2d_error_m": float(ols_2d.mean()),
            "mean_3d_error_m": float(ols_3d.mean()),
        },
        "effective_geometry_per_epoch": geometry,
        "representative_epochs": [
            representative_epoch(evaluations[index]) for index in representative_indices
        ],
        "wls_failures_included_by_released_predictor": failures,
        "aggregation": {
            "formula_2d": "mean_i(sqrt(E_i^2 + N_i^2))",
            "formula_3d": "mean_i(sqrt(E_i^2 + N_i^2 + U_i^2))",
        },
        "input_provenance": prepared.input_provenance,
        "per_epoch_csv": {
            "path": str(csv_path.resolve()),
            "sha256": sha256(csv_path.resolve()),
            "rows": len(evaluations),
        },
    }


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


__all__ = [
    "DATASET_SPECS", "EpochEvaluation", "PAPER_HYBRID_METRICS",
    "common_input_arguments", "evaluate_epoch", "evaluate_prepared_dataset",
    "infer_hybrid", "inputs_from_args", "load_frozen_hybrid",
    "normalize_with_frozen_klt3", "prepare_dataset", "repository_root",
    "resolve_input_paths",
    "summarize_results", "verify_checkpoint", "write_results_csv", "write_summary",
]
