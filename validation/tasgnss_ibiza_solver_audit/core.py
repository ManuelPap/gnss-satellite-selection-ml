"""Pure helpers for the TASGNSS Ibiza comparison.

This module deliberately has no TASGNSS, pyrtklib, Torch, checkpoint, or
ground-truth side effects.  The executable runner imports the external stack
only after selecting and validating its read-only source/runtime paths.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pymap3d as p3d


IBIZA_NPZ_SHA256 = "edb0189e9eadf3266d75984e3041a90306dd44b3ebecc0101eb188f933dc88c5"
PAIR_EQUAL_TOLERANCE_M = 1.0e-12
REQUIRED_PHASE_A_ARRAYS = frozenset(
    {
        "corrected_pseudorange_m",
        "satellite_position_ecef_m",
        "satellite_clock_bias_s",
        "epoch_ols_initial_state",
        "system_clock_index",
        "epoch_offsets",
        "epoch_split_index",
        "epoch_time_gpst_like_s",
    }
)
MISSING_FIXED_CORRECTION_ARRAYS = (
    "tasgnss_broadcast_ionosphere_m",
    "tasgnss_saastamoinen_troposphere_m",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_a_compatibility(array_names: Iterable[str]) -> dict[str, object]:
    """Evaluate the strict same-input gate without reconstructing corrections."""

    names = frozenset(array_names)
    missing_core = sorted(REQUIRED_PHASE_A_ARRAYS - names)
    missing_corrections = [name for name in MISSING_FIXED_CORRECTION_ARRAYS if name not in names]
    passed = not missing_core and not missing_corrections
    return {
        "passed": passed,
        "decision": "proceed_phase_a" if passed else "stop_phase_a",
        "missing_core_arrays": missing_core,
        "missing_fixed_correction_arrays": missing_corrections,
        "sagnac_exactly_derivable_from_frozen_geometry": True,
        "sagnac_derivation_permitted_for_gate": True,
        "reason": (
            "All required quantities are frozen."
            if passed
            else (
                "The frozen artifact does not contain the current TASGNSS fixed "
                "atmospheric-correction inputs. Current TASGNSS obtains them from "
                "the original navigation data at a separate RTKLIB pntpos receiver "
                "position; recomputing them would create a new preprocessing result, "
                "not recover a same input."
            )
        ),
    }


def trusted_reference_ecef(reference_document: Mapping[str, object]) -> np.ndarray:
    values = reference_document["trusted_reference_ecef_m"]
    if not isinstance(values, Mapping):
        raise TypeError("trusted_reference_ecef_m must be an object")
    result = np.asarray([values["x"], values["y"], values["z"]], dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("trusted ECEF reference is invalid")
    return result


def enu_errors(estimated_ecef_m: np.ndarray, reference_ecef_m: np.ndarray) -> np.ndarray:
    estimates = np.asarray(estimated_ecef_m, dtype=np.float64)
    reference = np.asarray(reference_ecef_m, dtype=np.float64)
    if estimates.ndim != 2 or estimates.shape[1] != 3:
        raise ValueError("estimated ECEF positions must have shape (epochs, 3)")
    if reference.shape != (3,):
        raise ValueError("reference ECEF position must have shape (3,)")
    latitude_deg, longitude_deg, height_m = p3d.ecef2geodetic(*reference)
    east, north, up = p3d.ecef2enu(
        estimates[:, 0],
        estimates[:, 1],
        estimates[:, 2],
        latitude_deg,
        longitude_deg,
        height_m,
    )
    return np.column_stack((east, north, up)).astype(np.float64, copy=False)


def error_components(enu_m: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(enu_m, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("ENU errors must have shape (epochs, 3)")
    return {
        "east": values[:, 0],
        "north": values[:, 1],
        "up": values[:, 2],
        "e2d": np.linalg.norm(values[:, :2], axis=1),
        "abs_up": np.abs(values[:, 2]),
        "e3d": np.linalg.norm(values, axis=1),
    }


def accuracy_summary(enu_m: np.ndarray, *, total_epochs: int) -> dict[str, float | int]:
    components = error_components(enu_m)
    solved = int(enu_m.shape[0])
    summary: dict[str, float | int] = {
        "solved_epochs": solved,
        "failed_epochs": int(total_epochs - solved),
        "total_epochs": int(total_epochs),
    }
    for axis in ("east", "north", "up"):
        values = components[axis]
        summary[f"{axis}_mean_m"] = float(np.mean(values))
        summary[f"{axis}_rms_m"] = float(np.sqrt(np.mean(np.square(values))))
    for norm in ("e2d", "e3d"):
        values = components[norm]
        summary[f"{norm}_mean_m"] = float(np.mean(values))
        summary[f"{norm}_median_m"] = float(np.median(values))
        summary[f"{norm}_rms_m"] = float(np.sqrt(np.mean(np.square(values))))
        summary[f"{norm}_p68_m"] = float(np.quantile(values, 0.68, method="linear"))
        summary[f"{norm}_p95_m"] = float(np.quantile(values, 0.95, method="linear"))
        summary[f"{norm}_max_m"] = float(np.max(values))
    return summary


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(data.size),
        "mean_m": float(np.mean(data)),
        "median_m": float(np.median(data)),
        "p68_m": float(np.quantile(data, 0.68, method="linear")),
        "p95_m": float(np.quantile(data, 0.95, method="linear")),
        "max_m": float(np.max(data)),
    }


def paired_delta_summary(
    candidate_enu_m: np.ndarray,
    reference_enu_m: np.ndarray,
    *,
    tolerance_m: float = PAIR_EQUAL_TOLERANCE_M,
) -> list[dict[str, float | int | str]]:
    candidate = error_components(candidate_enu_m)
    reference = error_components(reference_enu_m)
    rows: list[dict[str, float | int | str]] = []
    for metric in ("e2d", "abs_up", "e3d"):
        delta = candidate[metric] - reference[metric]
        improved = delta < -tolerance_m
        worsened = delta > tolerance_m
        equal = ~(improved | worsened)
        rows.append(
            {
                "metric": metric,
                "paired_epochs": int(delta.size),
                "mean_delta_m": float(np.mean(delta)),
                "median_delta_m": float(np.median(delta)),
                "p05_delta_m": float(np.quantile(delta, 0.05, method="linear")),
                "p95_delta_m": float(np.quantile(delta, 0.95, method="linear")),
                "fraction_improved": float(np.mean(improved)),
                "fraction_worsened": float(np.mean(worsened)),
                "fraction_effectively_equal": float(np.mean(equal)),
                "equal_tolerance_m": float(tolerance_m),
            }
        )
    return rows


def historical_neutral_rows(path: Path) -> dict[str, np.ndarray]:
    """Load one seed-independent copy of the already-established neutral solve."""

    with np.load(path, allow_pickle=False) as data:
        mask = (
            (data["architecture"] == "TDL-B")
            & (data["seed"] == 0)
            & (data["ablation"] == "B-neutral")
        )
        if int(np.sum(mask)) == 0:
            raise RuntimeError("historical neutral rows are absent")
        return {
            "split_epoch_index": data["source_split_epoch_index"][mask].astype(np.int64),
            "timestamp": data["timestamp_gpst_like_s"][mask].astype(np.float64),
            "state": data["estimated_receiver_state"][mask].astype(np.float64),
            "solved": data["solved"][mask].astype(bool),
        }


def write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
