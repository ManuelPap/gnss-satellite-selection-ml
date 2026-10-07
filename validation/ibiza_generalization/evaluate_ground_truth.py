"""Independent ground-truth evaluation of completed frozen Ibiza results.

This module reads immutable JSONL inference products and the already stored
equal-weight OLS positions from the deterministic Ibiza NPZ.  It has no route
to model loading, preprocessing, normalization, training, or WLS execution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
from typing import Iterable, Mapping, Sequence

import numpy as np
import pymap3d


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MANIFEST_PATH = Path(__file__).with_name("ibiz00esp_reference.json")
PREPROCESSED_MANIFEST_PATH = Path(__file__).with_name(
    "ibiza_preprocessed_manifest.json"
)
DEFAULT_FROZEN_RESULTS_DIR = (
    REPOSITORY_ROOT.parent
    / "external_data/ibiza_2025_01_01/results/frozen_tdl"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT.parent
    / "external_data/ibiza_2025_01_01/results/ground_truth_evaluation"
)
DEFAULT_IBIZA_NPZ = (
    REPOSITORY_ROOT.parent
    / "external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz"
)
DEFAULT_OBSERVATION_RINEX = (
    REPOSITORY_ROOT.parent
    / "external_data/ibiza_2025_01_01/raw/observation/"
    "IBIZ00ESP_R_20250010000_01D_30S_MO.rnx"
)
IBIZA_NPZ_SHA256 = (
    "edb0189e9eadf3266d75984e3041a90306dd44b3ebecc0101eb188f933dc88c5"
)
ARCHITECTURES = ("TDL-B", "TDL-W", "TDL-BW")
SEEDS = tuple(range(10))
EXPECTED_RECORDS_PER_MODEL = 2856
EVALUATION_SCHEMA_VERSION = 1
QUANTILE_METHOD = "linear"

AXIS_METRICS = tuple(
    f"{axis}_{stat}_m"
    for axis in ("e", "n", "u")
    for stat in ("mean", "std", "rms")
)
RADIAL_METRICS = tuple(
    f"{dimension}_{stat}_m"
    for dimension in ("e2d", "e3d")
    for stat in ("mean", "median", "rms", "p68", "p95", "max")
)
ERROR_METRICS = AXIS_METRICS + RADIAL_METRICS
MAJOR_PAIRED_METRICS = RADIAL_METRICS


@dataclass(frozen=True)
class ValidatedResultFile:
    architecture: str
    seed: int
    path: Path
    sha256: str
    record_count: int
    solved_count: int


@dataclass(frozen=True)
class FrozenResultValidation:
    results_dir: Path
    run_manifest_path: Path
    run_manifest_sha256: str
    files: tuple[ValidatedResultFile, ...]
    accepted_epoch_indices: tuple[int, ...]
    source_split_epoch_indices: tuple[int, ...]
    timestamps_gpst_like_s: tuple[float, ...]
    input_sha256: tuple[tuple[str, str], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), REPOSITORY_ROOT)).as_posix()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"


def _canonical_line(value: Mapping[str, object]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def result_filename(architecture: str, seed: int) -> str:
    slug = {"TDL-B": "tdl_b", "TDL-W": "tdl_w", "TDL-BW": "tdl_bw"}[
        architecture
    ]
    return f"{slug}_seed_{seed}.jsonl"


def error_filename(architecture: str, seed: int) -> str:
    return result_filename(architecture, seed).replace(".jsonl", "_errors.jsonl")


def load_reference_manifest(
    path: Path = REFERENCE_MANIFEST_PATH,
) -> tuple[dict[str, object], np.ndarray]:
    text = path.read_text(encoding="utf-8")
    if "/home/" in text:
        raise RuntimeError("tracked reference manifest contains a personal path")
    manifest = json.loads(text)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported Ibiza reference manifest schema")
    if manifest.get("station_id") != "IBIZ00ESP":
        raise RuntimeError("reference manifest is not for IBIZ00ESP")
    ecef = manifest.get("trusted_reference_ecef_m")
    if not isinstance(ecef, dict):
        raise RuntimeError("reference manifest lacks trusted ECEF coordinates")
    reference = np.asarray([ecef.get(axis) for axis in ("x", "y", "z")], dtype=float)
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise RuntimeError("trusted ECEF coordinate must contain three finite values")
    expected = np.asarray([4967979.25220, 125663.49615, 3984693.03705])
    if not np.array_equal(reference, expected):
        raise RuntimeError("tracked trusted ECEF coordinate differs from the supplied value")
    return manifest, reference


def _snapshot_input_hashes(results_dir: Path) -> dict[str, str]:
    paths = [results_dir / "run_manifest.json"] + sorted(results_dir.glob("*.jsonl"))
    return {path.name: sha256_file(path) for path in paths}


def validate_frozen_result_set(
    results_dir: Path = DEFAULT_FROZEN_RESULTS_DIR,
) -> FrozenResultValidation:
    """Validate all completed inference products without modifying their bytes."""

    results_dir = results_dir.resolve()
    expected_identities = [
        (architecture, seed) for architecture in ARCHITECTURES for seed in SEEDS
    ]
    expected_names = {
        result_filename(architecture, seed)
        for architecture, seed in expected_identities
    }
    actual_names = {path.name for path in results_dir.glob("*.jsonl")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise RuntimeError(
            f"frozen result inventory is not exactly 30 files; "
            f"missing={missing}, unexpected={unexpected}"
        )

    run_manifest_path = results_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    jobs = run_manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 30:
        raise RuntimeError("run_manifest.json must describe exactly 30 jobs")
    by_identity: dict[tuple[str, int], dict[str, object]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise RuntimeError("run manifest job is not an object")
        identity = (str(job.get("architecture")), int(job.get("seed", -1)))
        if identity in by_identity:
            raise RuntimeError(f"duplicate run-manifest job: {identity}")
        by_identity[identity] = job
    if set(by_identity) != set(expected_identities):
        raise RuntimeError("run manifest architectures/seeds are not the exact 3 x 10 plan")
    if run_manifest.get("checkpoint_inventory", {}).get("selected_job_count") != 30:
        raise RuntimeError("run manifest selected-job count is not 30")

    input_hashes_before = _snapshot_input_hashes(results_dir)
    common_indices: tuple[int, ...] | None = None
    common_split_indices: tuple[int, ...] | None = None
    common_timestamps: tuple[float, ...] | None = None
    validated: list[ValidatedResultFile] = []

    for architecture, seed in expected_identities:
        path = results_dir / result_filename(architecture, seed)
        job = by_identity[(architecture, seed)]
        if job.get("result_file") != path.name:
            raise RuntimeError(f"run manifest points to the wrong file for {architecture}/{seed}")
        actual_hash = input_hashes_before[path.name]
        if job.get("result_file_sha256") != actual_hash:
            raise RuntimeError(f"SHA-256 mismatch for {path.name}")
        if int(job.get("record_count", -1)) != EXPECTED_RECORDS_PER_MODEL:
            raise RuntimeError(f"run manifest record count is wrong for {path.name}")
        if int(job.get("failed_epoch_count", -1)) != 0:
            raise RuntimeError(f"run manifest reports failed epochs for {path.name}")

        indices: list[int] = []
        split_indices: list[int] = []
        timestamps: list[float] = []
        solved_count = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.endswith("\n"):
                    raise RuntimeError(f"{path.name}:{line_number} lacks final newline")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(f"{path.name}:{line_number} is not an object")
                if row.get("architecture") != architecture or int(row.get("seed", -1)) != seed:
                    raise RuntimeError(f"row identity mismatch in {path.name}:{line_number}")
                if row.get("source_ibiza_npz_sha256") != IBIZA_NPZ_SHA256:
                    raise RuntimeError(f"wrong source NPZ hash in {path.name}:{line_number}")
                indices.append(int(row["accepted_epoch_index"]))
                split_indices.append(int(row["source_split_epoch_index"]))
                timestamps.append(float(row["timestamp_gpst_like_s"]))
                if row.get("solution_status") == "solved":
                    estimate = np.asarray(row.get("estimated_ecef_m"), dtype=float)
                    if estimate.shape != (3,) or not np.all(np.isfinite(estimate)):
                        raise RuntimeError(f"invalid solved ECEF state in {path.name}:{line_number}")
                    solved_count += 1

        if len(indices) != EXPECTED_RECORDS_PER_MODEL:
            raise RuntimeError(
                f"{path.name} has {len(indices)} records, expected {EXPECTED_RECORDS_PER_MODEL}"
            )
        if solved_count != EXPECTED_RECORDS_PER_MODEL:
            raise RuntimeError(f"{path.name} does not contain 2,856 solved epochs")
        sequence = tuple(indices)
        split_sequence = tuple(split_indices)
        timestamp_sequence = tuple(timestamps)
        if common_indices is None:
            common_indices = sequence
            common_split_indices = split_sequence
            common_timestamps = timestamp_sequence
        elif (
            sequence != common_indices
            or split_sequence != common_split_indices
            or timestamp_sequence != common_timestamps
        ):
            raise RuntimeError(f"{path.name} does not use the common epoch sequence")
        validated.append(
            ValidatedResultFile(
                architecture=architecture,
                seed=seed,
                path=path,
                sha256=actual_hash,
                record_count=len(indices),
                solved_count=solved_count,
            )
        )

    expected_indices = tuple(range(EXPECTED_RECORDS_PER_MODEL))
    if common_indices != expected_indices:
        raise RuntimeError("accepted epoch indices are not exactly 0..2855")
    input_hashes_after = _snapshot_input_hashes(results_dir)
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("an immutable inference input changed during validation")
    assert common_split_indices is not None and common_timestamps is not None
    return FrozenResultValidation(
        results_dir=results_dir,
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=input_hashes_before[run_manifest_path.name],
        files=tuple(validated),
        accepted_epoch_indices=common_indices,
        source_split_epoch_indices=common_split_indices,
        timestamps_gpst_like_s=common_timestamps,
        input_sha256=tuple(sorted(input_hashes_before.items())),
    )


def assert_frozen_inputs_unchanged(validation: FrozenResultValidation) -> None:
    expected = dict(validation.input_sha256)
    actual = _snapshot_input_hashes(validation.results_dir)
    if actual != expected:
        changed = sorted(name for name in set(expected) | set(actual) if expected.get(name) != actual.get(name))
        raise RuntimeError(f"immutable inference inputs changed: {changed}")


def reference_geodetic(reference_ecef_m: Sequence[float]) -> tuple[float, float, float]:
    reference = np.asarray(reference_ecef_m, dtype=float)
    latitude_deg, longitude_deg, height_m = pymap3d.ecef2geodetic(*reference)
    return float(latitude_deg), float(longitude_deg), float(height_m)


def ecef_differences_to_enu(
    ecef_difference_m: np.ndarray | Sequence[float],
    reference_ecef_m: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Rotate signed ECEF differences to ENU at the trusted reference point."""

    differences = np.asarray(ecef_difference_m, dtype=float)
    if differences.shape[-1:] != (3,):
        raise ValueError("ECEF differences must end in an XYZ dimension of length 3")
    latitude_deg, longitude_deg, _height_m = reference_geodetic(reference_ecef_m)
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
    sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)
    rotation = np.asarray(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=float,
    )
    return differences @ rotation.T


def evaluate_ecef_positions(
    estimated_ecef_m: np.ndarray | Sequence[Sequence[float]],
    reference_ecef_m: np.ndarray | Sequence[float],
) -> dict[str, np.ndarray]:
    estimates = np.asarray(estimated_ecef_m, dtype=float)
    reference = np.asarray(reference_ecef_m, dtype=float)
    if estimates.ndim != 2 or estimates.shape[1] != 3:
        raise ValueError("estimated ECEF positions must have shape (epochs, 3)")
    if reference.shape != (3,):
        raise ValueError("reference ECEF position must have shape (3,)")
    if not np.all(np.isfinite(estimates)) or not np.all(np.isfinite(reference)):
        raise ValueError("position evaluation requires finite coordinates")
    differences = estimates - reference
    enu = ecef_differences_to_enu(differences, reference)
    e2d = np.hypot(enu[:, 0], enu[:, 1])
    e3d = np.linalg.norm(enu, axis=1)
    return {
        "ecef_difference_m": differences,
        "enu_error_m": enu,
        "e2d_m": e2d,
        "e3d_m": e3d,
    }


def linear_quantile(values: Sequence[float] | np.ndarray, probability: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("quantiles require a non-empty finite one-dimensional array")
    return float(np.quantile(array, probability, method=QUANTILE_METHOD))


def compute_error_metrics(enu_error_m: np.ndarray, total_epochs: int) -> dict[str, object]:
    enu = np.asarray(enu_error_m, dtype=float)
    if enu.ndim != 2 or enu.shape[1] != 3 or len(enu) == 0:
        raise ValueError("metrics require at least one solved ENU row")
    if not np.all(np.isfinite(enu)):
        raise ValueError("metrics require finite ENU errors")
    if total_epochs < len(enu):
        raise ValueError("solved epoch count cannot exceed total epoch count")
    result: dict[str, object] = {
        "solved_epochs": int(len(enu)),
        "total_epochs": int(total_epochs),
        "availability": float(len(enu) / total_epochs),
    }
    for index, axis in enumerate(("e", "n", "u")):
        values = enu[:, index]
        result[f"{axis}_mean_m"] = float(np.mean(values))
        result[f"{axis}_std_m"] = float(np.std(values, ddof=0))
        result[f"{axis}_rms_m"] = float(np.sqrt(np.mean(np.square(values))))
    for name, values in (
        ("e2d", np.hypot(enu[:, 0], enu[:, 1])),
        ("e3d", np.linalg.norm(enu, axis=1)),
    ):
        result[f"{name}_mean_m"] = float(np.mean(values))
        result[f"{name}_median_m"] = float(np.median(values))
        result[f"{name}_rms_m"] = float(np.sqrt(np.mean(np.square(values))))
        result[f"{name}_p68_m"] = linear_quantile(values, 0.68)
        result[f"{name}_p95_m"] = linear_quantile(values, 0.95)
        result[f"{name}_max_m"] = float(np.max(values))
    return result


def aggregate_seed_metrics(
    per_seed: Sequence[Mapping[str, object]],
    expected_architectures: Sequence[str] = ARCHITECTURES,
) -> list[dict[str, object]]:
    """Aggregate ten seed-level values per architecture; epochs are never pooled."""

    rows: list[dict[str, object]] = []
    metric_names = ("solved_epochs", "availability") + ERROR_METRICS
    for architecture in expected_architectures:
        selected = sorted(
            (row for row in per_seed if row.get("architecture") == architecture),
            key=lambda row: int(row["seed"]),
        )
        if [int(row["seed"]) for row in selected] != list(SEEDS):
            raise RuntimeError(f"{architecture} seed-level aggregation requires seeds 0..9")
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            rows.append(
                {
                    "architecture": architecture,
                    "metric": metric,
                    "seed_count": 10,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=0)),
                    "iqr": linear_quantile(values, 0.75) - linear_quantile(values, 0.25),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return rows


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def evaluate_model_result(
    source: ValidatedResultFile,
    output_path: Path,
    reference_ecef_m: np.ndarray,
) -> dict[str, object]:
    rows = _load_jsonl(source.path)
    solved_positions: list[list[float]] = []
    solved_row_positions: list[int] = []
    for position, row in enumerate(rows):
        if row.get("solution_status") == "solved":
            solved_positions.append([float(value) for value in row["estimated_ecef_m"]])
            solved_row_positions.append(position)
    evaluated = evaluate_ecef_positions(np.asarray(solved_positions), reference_ecef_m)
    solved_lookup = {position: index for index, position in enumerate(solved_row_positions)}
    enu_rows: list[np.ndarray] = []
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for position, row in enumerate(rows):
                output: dict[str, object] = {
                    "schema_version": EVALUATION_SCHEMA_VERSION,
                    "architecture": source.architecture,
                    "seed": source.seed,
                    "source_result_file": source.path.name,
                    "source_result_sha256": source.sha256,
                    "accepted_epoch_index": int(row["accepted_epoch_index"]),
                    "source_split_epoch_index": int(row["source_split_epoch_index"]),
                    "timestamp_gpst_like_s": float(row["timestamp_gpst_like_s"]),
                    "solution_status": str(row["solution_status"]),
                    "estimated_ecef_m": row["estimated_ecef_m"],
                }
                evaluated_index = solved_lookup.get(position)
                if evaluated_index is None:
                    output.update(
                        {
                            "ecef_difference_m": None,
                            "enu_error_m": None,
                            "e2d_m": None,
                            "e3d_m": None,
                        }
                    )
                else:
                    difference = evaluated["ecef_difference_m"][evaluated_index]
                    enu = evaluated["enu_error_m"][evaluated_index]
                    enu_rows.append(enu)
                    output.update(
                        {
                            "ecef_difference_m": [float(value) for value in difference],
                            "enu_error_m": {
                                "east": float(enu[0]),
                                "north": float(enu[1]),
                                "up": float(enu[2]),
                            },
                            "e2d_m": float(evaluated["e2d_m"][evaluated_index]),
                            "e3d_m": float(evaluated["e3d_m"][evaluated_index]),
                        }
                    )
                stream.write(_canonical_line(output))
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    metrics = compute_error_metrics(np.vstack(enu_rows), len(rows))
    return {"architecture": source.architecture, "seed": source.seed, **metrics}


def load_stored_ols_baseline(
    dataset_path: Path,
    validation: FrozenResultValidation,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if sha256_file(dataset_path) != IBIZA_NPZ_SHA256:
        raise RuntimeError("Ibiza NPZ SHA-256 does not match the frozen dataset")
    with np.load(dataset_path, allow_pickle=False) as dataset:
        required = (
            "epoch_ols_initial_state",
            "epoch_split_index",
            "epoch_time_gpst_like_s",
        )
        missing = [name for name in required if name not in dataset.files]
        if missing:
            raise RuntimeError(
                "validated non-ML OLS baseline is not stored in the Ibiza NPZ: "
                + ", ".join(missing)
            )
        state = np.asarray(dataset["epoch_ols_initial_state"], dtype=float)
        split = np.asarray(dataset["epoch_split_index"], dtype=np.int64)
        timestamp = np.asarray(dataset["epoch_time_gpst_like_s"], dtype=float)
    if state.shape != (EXPECTED_RECORDS_PER_MODEL, 7) or not np.all(np.isfinite(state)):
        raise RuntimeError("stored OLS baseline has an invalid state array")
    if tuple(split.tolist()) != validation.source_split_epoch_indices:
        raise RuntimeError("stored OLS baseline split epochs differ from model results")
    if tuple(timestamp.tolist()) != validation.timestamps_gpst_like_s:
        raise RuntimeError("stored OLS baseline timestamps differ from model results")
    metadata = {"split": split, "timestamp": timestamp}
    return state[:, :3].copy(), metadata


def evaluate_baseline(
    positions_ecef_m: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    output_path: Path,
    reference_ecef_m: np.ndarray,
) -> dict[str, object]:
    evaluated = evaluate_ecef_positions(positions_ecef_m, reference_ecef_m)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for index in range(len(positions_ecef_m)):
                enu = evaluated["enu_error_m"][index]
                row = {
                    "schema_version": EVALUATION_SCHEMA_VERSION,
                    "baseline": "stored_equal_weight_ols",
                    "source_array": "epoch_ols_initial_state[:, :3]",
                    "source_ibiza_npz_sha256": IBIZA_NPZ_SHA256,
                    "accepted_epoch_index": index,
                    "source_split_epoch_index": int(metadata["split"][index]),
                    "timestamp_gpst_like_s": float(metadata["timestamp"][index]),
                    "solution_status": "solved",
                    "estimated_ecef_m": [float(value) for value in positions_ecef_m[index]],
                    "ecef_difference_m": [
                        float(value) for value in evaluated["ecef_difference_m"][index]
                    ],
                    "enu_error_m": {
                        "east": float(enu[0]),
                        "north": float(enu[1]),
                        "up": float(enu[2]),
                    },
                    "e2d_m": float(evaluated["e2d_m"][index]),
                    "e3d_m": float(evaluated["e3d_m"][index]),
                }
                stream.write(_canonical_line(row))
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "baseline": "stored_equal_weight_ols",
        "source_array": "epoch_ols_initial_state[:, :3]",
        **compute_error_metrics(evaluated["enu_error_m"], len(positions_ecef_m)),
    }


def paired_metric_differences(
    per_seed: Sequence[Mapping[str, object]],
    baseline: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed_row in per_seed:
        row: dict[str, object] = {
            "architecture": seed_row["architecture"],
            "seed": int(seed_row["seed"]),
            "common_epoch_count": int(seed_row["solved_epochs"]),
            "difference_definition": "ML seed metric minus common stored OLS baseline metric",
        }
        for metric in MAJOR_PAIRED_METRICS:
            row[f"delta_{metric}"] = float(seed_row[metric]) - float(baseline[metric])
        rows.append(row)
    return rows


def parse_rinex_approx_position(path: Path) -> np.ndarray | None:
    with path.open(encoding="ascii", errors="strict") as stream:
        for line in stream:
            if "APPROX POSITION XYZ" in line[60:]:
                values = line[:60].split()
                if len(values) < 3:
                    raise RuntimeError("malformed RINEX APPROX POSITION XYZ header")
                return np.asarray([float(value) for value in values[:3]], dtype=float)
            if "END OF HEADER" in line[60:]:
                break
    return None


def reference_provenance_check(
    observation_path: Path,
    reference_ecef_m: np.ndarray,
) -> dict[str, object]:
    preprocessing_manifest = json.loads(
        PREPROCESSED_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    expected = preprocessing_manifest["inputs"]["observation"]
    if observation_path.name != expected["filename"]:
        raise RuntimeError("RINEX observation filename differs from preprocessing provenance")
    observation_hash = sha256_file(observation_path)
    if observation_hash != expected["sha256"]:
        raise RuntimeError("RINEX observation hash differs from preprocessing provenance")
    approximate = parse_rinex_approx_position(observation_path)
    if approximate is None:
        return {
            "available": False,
            "observation_file": observation_path.name,
            "observation_sha256": observation_hash,
        }
    evaluated = evaluate_ecef_positions(approximate.reshape(1, 3), reference_ecef_m)
    difference = evaluated["ecef_difference_m"][0]
    enu = evaluated["enu_error_m"][0]
    return {
        "available": True,
        "purpose": "provenance check only",
        "observation_file": observation_path.name,
        "observation_sha256": observation_hash,
        "rinex_approx_position_ecef_m": [float(value) for value in approximate],
        "ecef_difference_m": {
            "dx": float(difference[0]),
            "dy": float(difference[1]),
            "dz": float(difference[2]),
        },
        "enu_offset_m": {
            "east": float(enu[0]),
            "north": float(enu[1]),
            "up": float(enu[2]),
        },
        "coordinate_difference_3d_m": float(evaluated["e3d_m"][0]),
    }


def _csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "pymap3d": importlib.metadata.version("pymap3d"),
        "operating_system": platform.system(),
        "machine": platform.machine(),
    }


def run_evaluation(
    *,
    results_dir: Path = DEFAULT_FROZEN_RESULTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dataset_path: Path = DEFAULT_IBIZA_NPZ,
    observation_path: Path = DEFAULT_OBSERVATION_RINEX,
    reference_manifest_path: Path = REFERENCE_MANIFEST_PATH,
    overwrite: bool = False,
) -> dict[str, object]:
    """Validate immutable inputs, evaluate all models and stored OLS, and export."""

    validation = validate_frozen_result_set(results_dir)
    reference_manifest, reference_ecef_m = load_reference_manifest(reference_manifest_path)
    positions, baseline_metadata = load_stored_ols_baseline(dataset_path, validation)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_outputs = [
        output_dir / error_filename(source.architecture, source.seed)
        for source in validation.files
    ]
    summary_outputs = [
        output_dir / "baseline_epoch_errors.jsonl",
        output_dir / "per_seed_metrics.json",
        output_dir / "per_seed_metrics.csv",
        output_dir / "across_seed_summary.json",
        output_dir / "across_seed_summary.csv",
        output_dir / "baseline_metrics.json",
        output_dir / "paired_vs_baseline.json",
        output_dir / "paired_vs_baseline.csv",
        output_dir / "reference_validation.json",
        output_dir / "evaluation_manifest.json",
    ]
    existing = [path for path in model_outputs + summary_outputs if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing[:5])
        raise FileExistsError(
            f"evaluation targets already exist ({names}); pass --overwrite to replace this set"
        )

    per_seed: list[dict[str, object]] = []
    for source, output_path in zip(validation.files, model_outputs, strict=True):
        per_seed.append(evaluate_model_result(source, output_path, reference_ecef_m))
    baseline = evaluate_baseline(
        positions,
        baseline_metadata,
        output_dir / "baseline_epoch_errors.jsonl",
        reference_ecef_m,
    )
    across_seed = aggregate_seed_metrics(per_seed)
    paired = paired_metric_differences(per_seed, baseline)
    provenance = reference_provenance_check(observation_path, reference_ecef_m)

    _atomic_write_text(output_dir / "per_seed_metrics.json", _canonical_json(per_seed))
    _atomic_write_text(output_dir / "per_seed_metrics.csv", _csv_text(per_seed))
    _atomic_write_text(
        output_dir / "across_seed_summary.json", _canonical_json(across_seed)
    )
    _atomic_write_text(
        output_dir / "across_seed_summary.csv", _csv_text(across_seed)
    )
    _atomic_write_text(output_dir / "baseline_metrics.json", _canonical_json(baseline))
    _atomic_write_text(output_dir / "paired_vs_baseline.json", _canonical_json(paired))
    _atomic_write_text(output_dir / "paired_vs_baseline.csv", _csv_text(paired))
    _atomic_write_text(
        output_dir / "reference_validation.json", _canonical_json(provenance)
    )

    assert_frozen_inputs_unchanged(validation)
    generated = sorted(path for path in model_outputs + summary_outputs[:-1])
    output_hashes = [
        {
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in generated
    ]
    latitude_deg, longitude_deg, height_m = reference_geodetic(reference_ecef_m)
    evaluation_manifest: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "completed",
        "reference": {
            "station_id": reference_manifest["station_id"],
            "manifest": _display_path(reference_manifest_path),
            "manifest_sha256": sha256_file(reference_manifest_path),
            "trusted_ecef_m": [float(value) for value in reference_ecef_m],
            "coordinate_reference_frame": reference_manifest[
                "coordinate_reference_frame"
            ],
            "source": reference_manifest["source"],
            "coordinate_epoch": reference_manifest["coordinate_epoch"],
            "derived_wgs84_geodetic": {
                "latitude_deg": latitude_deg,
                "longitude_deg": longitude_deg,
                "ellipsoidal_height_m": height_m,
            },
        },
        "frozen_result_validation": {
            "source_directory": _display_path(validation.results_dir),
            "run_manifest_sha256": validation.run_manifest_sha256,
            "result_file_count": len(validation.files),
            "architectures": list(ARCHITECTURES),
            "seeds_per_architecture": list(SEEDS),
            "records_per_file": EXPECTED_RECORDS_PER_MODEL,
            "total_records": sum(source.record_count for source in validation.files),
            "solved_records": sum(source.solved_count for source in validation.files),
            "common_epochs_verified": True,
            "result_hashes_match_run_manifest": True,
            "inputs_byte_identical_after_evaluation": True,
        },
        "source_dataset": {
            "path": _display_path(dataset_path),
            "sha256": IBIZA_NPZ_SHA256,
            "baseline_available": True,
            "baseline": "stored equal-weight OLS epoch_ols_initial_state[:, :3]",
        },
        "metric_policy": {
            "ecef_difference": "estimated ECEF minus trusted reference ECEF",
            "enu_rotation": "standard WGS84 reference-latitude/longitude ECEF-to-ENU rotation",
            "axis_signs": "signed east, north, and up",
            "standard_deviation": "population standard deviation (ddof=0)",
            "rms": "sqrt(mean(square(values)))",
            "quantiles": f"NumPy quantile probabilities 0.68 and 0.95 with method={QUANTILE_METHOD}",
            "across_seed_unit": "ten seed-level metric values; epoch errors are not pooled across seeds",
            "paired_difference": "ML seed metric minus common stored OLS baseline metric",
        },
        "per_epoch_error_schema": {
            "identity": [
                "architecture",
                "seed",
                "accepted_epoch_index",
                "source_split_epoch_index",
                "timestamp_gpst_like_s",
            ],
            "source_output": ["solution_status", "estimated_ecef_m"],
            "errors": [
                "ecef_difference_m=[dX,dY,dZ]",
                "enu_error_m={east,north,up}",
                "e2d_m",
                "e3d_m",
            ],
            "unsolved_policy": "retain row identity and status with null error fields",
        },
        "runtime_versions": _runtime_versions(),
        "scientific_controls": {
            "inference_rerun": False,
            "preprocessing_rerun": False,
            "training_or_fine_tuning": False,
            "normalization_recomputed": False,
            "wls_called_or_modified": False,
            "ground_truth_scope": "evaluation module and generated evaluation outputs only",
        },
        "generated_files": output_hashes,
    }
    _atomic_write_text(
        output_dir / "evaluation_manifest.json", _canonical_json(evaluation_manifest)
    )
    assert_frozen_inputs_unchanged(validation)
    return evaluation_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate completed frozen Ibiza inference results against IBIZ00ESP"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_FROZEN_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_IBIZA_NPZ)
    parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION_RINEX)
    parser.add_argument(
        "--reference-manifest", type=Path, default=REFERENCE_MANIFEST_PATH
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the deterministic evaluation output set, never inference inputs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    manifest = run_evaluation(
        results_dir=arguments.results_dir,
        output_dir=arguments.output_dir,
        dataset_path=arguments.dataset,
        observation_path=arguments.observation,
        reference_manifest_path=arguments.reference_manifest,
        overwrite=arguments.overwrite,
    )
    validation = manifest["frozen_result_validation"]
    print(
        f"evaluated {validation['result_file_count']} immutable result files / "
        f"{validation['total_records']} records into {arguments.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
