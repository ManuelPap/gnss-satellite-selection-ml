"""Ground-truth-isolated official-RTKLIB SPP experiment primitives.

The solve phase in this module has no reference/ground-truth argument and does
not read the reference file.  It solves the complete observation interval,
serializes all ECEF solutions, and performs timestamp-only support matching.
Only :func:`evaluate_serialized_positions` opens the ground-truth document.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from validation.tasgnss_ibiza_solver_audit.core import (
    enu_errors,
    trusted_reference_ecef,
)
from validation.tdl_9feature_tasgnss_analysis.core import percentile_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = PROJECT_ROOT.parent
PACKAGE_ROOT = Path(__file__).resolve().parent

RTKLIB_ROOT = PHD_ROOT / "external_references/RTKLIB"
OBSERVATION = (
    PHD_ROOT
    / "external_data/ibiza_2025_01_01/raw/observation/"
    "IBIZ00ESP_R_20250010000_01D_30S_MO.rnx"
)
NAVIGATION = (
    PHD_ROOT
    / "external_data/ibiza_2025_01_01/raw/navigation/"
    "BRDM00DLR_S_20250010000_01D_MN.rnx"
)
REFERENCE = PROJECT_ROOT / "validation/ibiza_generalization/ibiz00esp_reference.json"
FROZEN_EVALUATION = (
    PHD_ROOT / "external_data/current_tdl_reproduction/ibiza_evaluation"
)
DEFAULT_OUTPUT = PHD_ROOT / "external_data/rtklib_ibiza_controlled_baseline"
CONFIG = PACKAGE_ROOT / "rtklib_config.conf"
CONFIGURATION_AUDIT = PACKAGE_ROOT / "configuration_audit.json"

OBSERVATION_SHA256 = "18dfcaa247fb69b8260eda1257279411ca2317a244d3c3afd37eb74bfaa8000b"
NAVIGATION_SHA256 = "423c488a7f0a4825c2e88d848775726b86f65b37ad6a6a63bc98a26fc8c0e1bb"
REFERENCE_SHA256 = "801309b88988ee7407bf2568016b580ba592b7a15bac4789f7f9a720116b94cd"
RTKLIB_HEAD = "180043ee24b6d2b168f98b64be15f69d50046b1a"
EXPECTED_CANDIDATE_EPOCHS = 2880
EXPECTED_PRIMARY_EPOCHS = 2849
SEEDS = tuple(range(10))
PERTURBATION_ECEF_M = np.asarray([37.0, -19.0, 11.0], dtype=np.float64)


@dataclass(frozen=True)
class FrozenSupport:
    candidate_epoch_ids: np.ndarray
    candidate_timestamps_gpst_s: np.ndarray
    required_epoch_ids: np.ndarray
    required_timestamps_gpst_s: np.ndarray
    support_sha256: str
    seed_artifact_sha256: dict[int, str]


@dataclass(frozen=True)
class RtklibSolutions:
    timestamps_gpst_s: np.ndarray
    ecef_m: np.ndarray
    quality: np.ndarray
    satellite_count: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def sha256_named_arrays(items: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode("utf-8"))
        digest.update(sha256_array(value).encode("ascii"))
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_provenance(repository: Path) -> dict[str, Any]:
    remotes: dict[str, dict[str, str]] = {}
    lines = _git_output(repository, "remote", "-v").splitlines()
    for line in lines:
        fields = line.split()
        if len(fields) >= 3:
            name, url = fields[:2]
            direction = fields[2].strip("()")
            remotes.setdefault(name, {})[direction] = url
    status = _git_output(repository, "status", "--short").splitlines()
    return {
        "path": str(repository.resolve()),
        "head": _git_output(repository, "rev-parse", "HEAD"),
        "describe": _git_output(repository, "describe", "--tags", "--always"),
        "branch": _git_output(repository, "branch", "--show-current"),
        "status_short": status,
        "clean": not status,
        "remotes": remotes,
    }


def verify_rtklib_identity() -> dict[str, Any]:
    provenance = git_provenance(RTKLIB_ROOT)
    if provenance["head"] != RTKLIB_HEAD:
        raise RuntimeError(
            f"official RTKLIB HEAD is {provenance['head']}, expected {RTKLIB_HEAD}"
        )
    return provenance


def assert_checkout_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    for key in ("head", "describe", "branch", "status_short", "remotes"):
        if before[key] != after[key]:
            raise RuntimeError(f"RTKLIB checkout changed during experiment: {key}")


def verify_scientific_inputs(*, include_reference: bool) -> dict[str, str]:
    expected = {
        "observation": (OBSERVATION, OBSERVATION_SHA256),
        "navigation": (NAVIGATION, NAVIGATION_SHA256),
    }
    if include_reference:
        expected["ground_truth"] = (REFERENCE, REFERENCE_SHA256)
    result: dict[str, str] = {}
    for name, (path, expected_hash) in expected.items():
        actual = sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(f"{name} SHA-256 is {actual}, expected {expected_hash}")
        result[name] = actual
    return result


def _seed_artifact_paths() -> dict[int, Path]:
    return {
        seed: FROZEN_EVALUATION / f"positions_pre_ground_truth/seed_{seed}.npz"
        for seed in SEEDS
    }


def load_frozen_primary_support() -> FrozenSupport:
    """Recover and cross-check the exact primary support from all ten NPZs."""

    manifest_path = FROZEN_EVALUATION / "ibiza_evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_manifest = manifest.get("artifacts", {}).get("positions_pre_ground_truth", {})
    common_candidate_ids: np.ndarray | None = None
    common_timestamps: np.ndarray | None = None
    common_paired: np.ndarray | None = None
    artifact_hashes: dict[int, str] = {}

    for seed, path in _seed_artifact_paths().items():
        actual_hash = sha256_file(path)
        declared = artifact_manifest.get(str(seed), {}).get("sha256")
        if actual_hash != declared:
            raise RuntimeError(f"frozen seed {seed} position artifact SHA-256 mismatch")
        artifact_hashes[seed] = actual_hash
        with np.load(path, allow_pickle=False) as data:
            candidate_ids = data["candidate_epoch_id"].astype(np.int64)
            timestamps = data["epoch_time_gpst_like_s"].astype(np.float64)
            paired = data["paired"].astype(bool)
        if candidate_ids.shape != (EXPECTED_CANDIDATE_EPOCHS,):
            raise RuntimeError(f"seed {seed} does not contain 2,880 candidates")
        if timestamps.shape != candidate_ids.shape or paired.shape != candidate_ids.shape:
            raise RuntimeError(f"seed {seed} frozen support arrays have inconsistent shapes")
        if int(np.sum(paired)) != EXPECTED_PRIMARY_EPOCHS:
            raise RuntimeError(f"seed {seed} primary support is not exactly 2,849 epochs")
        if common_candidate_ids is None:
            common_candidate_ids = candidate_ids.copy()
            common_timestamps = timestamps.copy()
            common_paired = paired.copy()
        elif not (
            np.array_equal(candidate_ids, common_candidate_ids)
            and np.array_equal(timestamps, common_timestamps)
            and np.array_equal(paired, common_paired)
        ):
            raise RuntimeError(
                f"seed {seed} primary epoch IDs/timestamps differ from seed 0"
            )

    assert common_candidate_ids is not None
    assert common_timestamps is not None
    assert common_paired is not None
    required_ids = common_candidate_ids[common_paired]
    required_timestamps = common_timestamps[common_paired]
    if np.unique(required_ids).size != EXPECTED_PRIMARY_EPOCHS:
        raise RuntimeError("frozen primary epoch IDs are not unique")
    if np.unique(required_timestamps).size != EXPECTED_PRIMARY_EPOCHS:
        raise RuntimeError("frozen primary timestamps are not unique")
    support_hash = sha256_named_arrays(
        (
            ("candidate_epoch_id", required_ids),
            ("epoch_time_gpst_like_s", required_timestamps),
        )
    )
    return FrozenSupport(
        candidate_epoch_ids=common_candidate_ids,
        candidate_timestamps_gpst_s=common_timestamps,
        required_epoch_ids=required_ids,
        required_timestamps_gpst_s=required_timestamps,
        support_sha256=support_hash,
        seed_artifact_sha256=artifact_hashes,
    )


def _gpst_calendar_seconds(date_text: str, time_text: str) -> float:
    # Both RTKLIB output and the frozen artifact use a GPST-like Unix calendar
    # representation.  No UTC leap-second conversion is applied here.
    parsed = datetime.strptime(
        f"{date_text} {time_text}", "%Y/%m/%d %H:%M:%S.%f"
    ).replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def parse_rtklib_pos(path: Path) -> RtklibSolutions:
    timestamps: list[float] = []
    ecef: list[list[float]] = []
    quality: list[int] = []
    satellite_count: list[int] = []
    with path.open(encoding="ascii", errors="strict") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith("%"):
                continue
            fields = line.split()
            if len(fields) < 7:
                raise RuntimeError(f"{path.name}:{line_number} is not a complete XYZ solution")
            try:
                timestamp = _gpst_calendar_seconds(fields[0], fields[1])
                position = [float(fields[2]), float(fields[3]), float(fields[4])]
                solution_quality = int(fields[5])
                satellites = int(fields[6])
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}:{line_number} has invalid RTKLIB fields"
                ) from error
            timestamps.append(timestamp)
            ecef.append(position)
            quality.append(solution_quality)
            satellite_count.append(satellites)
    result = RtklibSolutions(
        timestamps_gpst_s=np.asarray(timestamps, dtype=np.float64),
        ecef_m=np.asarray(ecef, dtype=np.float64).reshape((-1, 3)),
        quality=np.asarray(quality, dtype=np.int16),
        satellite_count=np.asarray(satellite_count, dtype=np.int16),
    )
    if result.timestamps_gpst_s.size == 0:
        raise RuntimeError("RTKLIB output contains no valid XYZ solutions")
    if not np.all(np.isfinite(result.ecef_m)):
        raise RuntimeError("RTKLIB output contains non-finite ECEF coordinates")
    if np.unique(result.timestamps_gpst_s).size != result.timestamps_gpst_s.size:
        duplicates = _duplicates(result.timestamps_gpst_s)
        raise RuntimeError(f"duplicate RTKLIB timestamps: {duplicates[:20]}")
    return result


def _duplicates(values: np.ndarray) -> list[float]:
    unique, counts = np.unique(values, return_counts=True)
    return unique[counts > 1].astype(float).tolist()


def match_required_support(
    solutions: RtklibSolutions, support: FrozenSupport
) -> dict[str, Any]:
    index_by_time = {
        float(timestamp): index
        for index, timestamp in enumerate(solutions.timestamps_gpst_s.tolist())
    }
    required = support.required_timestamps_gpst_s
    missing = [float(value) for value in required if float(value) not in index_by_time]
    required_set = set(required.astype(float).tolist())
    extra = [
        float(value)
        for value in solutions.timestamps_gpst_s
        if float(value) not in required_set
    ]
    matched_indices = np.asarray(
        [index_by_time[float(value)] for value in required if float(value) in index_by_time],
        dtype=np.int64,
    )
    return {
        "required_primary_epoch_count": int(required.size),
        "rtklib_solution_count": int(solutions.timestamps_gpst_s.size),
        "rtklib_solution_count_on_required_support": int(matched_indices.size),
        "missing_required_epoch_count": len(missing),
        "missing_required_timestamps_gpst_s": missing,
        "extra_rtklib_epoch_count": len(extra),
        "extra_rtklib_timestamps_gpst_s": extra,
        "exact_timestamp_match": not missing and matched_indices.size == required.size,
        "matched_solution_indices": matched_indices,
    }


def require_complete_primary_support(match: Mapping[str, Any]) -> None:
    if not bool(match["exact_timestamp_match"]):
        raise RuntimeError(
            "exact 2,849-epoch primary support is incomplete; "
            f"missing={match['missing_required_timestamps_gpst_s']}"
        )


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))
    temporary.replace(path)


def _string_array(value: str) -> np.ndarray:
    return np.asarray([value], dtype=f"U{max(1, len(value))}")


def serialize_pre_ground_truth(
    path: Path,
    solutions: RtklibSolutions,
    support: FrozenSupport,
    match: Mapping[str, Any],
    *,
    config_sha256: str,
    observation_sha256: str,
    navigation_sha256: str,
    raw_output_sha256: str,
) -> None:
    """Serialize every solver-dependent value without opening ground truth."""

    indices = np.asarray(match["matched_solution_indices"], dtype=np.int64)
    arrays = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "timestamp_gpst_like_s": solutions.timestamps_gpst_s,
        "position_ecef_m": solutions.ecef_m,
        "solution_quality": solutions.quality,
        "satellite_count": solutions.satellite_count,
        "required_epoch_id": support.required_epoch_ids,
        "required_timestamp_gpst_like_s": support.required_timestamps_gpst_s,
        "matched_solution_index": indices,
        "support_complete": np.asarray(
            [int(bool(match["exact_timestamp_match"]))], dtype=np.uint8
        ),
        "support_sha256": _string_array(support.support_sha256),
        "config_sha256": _string_array(config_sha256),
        "observation_sha256": _string_array(observation_sha256),
        "navigation_sha256": _string_array(navigation_sha256),
        "raw_output_sha256": _string_array(raw_output_sha256),
        "ground_truth_available": np.asarray([0], dtype=np.uint8),
    }
    write_deterministic_npz(path, arrays)


def load_pre_ground_truth(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    required = {
        "timestamp_gpst_like_s",
        "position_ecef_m",
        "solution_quality",
        "satellite_count",
        "required_epoch_id",
        "required_timestamp_gpst_like_s",
        "matched_solution_index",
        "support_complete",
        "support_sha256",
        "config_sha256",
        "observation_sha256",
        "navigation_sha256",
        "raw_output_sha256",
        "ground_truth_available",
    }
    if not required.issubset(arrays):
        raise RuntimeError(f"serialized RTKLIB positions lack {sorted(required - arrays.keys())}")
    if int(arrays["ground_truth_available"][0]) != 0:
        raise RuntimeError("pre-ground-truth artifact incorrectly declares ground truth")
    return arrays


def raw_position_fingerprints(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    names = (
        "timestamp_gpst_like_s",
        "position_ecef_m",
        "solution_quality",
        "satellite_count",
        "required_epoch_id",
        "required_timestamp_gpst_like_s",
        "matched_solution_index",
        "support_complete",
        "support_sha256",
        "config_sha256",
        "observation_sha256",
        "navigation_sha256",
        "raw_output_sha256",
    )
    return {name: sha256_array(np.asarray(arrays[name])) for name in names}


def solution_metrics(
    positions_ecef_m: np.ndarray, reference_ecef_m: np.ndarray
) -> dict[str, Any]:
    """Use the same ENU and NumPy-linear percentile code as the Ibiza evaluator."""

    enu = enu_errors(positions_ecef_m, reference_ecef_m)
    error_2d = np.linalg.norm(enu[:, :2], axis=1)
    error_3d = np.linalg.norm(enu, axis=1)
    return {
        "2d": percentile_metrics(error_2d),
        "3d": percentile_metrics(error_3d),
        "east_rms": float(np.sqrt(np.mean(enu[:, 0] ** 2))),
        "north_rms": float(np.sqrt(np.mean(enu[:, 1] ** 2))),
        "up_rms": float(np.sqrt(np.mean(enu[:, 2] ** 2))),
    }


def _evaluate_with_reference(
    arrays: Mapping[str, np.ndarray], reference_ecef_m: np.ndarray
) -> dict[str, Any]:
    if not bool(arrays["support_complete"][0]):
        return {
            "status": "incomplete_exact_support",
            "metrics": None,
            "inclusion_policy": (
                "No primary metrics computed because exact required timestamp support "
                "was incomplete; support was not silently shrunk."
            ),
        }
    indices = arrays["matched_solution_index"].astype(np.int64)
    required_timestamps = arrays["required_timestamp_gpst_like_s"].astype(np.float64)
    matched_timestamps = arrays["timestamp_gpst_like_s"][indices].astype(np.float64)
    if not np.array_equal(matched_timestamps, required_timestamps):
        raise RuntimeError("serialized primary support is not an exact timestamp match")
    metrics = solution_metrics(
        arrays["position_ecef_m"][indices].astype(np.float64), reference_ecef_m
    )
    return {
        "status": "complete_exact_support",
        "metrics": metrics,
        "inclusion_policy": (
            "exact frozen timestamp membership and RTKLIB solution availability only; "
            "no ground-truth error threshold"
        ),
    }


def _neutral_and_learned_columns() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    per_seed_path = FROZEN_EVALUATION / "per_seed_metrics.json"
    across_path = FROZEN_EVALUATION / "across_seed_summary.json"
    per_seed = json.loads(per_seed_path.read_text(encoding="utf-8"))
    seed_rows = per_seed["supports"]["primary_current_stack"]
    if len(seed_rows) != 10:
        raise RuntimeError("frozen per-seed metrics do not contain all ten seeds")
    neutral = seed_rows[0]["neutral_paired"]
    if any(row["neutral_paired"] != neutral for row in seed_rows[1:]):
        raise RuntimeError("neutral TASGNSS summaries differ despite identical support")

    across = json.loads(across_path.read_text(encoding="utf-8"))
    primary = across["supports"]["primary_current_stack"]
    if primary["seed_count"] != 10:
        raise RuntimeError("frozen across-seed summary is not based on ten seeds")
    source_metrics = primary["metrics"]
    learned = {
        dimension: {
            statistic: float(source_metrics[f"learned_{dimension}_{statistic}"]["mean"])
            for statistic in ("mean", "median", "rms", "p68", "p95", "max")
        }
        for dimension in ("2d", "3d")
    }
    for axis in ("east", "north", "up"):
        learned[f"{axis}_rms"] = float(
            source_metrics[f"learned_{axis}_rms"]["mean"]
        )
    hashes = {
        "per_seed_metrics_sha256": sha256_file(per_seed_path),
        "across_seed_summary_sha256": sha256_file(across_path),
    }
    return neutral, learned, hashes


def controlled_comparison(rtklib_metrics: Mapping[str, Any]) -> dict[str, Any]:
    neutral, learned, hashes = _neutral_and_learned_columns()
    return {
        "schema_version": 1,
        "support": "exact frozen 2,849 primary current-stack timestamps",
        "columns": {
            "RTKLIB official SPP": {
                "aggregation": "single deterministic solution",
                "metrics": rtklib_metrics,
            },
            "Neutral TASGNSS": {
                "aggregation": "single deterministic solution",
                "metrics": neutral,
            },
            "HybridShareSysNet + TASGNSS": {
                "aggregation": "mean of ten seed-level metric summaries; epochs are not pooled",
                "metrics": learned,
            },
        },
        "frozen_neural_sources": hashes,
    }


def evaluate_serialized_positions(
    positions_path: Path, reference_path: Path = REFERENCE
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Open ground truth only after the immutable position artifact is loaded."""

    arrays = load_pre_ground_truth(positions_path)
    before = raw_position_fingerprints(arrays)

    # This is the sole reference-file open in the experiment implementation.
    reference_document = json.loads(reference_path.read_text(encoding="utf-8"))
    reference = trusted_reference_ecef(reference_document)
    primary = _evaluate_with_reference(arrays, reference)
    perturbed = _evaluate_with_reference(arrays, reference + PERTURBATION_ECEF_M)
    after = raw_position_fingerprints(arrays)
    raw_unchanged = {name: before[name] == after[name] for name in before}
    metrics_changed = (
        primary["metrics"] is not None
        and primary["metrics"] != perturbed["metrics"]
    )
    if not all(raw_unchanged.values()):
        raise RuntimeError("ground-truth perturbation changed raw RTKLIB products")
    if primary["metrics"] is not None and not metrics_changed:
        raise RuntimeError("ground-truth perturbation did not change RTKLIB metrics")
    leakage = {
        "reference_opened_after_positions_serialized": True,
        "reference_passed_to_rtklib_execution": False,
        "perturbation_ecef_m": PERTURBATION_ECEF_M.tolist(),
        "raw_solution_timestamps_unchanged": raw_unchanged["timestamp_gpst_like_s"],
        "raw_ecef_positions_unchanged": raw_unchanged["position_ecef_m"],
        "epoch_support_matching_unchanged": all(
            raw_unchanged[name]
            for name in (
                "required_epoch_id",
                "required_timestamp_gpst_like_s",
                "matched_solution_index",
                "support_complete",
                "support_sha256",
            )
        ),
        "solver_status_unchanged": all(
            raw_unchanged[name] for name in ("solution_quality", "satellite_count")
        ),
        "configuration_unchanged": raw_unchanged["config_sha256"],
        "input_hashes_unchanged": all(
            raw_unchanged[name]
            for name in ("observation_sha256", "navigation_sha256")
        ),
        "enu_and_2d_3d_metrics_changed": metrics_changed,
        "only_evaluation_metrics_changed": all(raw_unchanged.values())
        and (metrics_changed or primary["metrics"] is None),
    }
    comparison = (
        controlled_comparison(primary["metrics"])
        if primary["metrics"] is not None
        else None
    )
    return primary, comparison, leakage


def rtklib_command(
    binary: Path,
    output_path: Path,
    *,
    start: str | None = None,
    end: str | None = None,
) -> list[str]:
    arguments = [str(binary), "-k", str(CONFIG), "-o", str(output_path)]
    if (start is None) != (end is None):
        raise ValueError("start and end must be provided together")
    if start is not None and end is not None:
        arguments.extend(["-ts", *start.split(), "-te", *end.split()])
    arguments.extend([str(OBSERVATION), str(NAVIGATION)])
    return arguments


def execute_rtklib(
    binary: Path,
    output_path: Path,
    *,
    start: str | None = None,
    end: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"rnx2rtkp is not executable: {binary}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        rtklib_command(binary, output_path, start=start, end=end),
        check=True,
        capture_output=True,
        text=True,
    )


def solve_full_interval(binary: Path, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Run full RINEX first and serialize positions, without accessing GT."""

    rtklib_before = verify_rtklib_identity()
    input_hashes = verify_scientific_inputs(include_reference=False)
    support = load_frozen_primary_support()
    config_hash = sha256_file(CONFIG)
    raw_path = output_dir / "rtklib_raw_full.pos"
    positions_path = output_dir / "rtklib_positions_pre_ground_truth.npz"
    completed = execute_rtklib(binary, raw_path)
    solutions = parse_rtklib_pos(raw_path)
    match = match_required_support(solutions, support)
    serialize_pre_ground_truth(
        positions_path,
        solutions,
        support,
        match,
        config_sha256=config_hash,
        observation_sha256=input_hashes["observation"],
        navigation_sha256=input_hashes["navigation"],
        raw_output_sha256=sha256_file(raw_path),
    )
    rtklib_after = verify_rtklib_identity()
    assert_checkout_unchanged(rtklib_before, rtklib_after)
    report = {
        "schema_version": 1,
        "phase": "pre_ground_truth_solve",
        "full_observation_interval_processed": True,
        "ground_truth_available_to_execution": False,
        "command": rtklib_command(binary, raw_path),
        "stderr": completed.stderr,
        "binary": str(binary.resolve()),
        "binary_sha256": sha256_file(binary),
        "raw_output": str(raw_path.resolve()),
        "raw_output_sha256": sha256_file(raw_path),
        "positions_pre_ground_truth": str(positions_path.resolve()),
        "positions_pre_ground_truth_sha256": sha256_file(positions_path),
        "support_sha256": support.support_sha256,
        "support": {key: value for key, value in match.items() if key != "matched_solution_indices"},
        "configuration_sha256": config_hash,
        "input_hashes": input_hashes,
        "rtklib_before": rtklib_before,
        "rtklib_after": rtklib_after,
        "rtklib_checkout_unchanged": True,
    }
    write_json(output_dir / "solve_report.json", report)
    return report


def finalize_evaluation(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    positions_path = output_dir / "rtklib_positions_pre_ground_truth.npz"
    solve_report_path = output_dir / "solve_report.json"
    solve_report = json.loads(solve_report_path.read_text(encoding="utf-8"))
    input_hashes = verify_scientific_inputs(include_reference=True)
    if input_hashes["observation"] != solve_report["input_hashes"]["observation"]:
        raise RuntimeError("observation changed between solve and evaluation")
    if input_hashes["navigation"] != solve_report["input_hashes"]["navigation"]:
        raise RuntimeError("navigation changed between solve and evaluation")
    primary, comparison, leakage = evaluate_serialized_positions(positions_path)
    metrics_document = {
        "schema_version": 1,
        "support": "primary_current_stack_exact_frozen_timestamps",
        **primary,
    }
    write_json(output_dir / "rtklib_primary_metrics.json", metrics_document)
    if comparison is not None:
        write_json(output_dir / "controlled_comparison.json", comparison)

    project = git_provenance(PROJECT_ROOT)
    rtklib = verify_rtklib_identity()
    manifest = {
        "schema_version": 1,
        "status": (
            "complete" if primary["status"] == "complete_exact_support" else "incomplete"
        ),
        "scientific_question": (
            "official RTKLIB conventional SPP versus neutral TASGNSS and frozen "
            "KLT3 HybridShareSysNet + TASGNSS on exact primary Ibiza support"
        ),
        "project_git": project,
        "official_rtklib_git": rtklib,
        "official_rtklib_checkout_clean": rtklib["clean"],
        "input_hashes": input_hashes,
        "configuration": {
            "path": str(CONFIG.resolve()),
            "sha256": sha256_file(CONFIG),
            "audit_path": str(CONFIGURATION_AUDIT.resolve()),
            "audit_sha256": sha256_file(CONFIGURATION_AUDIT),
        },
        "artifacts": {
            "raw_rtklib_output": {
                "path": solve_report["raw_output"],
                "sha256": solve_report["raw_output_sha256"],
            },
            "positions_pre_ground_truth": {
                "path": str(positions_path.resolve()),
                "sha256": sha256_file(positions_path),
            },
            "primary_metrics": {
                "path": str((output_dir / "rtklib_primary_metrics.json").resolve()),
                "sha256": sha256_file(output_dir / "rtklib_primary_metrics.json"),
            },
        },
        "frozen_primary_support_sha256": solve_report["support_sha256"],
        "counts": solve_report["support"],
        "metric_definitions": {
            "coordinate_transform": (
                "validation.tasgnss_ibiza_solver_audit.core.enu_errors; pymap3d "
                "ECEF-to-ENU about frozen reference"
            ),
            "radial_2d": "sqrt(east^2+north^2)",
            "radial_3d": "sqrt(east^2+north^2+up^2)",
            "statistics": "mean, median, RMS, NumPy linear P68/P95, maximum",
            "component_statistics": "East/North/Up RMS",
        },
        "ground_truth_boundary": {
            "reference_path": str(REFERENCE.resolve()),
            "reference_sha256": input_hashes["ground_truth"],
            "leakage_audit": leakage,
        },
        "exact_support_status": primary["status"],
    }
    if comparison is not None:
        manifest["artifacts"]["controlled_comparison"] = {
            "path": str((output_dir / "controlled_comparison.json").resolve()),
            "sha256": sha256_file(output_dir / "controlled_comparison.json"),
        }
    write_json(output_dir / "rtklib_ibiza_manifest.json", manifest)
    return manifest


def audit_pre_run() -> dict[str, Any]:
    """Ground-truth-free pre-run audit suitable before the full solve."""

    rtklib = verify_rtklib_identity()
    input_hashes = verify_scientific_inputs(include_reference=False)
    support = load_frozen_primary_support()
    return {
        "project": git_provenance(PROJECT_ROOT),
        "rtklib": rtklib,
        "rnx2rtkp_candidates": [
            str(path.resolve())
            for path in RTKLIB_ROOT.rglob("rnx2rtkp")
            if path.is_file() and os.access(path, os.X_OK)
        ],
        "input_hashes": input_hashes,
        "config": str(CONFIG.resolve()),
        "config_sha256": sha256_file(CONFIG),
        "configuration_audit": json.loads(CONFIGURATION_AUDIT.read_text(encoding="utf-8")),
        "frozen_support": {
            "source_candidate_epochs": int(support.candidate_epoch_ids.size),
            "exact_paired_epochs": int(support.required_epoch_ids.size),
            "all_ten_seeds_identical": True,
            "support_sha256": support.support_sha256,
            "first_timestamp_gpst_like_s": float(support.required_timestamps_gpst_s[0]),
            "last_timestamp_gpst_like_s": float(support.required_timestamps_gpst_s[-1]),
        },
        "ground_truth_opened": False,
    }
