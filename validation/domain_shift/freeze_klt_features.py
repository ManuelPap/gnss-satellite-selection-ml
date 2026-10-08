#!/usr/bin/env python3
"""Freeze exact KLT1/KLT2 paper-era raw features before neural inference."""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np

from validation.ibiza_generalization.runtime_cache import (
    DEFAULT_RUNTIME_DIR,
    PYRTKLIB_COMMIT,
    PYRTKLIB_VERSION,
    TDL_COMMIT,
    resolve_runtime,
)
from validation.paper_weightnet.held_out import (
    DATASET_SPECS,
    FEATURE_NAMES,
    FEATURE_UNITS,
    PreparedFeatureDataset,
    file_record,
    prepare_feature_dataset,
    resolve_feature_input_paths,
    sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = REPOSITORY_ROOT.parent
DEFAULT_DATA_ROOT = PHD_ROOT / "external_data/TDL-GNSS/data"
DEFAULT_SOURCE_ARCHIVE = PHD_ROOT / "external_data/TDL-GNSS/data.zip"
DEFAULT_OUTPUT = PHD_ROOT / "external_data/domain_shift/klt_features"
DEFAULT_HISTORICAL_REFERENCE = (
    REPOSITORY_ROOT / "results/paper_weightnet_seed_sensitivity/seed_0.json"
)
DEFAULT_KLT1_WEIGHT_TRACE = (
    REPOSITORY_ROOT / "validation/paper_weightnet/klt1_nn_weight_trace.npz"
)
DEFAULT_KLT1_PAPER_TRACE = (
    REPOSITORY_ROOT / "validation/real_klt/paper_epoch_trace.npz"
)
DEFAULT_IBIZA_FEATURES = (
    PHD_ROOT / "external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz"
)

SOURCE_ARCHIVE_SHA256 = (
    "2afd7b1e395f8494e6992d1e109f9e8d9d83d992bf6d446d83d91aaf46cfc721"
)
KLT1_WEIGHT_TRACE_SHA256 = (
    "8937ac92c28d39f158282c73716030e67baa397d7d60ae1e0667495368f0c92f"
)
KLT1_PAPER_TRACE_SHA256 = (
    "e18be4fc63147e2a2fab56d0fe2f04e249bf653c844fecbb6a4c502a29ea95ef"
)
IBIZA_FEATURE_SHA256 = (
    "edb0189e9eadf3266d75984e3041a90306dd44b3ebecc0101eb188f933dc88c5"
)
EXPECTED_COUNTS = {
    "KLT1": {"epochs": 203, "rows": 4676},
    "KLT2": {"epochs": 209, "rows": 4914},
}
OUTPUT_FILENAMES = {
    "KLT1": "klt1_paper_features.npz",
    "KLT2": "klt2_paper_features.npz",
}
FEATURE_COLUMNS = (
    {"index": 0, "name": "C/N0", "source": "SNR[0]/1000", "unit": "SNR[0]/1000"},
    {"index": 1, "name": "elevation", "source": "historical final OLS azel[:,1]", "unit": "radian"},
    {"index": 2, "name": "OLS residual", "source": "historical equal-weight OLS residual", "unit": "metre"},
)
HISTORICAL_ARRAY_MAP = {
    "epoch_offsets": "epoch_offsets",
    "epoch_times": "epoch_times_gpst_like_s",
    "features": "features",
    "satellite_ids": "satellite_ids",
    "satellite_positions": "satellite_positions_ecef_m",
    "satellite_clock_bias": "satellite_clock_bias_s",
    "corrected_pseudorange": "corrected_pseudorange_m",
    "system_clock_indices": "system_clock_indices",
    "initial_states": "initial_states",
}
HISTORICAL_ARRAY_MEANINGS = {
    "epoch_offsets": "CSR-style boundaries formed from every full-held-out PreparedEpoch feature-row count",
    "epoch_times": "one observation timestamp from every full-held-out PreparedEpoch",
    "features": "concatenated full-held-out raw [SNR[0]/1000, elevation, OLS residual] rows",
    "satellite_ids": "concatenated retained satellite identifiers in full-held-out row order",
    "satellite_positions": "concatenated retained broadcast satellite ECEF positions",
    "satellite_clock_bias": "concatenated retained satellite clock biases",
    "corrected_pseudorange": "concatenated retained historical prange-corrected pseudoranges",
    "system_clock_indices": "concatenated retained seven-state constellation clock indices",
    "initial_states": "one seven-state equal-weight OLS solution per full-held-out epoch",
}
SCHEMA_DESCRIPTIONS = {
    "schema_version": "cache schema version",
    "dataset": "dataset label",
    "feature_column_names": "ordered human-readable feature names",
    "feature_column_units": "ordered raw feature units",
    "features": "raw pre-StandardizeLayer feature matrix",
    "cn0_snr0_div_1000": "feature column 0 copied from historical SNR output",
    "elevation_rad": "feature column 1 from final historical OLS geometry",
    "ols_residual_m": "feature column 2 from historical equal-weight OLS",
    "row_index": "zero-based global retained-row index",
    "epoch_index": "zero-based retained epoch index for each row",
    "row_index_within_epoch": "zero-based retained row order within each epoch",
    "epoch_offsets": "CSR-style row boundaries, length epoch_count + 1",
    "epoch_row_counts": "retained rows per epoch",
    "epoch_times_gpst_like_s": "historical observation epoch timestamp",
    "candidate_epoch_indices": "strict-window candidate index before OLS status rejection",
    "split_epoch_indices": "index in historical split_obs output",
    "satellite_ids": "pyrtklib satellite identifiers in retained row order",
    "satellite_numbers": "pyrtklib numeric satellite identifiers",
    "constellations": "leading system letter derived from satellite_ids",
    "raw_snr_units": "unscaled observation SNR[0] values",
    "raw_pseudorange_m": "raw observation P[0] values",
    "satellite_positions_ecef_m": "historical broadcast satellite positions",
    "satellite_clock_bias_s": "historical satellite clock biases",
    "corrected_pseudorange_m": "historical prange-corrected pseudoranges",
    "system_clock_indices": "seven-state constellation clock indices",
    "initial_states": "historical equal-weight OLS seven-state solutions",
}
HISTORICAL_RESULT_DIRECTORIES = (
    "results/paper_weightnet",
    "results/paper_biasnet",
    "results/paper_biasnet_corrected_gt",
    "results/paper_hybrid",
    "results/paper_weightnet_seed_sensitivity",
    "results/paper_biasnet_seed_sensitivity",
    "results/paper_hybrid_seed_sensitivity",
)


def _update_hash(digest: object, label: str, payload: bytes) -> None:
    encoded = label.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def array_sha256(values: np.ndarray) -> str:
    """Match the established seed-sensitivity array hash convention."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    _update_hash(digest, "dtype", array.dtype.str.encode("ascii"))
    _update_hash(digest, "shape", json.dumps(array.shape).encode("ascii"))
    _update_hash(digest, "values", array.tobytes(order="C"))
    return digest.hexdigest()


def named_array_sha256(values: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(values, key=lambda item: item[0]):
        _update_hash(digest, name, array_sha256(array).encode("ascii"))
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _verified_file(path: Path, expected: str, label: str) -> dict[str, object]:
    record = file_record(path)
    if record["sha256"] != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {record['sha256']}"
        )
    return record


def _repository_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout)


def _tree_snapshot() -> dict[str, object]:
    records: dict[str, str] = {}
    for relative in HISTORICAL_RESULT_DIRECTORIES:
        directory = REPOSITORY_ROOT / relative
        if not directory.is_dir():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            records[str(path.relative_to(REPOSITORY_ROOT))] = sha256(path)
    return {
        "file_count": len(records),
        "snapshot_sha256": _json_hash(records),
    }


def _file_snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256(path.resolve()) for path in sorted(set(paths))}


def build_cache_arrays(prepared: PreparedFeatureDataset) -> dict[str, np.ndarray]:
    epochs = prepared.epochs
    counts = np.asarray([epoch.features.shape[0] for epoch in epochs], dtype=np.int64)
    offsets = np.zeros(len(epochs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    features = np.concatenate([epoch.features for epoch in epochs]).astype(
        np.float64, copy=False
    )
    satellite_ids = np.concatenate([epoch.satellite_ids for epoch in epochs])
    epoch_index = np.repeat(np.arange(len(epochs), dtype=np.int64), counts)
    row_within_epoch = np.concatenate(
        [np.arange(count, dtype=np.int64) for count in counts]
    )
    arrays = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "dataset": np.asarray([prepared.spec.name], dtype="U4"),
        "feature_column_names": np.asarray(FEATURE_NAMES, dtype="U16"),
        "feature_column_units": np.asarray(FEATURE_UNITS, dtype="U32"),
        "features": features,
        "cn0_snr0_div_1000": features[:, 0].copy(),
        "elevation_rad": features[:, 1].copy(),
        "ols_residual_m": features[:, 2].copy(),
        "row_index": np.arange(features.shape[0], dtype=np.int64),
        "epoch_index": epoch_index,
        "row_index_within_epoch": row_within_epoch,
        "epoch_offsets": offsets,
        "epoch_row_counts": counts,
        "epoch_times_gpst_like_s": np.asarray(
            [epoch.epoch_time for epoch in epochs], dtype=np.float64
        ),
        "candidate_epoch_indices": np.asarray(
            [epoch.candidate_epoch_index for epoch in epochs], dtype=np.int64
        ),
        "split_epoch_indices": np.asarray(
            [epoch.split_epoch_index for epoch in epochs], dtype=np.int64
        ),
        "satellite_ids": satellite_ids,
        "satellite_numbers": np.concatenate(
            [epoch.satellite_numbers for epoch in epochs]
        ).astype(np.int64, copy=False),
        "constellations": np.asarray([item[0] for item in satellite_ids], dtype="U1"),
        "raw_snr_units": np.concatenate(
            [epoch.raw_snr_units for epoch in epochs]
        ).astype(np.float64, copy=False),
        "raw_pseudorange_m": np.concatenate(
            [epoch.raw_pseudorange_m for epoch in epochs]
        ).astype(np.float64, copy=False),
        "satellite_positions_ecef_m": np.concatenate(
            [epoch.satellite_positions_ecef_m for epoch in epochs]
        ).astype(np.float64, copy=False),
        "satellite_clock_bias_s": np.concatenate(
            [epoch.satellite_clock_bias_s for epoch in epochs]
        ).astype(np.float64, copy=False),
        "corrected_pseudorange_m": np.concatenate(
            [epoch.corrected_pseudorange_m for epoch in epochs]
        ).astype(np.float64, copy=False),
        "system_clock_indices": np.concatenate(
            [epoch.system_clock_indices for epoch in epochs]
        ).astype(np.int64, copy=False),
        "initial_states": np.stack(
            [epoch.initial_ols_state for epoch in epochs]
        ).astype(np.float64, copy=False),
    }
    return arrays


def validate_cache_arrays(
    dataset: str,
    prepared: PreparedFeatureDataset,
    arrays: Mapping[str, np.ndarray],
) -> None:
    expected = EXPECTED_COUNTS[dataset]
    epoch_count = len(prepared.epochs)
    row_count = int(arrays["features"].shape[0])
    if epoch_count != expected["epochs"] or row_count != expected["rows"]:
        raise RuntimeError(
            f"{dataset} count mismatch: got {epoch_count} epochs/{row_count} rows; "
            f"expected {expected['epochs']}/{expected['rows']}"
        )
    if prepared.candidate_epoch_count != epoch_count or prepared.invalid_epochs:
        raise RuntimeError(
            f"{dataset} historical validity mismatch: candidates="
            f"{prepared.candidate_epoch_count}, invalid={prepared.invalid_epochs!r}"
        )
    features = arrays["features"]
    if features.shape != (row_count, 3) or features.dtype != np.float64:
        raise RuntimeError(f"{dataset} unexpected feature matrix schema")
    columns = np.column_stack(
        (
            arrays["cn0_snr0_div_1000"],
            arrays["elevation_rad"],
            arrays["ols_residual_m"],
        )
    )
    if not np.array_equal(features, columns):
        raise RuntimeError(f"{dataset} feature matrix does not equal named columns")
    if not np.array_equal(features[:, 0], arrays["raw_snr_units"] / 1000.0):
        raise RuntimeError(f"{dataset} feature column 0 is not raw SNR[0]/1000")
    if tuple(arrays["feature_column_names"].tolist()) != FEATURE_NAMES:
        raise RuntimeError(f"{dataset} feature-name order changed")
    if tuple(arrays["feature_column_units"].tolist()) != FEATURE_UNITS:
        raise RuntimeError(f"{dataset} feature-unit order changed")
    offsets = arrays["epoch_offsets"]
    counts = arrays["epoch_row_counts"]
    if (
        offsets.shape != (epoch_count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != row_count
        or not np.array_equal(np.diff(offsets), counts)
        or int(counts.sum()) != row_count
    ):
        raise RuntimeError(f"{dataset} epoch boundaries do not reconcile")
    if not np.array_equal(
        arrays["constellations"],
        np.asarray([item[0] for item in arrays["satellite_ids"]], dtype="U1"),
    ):
        raise RuntimeError(f"{dataset} constellation labels do not match satellite IDs")
    if not np.array_equal(arrays["row_index"], np.arange(row_count)):
        raise RuntimeError(f"{dataset} global row ordering is not contiguous")
    for name, values in arrays.items():
        if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
            raise RuntimeError(f"{dataset} array {name} contains non-finite values")


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write byte-stable compressed NPZ entries with fixed ZIP metadata."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite scientific artifact: {path}")
    with zipfile.ZipFile(
        path, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, array in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compresslevel=9)


def _schema(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "description": SCHEMA_DESCRIPTIONS[name],
        }
        for name, values in arrays.items()
    }


def _historical_cross_check(
    dataset: str,
    arrays: Mapping[str, np.ndarray],
    reference: Mapping[str, object],
) -> dict[str, object]:
    held_out = reference["controls"]["held_out_data"]["datasets"][dataset]  # type: ignore[index]
    expected_hashes = held_out["array_sha256"]  # type: ignore[index]
    comparisons: dict[str, object] = {}
    for historical_name, cache_name in HISTORICAL_ARRAY_MAP.items():
        actual = array_sha256(arrays[cache_name])
        expected = str(expected_hashes[historical_name])  # type: ignore[index]
        comparisons[historical_name] = {
            "cache_array": cache_name,
            "meaning": HISTORICAL_ARRAY_MEANINGS[historical_name],
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        }
    mismatches = [name for name, value in comparisons.items() if not value["matches"]]  # type: ignore[index]
    if mismatches:
        raise RuntimeError(f"{dataset} differs from historical held-out rows: {mismatches}")
    expected_counts = (
        int(held_out["valid_epoch_count"]),  # type: ignore[index]
        int(held_out["retained_measurement_count"]),  # type: ignore[index]
    )
    actual_counts = (len(arrays["epoch_offsets"]) - 1, len(arrays["features"]))
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"{dataset} historical-reference counts differ: {actual_counts} != {expected_counts}"
        )
    return {
        "all_available_non_ground_truth_array_hashes_match": True,
        "matched_array_count": len(comparisons),
        "array_comparisons": comparisons,
        "same_stage_authority": {
            "artifact_field": f"controls.held_out_data.datasets.{dataset}",
            "producer": "validation.paper_weightnet_seed_sensitivity.run_seed.prepared_dataset_identity",
            "producer_input": "PreparedDataset returned by validation.paper_weightnet.held_out.prepare_dataset",
            "support": "complete retained multi-constellation epoch",
            "proof": (
                "prepared_dataset_identity concatenates these arrays directly from "
                "PreparedDataset.epochs; prepare_dataset obtains that same row set from "
                "prepare_feature_dataset before attaching ground truth"
            ),
            "ground_truth_arrays_excluded": ["gt_times", "ground_truth"],
        },
        "historical_dataset_identity_sha256": held_out["identity_sha256"],  # type: ignore[index]
        "identity_hash_reproduced": False,
        "identity_hash_limitation": (
            "The historical identity includes gt_times and ground_truth arrays; "
            "this pre-inference cache deliberately excludes both."
        ),
    }


def _gps_only_supplementary_diagnostic(
    arrays: Mapping[str, np.ndarray], weight_trace: Path, paper_trace: Path
) -> dict[str, object]:
    first_stop = int(arrays["epoch_offsets"][1])
    first_ids = arrays["satellite_ids"][:first_stop]
    by_id = {str(item): index for index, item in enumerate(first_ids)}
    with np.load(weight_trace, allow_pickle=False) as trace:
        trace_ids = np.asarray(trace["satellite_ids"])
        rows = np.asarray([by_id[str(item)] for item in trace_ids], dtype=np.int64)
        full_support_features = arrays["features"][rows]
        gps_only_features = np.asarray(trace["features"])
        gps_only_ols_state = np.asarray(trace["ols_initial_state"])
    feature_difference = full_support_features - gps_only_features
    if not np.array_equal(full_support_features[:, 0], gps_only_features[:, 0]):
        raise RuntimeError("overlapping GPS observations changed raw C/N0")

    paper_checks: dict[str, bool] = {}
    with np.load(paper_trace, allow_pickle=False) as trace:
        paper_ids = np.asarray(trace["satellite_ids"])
        paper_rows = np.asarray([by_id[str(item)] for item in paper_ids], dtype=np.int64)
        paper_checks = {
            "epoch_time": bool(
                np.array_equal(
                    np.asarray(arrays["epoch_times_gpst_like_s"][0]),
                    np.asarray(trace["epoch_time"]),
                )
            ),
            "satellite_ids": bool(
                np.array_equal(arrays["satellite_ids"][paper_rows], paper_ids)
            ),
            "raw_pseudorange_m": bool(
                np.array_equal(
                    arrays["raw_pseudorange_m"][paper_rows], trace["raw_pseudorange_m"]
                )
            ),
            "satellite_positions_ecef_m": bool(
                np.array_equal(
                    arrays["satellite_positions_ecef_m"][paper_rows],
                    trace["satellite_positions_ecef_m"],
                )
            ),
            "satellite_clock_bias_s": bool(
                np.array_equal(
                    arrays["satellite_clock_bias_s"][paper_rows],
                    trace["satellite_clock_bias_s"],
                )
            ),
            "corrected_pseudorange_m": bool(
                np.array_equal(
                    arrays["corrected_pseudorange_m"][paper_rows],
                    np.asarray(trace["corrected_pseudorange_m"]).reshape(-1),
                )
            ),
        }
        paper_has_features = "features" in trace.files
    if not all(paper_checks.values()):
        failed = [name for name, passed in paper_checks.items() if not passed]
        raise RuntimeError(f"KLT1 paper epoch trace mismatch: {failed}")
    full_support_ols_state = arrays["initial_states"][0]
    return {
        "classification": "non-applicable as an exact full-held-out feature regression oracle",
        "reason": (
            "Both traces select eight GPS satellites before get_ls_pnt_pos; the "
            "full held-out stage calls get_ls_pnt_pos on the complete 17-satellite "
            "multi-constellation epoch, so the OLS design matrix, estimated state, "
            "and residuals differ. This is a support-stage difference, not numerical error."
        ),
        "used_as_same_stage_regression_authority": False,
        "full_held_out_support_count": first_stop,
        "gps_only_support_count": len(rows),
        "klt1_nn_weight_trace": {
            "path": str(weight_trace.resolve()),
            "sha256": sha256(weight_trace),
            "rows_compared": len(rows),
            "satellite_ids": trace_ids.tolist(),
            "cn0_exactly_equal": True,
            "maximum_absolute_elevation_difference_rad": float(
                np.max(np.abs(feature_difference[:, 1]))
            ),
            "maximum_absolute_ols_residual_difference_m": float(
                np.max(np.abs(feature_difference[:, 2]))
            ),
            "ols_residuals_expected_to_match": False,
            "ols_states_expected_to_match": False,
            "ols_states_exactly_equal": bool(
                np.array_equal(full_support_ols_state, gps_only_ols_state)
            ),
            "maximum_absolute_ols_state_difference": float(
                np.max(np.abs(full_support_ols_state - gps_only_ols_state))
            ),
        },
        "paper_epoch_trace": {
            "path": str(paper_trace.resolve()),
            "sha256": sha256(paper_trace),
            "rows_compared": len(paper_rows),
            "contains_raw_feature_matrix": paper_has_features,
            "supplementary_common_input_checks": paper_checks,
            "feature_comparison_limitation": (
                "paper_epoch_trace.npz does not store the raw three-column feature "
                "matrix and its OLS state is from GPS-only support"
            ),
        },
    }


def _compare_runs(
    first: Mapping[str, dict[str, object]], second: Mapping[str, dict[str, object]]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for dataset in EXPECTED_COUNTS:
        first_arrays = first[dataset]["arrays"]
        second_arrays = second[dataset]["arrays"]
        if first_arrays.keys() != second_arrays.keys():  # type: ignore[union-attr]
            raise RuntimeError(f"{dataset} repeated run changed array names")
        unequal = [
            name
            for name in first_arrays  # type: ignore[union-attr]
            if not np.array_equal(first_arrays[name], second_arrays[name])  # type: ignore[index]
        ]
        if unequal:
            raise RuntimeError(f"{dataset} repeated preprocessing differs: {unequal}")
        first_path = first[dataset]["path"]
        second_path = second[dataset]["path"]
        first_file_hash = sha256(first_path)  # type: ignore[arg-type]
        second_file_hash = sha256(second_path)  # type: ignore[arg-type]
        first_content_hash = named_array_sha256(first_arrays.items())  # type: ignore[union-attr]
        second_content_hash = named_array_sha256(second_arrays.items())  # type: ignore[union-attr]
        if first_file_hash != second_file_hash or first_content_hash != second_content_hash:
            raise RuntimeError(f"{dataset} repeated cache hashes differ")
        result[dataset] = {
            "arrays_exactly_equal": True,
            "npz_bytes_identical": True,
            "first_npz_sha256": first_file_hash,
            "second_npz_sha256": second_file_hash,
            "first_content_sha256": first_content_hash,
            "second_content_sha256": second_content_hash,
        }
    return result


def _one_run(
    directory: Path,
    inputs: Mapping[str, object],
    historical_reference: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for dataset in EXPECTED_COUNTS:
        prepared = prepare_feature_dataset(DATASET_SPECS[dataset], inputs[dataset])  # type: ignore[arg-type]
        arrays = build_cache_arrays(prepared)
        validate_cache_arrays(dataset, prepared, arrays)
        cross_check = _historical_cross_check(dataset, arrays, historical_reference)
        path = directory / OUTPUT_FILENAMES[dataset]
        write_deterministic_npz(path, arrays)
        result[dataset] = {
            "arrays": arrays,
            "path": path,
            "prepared": prepared,
            "historical_cross_check": cross_check,
        }
    return result


def _constellation_counts(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(values, return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}


def freeze_feature_caches(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    data_root: Path = DEFAULT_DATA_ROOT,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    historical_reference_path: Path = DEFAULT_HISTORICAL_REFERENCE,
    weight_trace: Path = DEFAULT_KLT1_WEIGHT_TRACE,
    paper_trace: Path = DEFAULT_KLT1_PAPER_TRACE,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    runtime, runtime_manifest = resolve_runtime(runtime_dir)
    inputs = {
        dataset: resolve_feature_input_paths(
            DATASET_SPECS[dataset], data_root=data_root, runtime_dir=runtime_dir
        )
        for dataset in EXPECTED_COUNTS
    }
    archive_record = _verified_file(
        source_archive.resolve(), SOURCE_ARCHIVE_SHA256, "KLT source archive"
    )
    weight_trace_record = _verified_file(
        weight_trace.resolve(), KLT1_WEIGHT_TRACE_SHA256, "KLT1 WeightNet trace"
    )
    paper_trace_record = _verified_file(
        paper_trace.resolve(), KLT1_PAPER_TRACE_SHA256, "KLT1 paper epoch trace"
    )
    ibiza_record = _verified_file(
        DEFAULT_IBIZA_FEATURES, IBIZA_FEATURE_SHA256, "frozen Ibiza feature artifact"
    )
    historical_reference = _load_json(historical_reference_path.resolve())
    historical_reference_record = file_record(historical_reference_path.resolve())

    protected_paths = {
        source_archive.resolve(),
        runtime.manifest,
        runtime.tdl_dir / "rtk_util.py",
        next((runtime.pyrtklib_site / "pyrtklib").glob("pyrtklib*.so")),
        weight_trace.resolve(),
        paper_trace.resolve(),
        DEFAULT_IBIZA_FEATURES.resolve(),
        historical_reference_path.resolve(),
    }
    for item in inputs.values():
        protected_paths.add(item.observation)
        for pattern in item.ephemeris_patterns:
            protected_paths.update(Path(path).resolve() for path in glob.glob(pattern))
    protected_before = _file_snapshot(protected_paths)
    historical_results_before = _tree_snapshot()

    first_dir = Path(
        tempfile.mkdtemp(prefix=".klt_features_repeat_1_", dir=output_dir.parent)
    )
    second_dir = Path(
        tempfile.mkdtemp(prefix=".klt_features_repeat_2_", dir=output_dir.parent)
    )
    try:
        first = _one_run(first_dir, inputs, historical_reference)
        second = _one_run(second_dir, inputs, historical_reference)
        determinism = _compare_runs(first, second)
        gps_only_diagnostics = _gps_only_supplementary_diagnostic(
            second["KLT1"]["arrays"], weight_trace.resolve(), paper_trace.resolve()  # type: ignore[arg-type]
        )

        protected_after = _file_snapshot(protected_paths)
        historical_results_after = _tree_snapshot()
        if protected_before != protected_after:
            raise RuntimeError("a protected source or frozen artifact changed")
        if historical_results_before != historical_results_after:
            raise RuntimeError("an existing historical result artifact changed")

        repository_commit = _repository_commit()
        datasets: dict[str, object] = {}
        for dataset in EXPECTED_COUNTS:
            arrays = second[dataset]["arrays"]  # type: ignore[assignment]
            prepared = second[dataset]["prepared"]
            temporary_path = second[dataset]["path"]
            datasets[dataset] = {
                "artifact": {
                    "path": str((output_dir / OUTPUT_FILENAMES[dataset]).resolve()),
                    "filename": OUTPUT_FILENAMES[dataset],
                    "size_bytes": temporary_path.stat().st_size,  # type: ignore[union-attr]
                    "sha256": sha256(temporary_path),  # type: ignore[arg-type]
                    "content_sha256": named_array_sha256(arrays.items()),  # type: ignore[union-attr]
                },
                "strict_interval_gpst_like_s": [
                    DATASET_SPECS[dataset].start_time,
                    DATASET_SPECS[dataset].end_time,
                ],
                "epoch_count": len(prepared.epochs),  # type: ignore[union-attr]
                "row_count": int(arrays["features"].shape[0]),  # type: ignore[index]
                "candidate_epoch_count": prepared.candidate_epoch_count,  # type: ignore[union-attr]
                "invalid_epoch_count": len(prepared.invalid_epochs),  # type: ignore[union-attr]
                "constellation_counts": _constellation_counts(arrays["constellations"]),  # type: ignore[index]
                "all_features_finite": bool(np.all(np.isfinite(arrays["features"]))),  # type: ignore[index]
                "source_inputs": prepared.input_provenance,  # type: ignore[union-attr]
                "npz_schema": _schema(arrays),  # type: ignore[arg-type]
                "historical_cross_check": second[dataset]["historical_cross_check"],
            }

        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "passed",
            "purpose": "immutable KLT1/KLT2 raw pre-inference feature caches",
            "repository": {
                "path": str(REPOSITORY_ROOT),
                "commit": repository_commit,
                "worktree_dirty": _repository_dirty(),
                "generator": file_record(Path(__file__)),
                "historical_preparation_module": file_record(
                    REPOSITORY_ROOT / "validation/paper_weightnet/held_out.py"
                ),
            },
            "provenance": {
                "tdl_gnss_commit": TDL_COMMIT,
                "pyrtklib_version": PYRTKLIB_VERSION,
                "pyrtklib_hypothesis_commit": PYRTKLIB_COMMIT,
                "pyrtklib_publication_version_uncertain": True,
                "runtime_manifest": file_record(runtime.manifest),
                "runtime_manifest_values": runtime_manifest,
                "pyrtklib_binary": file_record(
                    next((runtime.pyrtklib_site / "pyrtklib").glob("pyrtklib*.so"))
                ),
                "tdl_rtk_util": file_record(runtime.tdl_dir / "rtk_util.py"),
                "source_archive": archive_record,
            },
            "feature_contract": {
                "columns": FEATURE_COLUMNS,
                "stored_dtype": "float64",
                "normalization_fitted": False,
                "normalization_applied": False,
                "standardize_layer_invoked": False,
            },
            "ground_truth": {
                "ground_truth_used_for_feature_generation": False,
                "ground_truth_path_resolved": False,
                "ground_truth_file_read": False,
                "ground_truth_present_in_cache": False,
                "separation": (
                    "resolve_feature_input_paths and prepare_feature_dataset expose no "
                    "ground-truth path or value; evaluation ground truth is attached only "
                    "by the separate prepare_dataset wrapper"
                ),
            },
            "execution_boundary": {
                "historical_preprocessing_invoked": True,
                "historical_equal_weight_ols_invoked": True,
                "checkpoint_loaded": False,
                "neural_model_constructed": False,
                "neural_forward_pass_invoked": False,
                "optimizer_constructed": False,
                "backward_invoked": False,
                "training_invoked": False,
                "learned_wls_invoked": False,
                "learned_position_computed": False,
            },
            "historical_reference": historical_reference_record,
            "datasets": datasets,
            "supplementary_gps_only_diagnostics": gps_only_diagnostics,
            "determinism": {
                "preprocessing_runs": 2,
                "fresh_temporary_output_locations": 2,
                "result": "exact arrays and byte-identical deterministic NPZ files",
                "datasets": determinism,
            },
            "immutability": {
                "source_and_frozen_file_count": len(protected_before),
                "source_and_frozen_snapshot_before_sha256": _json_hash(protected_before),
                "source_and_frozen_snapshot_after_sha256": _json_hash(protected_after),
                "source_and_frozen_files_unchanged": True,
                "historical_results_before": historical_results_before,
                "historical_results_after": historical_results_after,
                "historical_results_unchanged": True,
                "klt1_weight_trace": weight_trace_record,
                "klt1_paper_trace": paper_trace_record,
                "ibiza_feature_artifact": ibiza_record,
            },
        }
        manifest_path = second_dir / "feature_cache_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.rmtree(first_dir)
        second_dir.rename(output_dir)
        return manifest
    except BaseException:
        if first_dir.exists():
            shutil.rmtree(first_dir)
        if second_dir.exists():
            shutil.rmtree(second_dir)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument(
        "--historical-reference", type=Path, default=DEFAULT_HISTORICAL_REFERENCE
    )
    parser.add_argument("--weight-trace", type=Path, default=DEFAULT_KLT1_WEIGHT_TRACE)
    parser.add_argument("--paper-trace", type=Path, default=DEFAULT_KLT1_PAPER_TRACE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = freeze_feature_caches(
        output_dir=args.output_dir,
        data_root=args.data_root,
        runtime_dir=args.runtime_dir,
        source_archive=args.source_archive,
        historical_reference_path=args.historical_reference,
        weight_trace=args.weight_trace,
        paper_trace=args.paper_trace,
    )
    concise = {
        dataset: {
            "epochs": manifest["datasets"][dataset]["epoch_count"],  # type: ignore[index]
            "rows": manifest["datasets"][dataset]["row_count"],  # type: ignore[index]
            "sha256": manifest["datasets"][dataset]["artifact"]["sha256"],  # type: ignore[index]
        }
        for dataset in EXPECTED_COUNTS
    }
    print(json.dumps({"output": str(args.output_dir.resolve()), **concise}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
