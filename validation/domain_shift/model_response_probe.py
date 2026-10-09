#!/usr/bin/env python3
"""Frozen-network response probe for exact KLT1, KLT2, and Ibiza rows.

The executable boundary in this module is deliberately narrow: verified raw
features -> checkpoint StandardizeLayer -> frozen network -> stored response.
It neither imports nor calls a positioning solver, ground-truth loader,
optimizer, or training routine.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch import nn

from validation.ibiza_generalization.checkpoints import (
    ARCHITECTURES,
    DEFAULT_MANIFEST as DEFAULT_CHECKPOINT_MANIFEST,
    REPOSITORY_ROOT,
    FrozenModel,
    load_frozen_model,
    sha256_file,
    validate_checkpoint_inventory,
)


PHD_ROOT = REPOSITORY_ROOT.parent
DEFAULT_OUTPUT = PHD_ROOT / "external_data/domain_shift/model_response"
DEFAULT_KLT1 = PHD_ROOT / "external_data/domain_shift/klt_features/klt1_paper_features.npz"
DEFAULT_KLT2 = PHD_ROOT / "external_data/domain_shift/klt_features/klt2_paper_features.npz"
DEFAULT_IBIZA = PHD_ROOT / "external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz"

INPUT_SHA256 = {
    "KLT1": "6912dc01ce126e006a1a91e12015f49f023f3dd4e94ea05a17b6f5ee07228ee5",
    "KLT2": "088d79ad05bebd8bba4084693b8af52a2ef14787f8e3328505f5cdff48ee8a3d",
    "Ibiza": "edb0189e9eadf3266d75984e3041a90306dd44b3ebecc0101eb188f933dc88c5",
}
EXPECTED_ROWS = {"KLT1": 4676, "KLT2": 4914, "Ibiza": 73204}
FEATURE_NAMES = ("C/N0", "elevation", "OLS residual")
FEATURE_UNITS = ("SNR[0]/1000", "radian", "metre")
FROZEN_MEAN = np.asarray(
    [29.084903717041016, 0.8471900224685669, -1.7873233559839719e-07],
    dtype=np.float64,
)
FROZEN_STD = np.asarray(
    [5.891841888427734, 0.2879891097545624, 4.3018035888671875],
    dtype=np.float64,
)
OUTPUT_SEMANTICS = {
    "TDL-B": ("bias_m",),
    "TDL-W": ("weight",),
    "TDL-BW": ("weight", "bias_m"),
}
ARCHITECTURE_SLUGS = {"TDL-B": "tdl_b", "TDL-W": "tdl_w", "TDL-BW": "tdl_bw"}
CODE_TO_CONSTELLATION = {1: "G", 2: "R", 3: "E", 4: "C"}
PERCENTILE_METHOD = "linear"

FEATURE_RELATIONSHIPS = {
    "cn0": (0, np.arange(0.0, 65.0, 5.0, dtype=np.float64), "SNR[0]/1000"),
    "elevation": (
        1,
        np.deg2rad(np.arange(0.0, 105.0, 15.0, dtype=np.float64)),
        "radian",
    ),
    "ols_residual": (
        2,
        np.asarray(
            [-1000, -100, -50, -25, -10, -5, -2, 0, 2, 5, 10, 25, 50, 100, 250, 1000],
            dtype=np.float64,
        ),
        "metre",
    ),
    "abs_ols_residual": (
        2,
        np.asarray([0, 1, 2, 5, 10, 25, 50, 100, 250, 1000], dtype=np.float64),
        "metre",
    ),
}
HIGH_CN0_KLT3_MAX = 39.0


@dataclass(frozen=True)
class FeatureDataset:
    name: str
    source_path: Path
    features: np.ndarray
    row_index: np.ndarray
    epoch_index: np.ndarray
    satellite_ids: np.ndarray
    constellations: np.ndarray
    extra_identity: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ForwardResult:
    normalized_features: np.ndarray
    outputs: Mapping[str, np.ndarray]
    eval_mode: bool
    gradients_disabled: bool
    state_dict_unchanged: bool
    repeated_forward_exact: bool
    state_sha256_before: str
    state_sha256_after: str


def _json_dump(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _constellation_labels(codes: np.ndarray) -> np.ndarray:
    try:
        return np.asarray([CODE_TO_CONSTELLATION[int(item)] for item in codes], dtype="U1")
    except KeyError as error:
        raise RuntimeError(f"unsupported Ibiza constellation code {error.args[0]}") from error


def _satellite_ids(constellations: np.ndarray, prns: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{system}{int(prn):02d}" for system, prn in zip(constellations, prns, strict=True)],
        dtype="U3",
    )


def _load_one(name: str, path: Path, expected_sha256: str) -> FeatureDataset:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{name} feature SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    features = np.asarray(arrays["features"])
    if features.dtype != np.float64 or features.shape != (EXPECTED_ROWS[name], 3):
        raise RuntimeError(f"{name} feature shape/dtype changed: {features.shape}/{features.dtype}")
    if not np.all(np.isfinite(features)):
        raise RuntimeError(f"{name} features contain non-finite values")
    named = np.column_stack(
        (arrays["cn0_snr0_div_1000"], arrays["elevation_rad"], arrays["ols_residual_m"])
    )
    if not np.array_equal(features, named):
        raise RuntimeError(f"{name} named feature columns changed order or values")
    count = features.shape[0]
    if name == "Ibiza":
        row_index = np.arange(count, dtype=np.int64)
        constellations = _constellation_labels(arrays["constellation_code"])
        satellite_ids = _satellite_ids(constellations, arrays["satellite_prn"])
        extras = {
            key: np.asarray(arrays[key])
            for key in (
                "source_row_index",
                "source_observation_index",
                "rtklib_satellite_number",
                "satellite_prn",
                "constellation_code",
            )
        }
    else:
        row_index = np.asarray(arrays["row_index"])
        constellations = np.asarray(arrays["constellations"])
        satellite_ids = np.asarray(arrays["satellite_ids"])
        extras = {
            "satellite_numbers": np.asarray(arrays["satellite_numbers"]),
            "row_index_within_epoch": np.asarray(arrays["row_index_within_epoch"]),
        }
    if not np.array_equal(row_index, np.arange(count, dtype=np.int64)):
        raise RuntimeError(f"{name} row indices are not contiguous in source order")
    row_arrays = {
        "epoch_index": np.asarray(arrays["epoch_index"]),
        "satellite_ids": satellite_ids,
        "constellations": constellations,
        **extras,
    }
    if any(values.shape != (count,) for values in row_arrays.values()):
        raise RuntimeError(f"{name} row identity arrays do not align")
    if not np.array_equal(constellations, np.asarray([item[0] for item in satellite_ids], dtype="U1")):
        raise RuntimeError(f"{name} satellite and constellation identities disagree")
    return FeatureDataset(
        name=name,
        source_path=path.resolve(),
        features=features,
        row_index=row_index,
        epoch_index=np.asarray(arrays["epoch_index"]),
        satellite_ids=satellite_ids,
        constellations=constellations,
        extra_identity=extras,
    )


def load_feature_datasets(
    klt1_path: Path = DEFAULT_KLT1,
    klt2_path: Path = DEFAULT_KLT2,
    ibiza_path: Path = DEFAULT_IBIZA,
) -> tuple[FeatureDataset, ...]:
    """Hash-check and load only the raw feature and row-identity arrays."""

    return tuple(
        _load_one(name, path.resolve(), INPUT_SHA256[name])
        for name, path in (("KLT1", klt1_path), ("KLT2", klt2_path), ("Ibiza", ibiza_path))
    )


def _update_hash(digest: object, label: str, payload: bytes) -> None:
    encoded = label.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    _update_hash(digest, "dtype", array.dtype.str.encode("ascii"))
    _update_hash(digest, "shape", json.dumps(array.shape).encode("ascii"))
    _update_hash(digest, "values", array.tobytes(order="C"))
    return digest.hexdigest()


def named_array_sha256(values: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(values, key=lambda item: item[0]):
        _update_hash(digest, name, array_sha256(np.asarray(array)).encode("ascii"))
    return digest.hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return named_array_sha256(
        (name, value.detach().cpu().numpy()) for name, value in state.items()
    )


def _state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _output_arrays(architecture: str, prediction: object) -> dict[str, np.ndarray]:
    if architecture == "TDL-B":
        if not isinstance(prediction, torch.Tensor):
            raise RuntimeError("TDL-B returned an unexpected output type")
        return {"bias_m": prediction.reshape(-1).detach().cpu().numpy().copy()}
    if architecture == "TDL-W":
        if not isinstance(prediction, torch.Tensor):
            raise RuntimeError("TDL-W returned an unexpected output type")
        return {"weight": prediction.reshape(-1).detach().cpu().numpy().copy()}
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        raise RuntimeError("TDL-BW must return the released (weight, bias) tuple")
    weight, bias = prediction
    return {
        "weight": weight.reshape(-1).detach().cpu().numpy().copy(),
        "bias_m": bias.reshape(-1).detach().cpu().numpy().copy(),
    }


def forward_only(
    frozen: FrozenModel,
    raw_features: np.ndarray,
    *,
    repeat: bool = True,
) -> ForwardResult:
    """Run one verified frozen model, with no operation after neural output."""

    model = frozen.model
    model.eval()
    if model.training:
        raise RuntimeError("model.eval() did not disable training mode")
    before = _state_snapshot(model)
    state_before_hash = _state_sha256(before)
    tensor = torch.as_tensor(raw_features, dtype=torch.float64, device="cpu")
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError("raw_features must have shape (rows, 3)")
    gradients_disabled = False
    with torch.inference_mode():
        gradients_disabled = not torch.is_grad_enabled()
        normalized = model.seq[0](tensor)
        first_prediction = model(tensor)
        first = _output_arrays(frozen.record.architecture, first_prediction)
        if repeat:
            second = _output_arrays(frozen.record.architecture, model(tensor))
        else:
            second = first
    if not gradients_disabled:
        raise RuntimeError("gradients were enabled inside the forward boundary")
    if tuple(first) != OUTPUT_SEMANTICS[frozen.record.architecture]:
        raise RuntimeError("released output semantics changed")
    if any(values.shape != (tensor.shape[0],) for values in first.values()):
        raise RuntimeError("neural output row count does not match input")
    if any(not np.all(np.isfinite(values)) for values in first.values()):
        raise RuntimeError("neural output contains non-finite values")
    repeated_exact = all(np.array_equal(first[key], second[key]) for key in first)
    if not repeated_exact:
        raise RuntimeError("repeated frozen forward passes differ")
    after = model.state_dict()
    unchanged = tuple(before) == tuple(after) and all(
        torch.equal(before[name], after[name]) for name in before
    )
    if not unchanged:
        raise RuntimeError("model state_dict changed during inference")
    state_after_hash = _state_sha256(after)
    normalized_array = normalized.detach().cpu().numpy().copy()
    expected_normalized = (raw_features - FROZEN_MEAN) / FROZEN_STD
    if not np.array_equal(normalized_array, expected_normalized):
        raise RuntimeError("checkpoint StandardizeLayer did not produce expected values")
    return ForwardResult(
        normalized_features=normalized_array,
        outputs=first,
        eval_mode=not model.training,
        gradients_disabled=gradients_disabled,
        state_dict_unchanged=unchanged,
        repeated_forward_exact=repeated_exact,
        state_sha256_before=state_before_hash,
        state_sha256_after=state_after_hash,
    )


def row_output_arrays(
    dataset: FeatureDataset,
    frozen: FrozenModel,
    result: ForwardResult,
) -> dict[str, np.ndarray]:
    count = dataset.features.shape[0]
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "dataset": np.asarray([dataset.name], dtype="U5"),
        "architecture": np.asarray([frozen.record.architecture], dtype="U6"),
        "seed": np.asarray([frozen.record.seed], dtype=np.int64),
        "feature_column_names": np.asarray(FEATURE_NAMES, dtype="U16"),
        "feature_column_units": np.asarray(FEATURE_UNITS, dtype="U16"),
        "row_index": dataset.row_index.copy(),
        "epoch_index": dataset.epoch_index.copy(),
        "satellite_ids": dataset.satellite_ids.copy(),
        "constellations": dataset.constellations.copy(),
        "features": dataset.features.copy(),
        "cn0_snr0_div_1000": dataset.features[:, 0].copy(),
        "elevation_rad": dataset.features[:, 1].copy(),
        "ols_residual_m": dataset.features[:, 2].copy(),
        "normalized_features": result.normalized_features.copy(),
    }
    arrays.update((name, values.copy()) for name, values in dataset.extra_identity.items())
    for semantic, values in result.outputs.items():
        arrays[f"predicted_{semantic}"] = values.copy()
    if arrays["features"].shape != (count, 3) or arrays["normalized_features"].shape != (count, 3):
        raise RuntimeError("row-level feature arrays are misaligned")
    return arrays


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a byte-stable NPZ and refuse to replace an artifact."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite scientific artifact: {path}")
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compresslevel=9)


def distribution(values: np.ndarray, semantic: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("distribution requires a finite non-empty vector")
    percentiles = np.percentile(values, [5, 25, 50, 75, 95], method=PERCENTILE_METHOD)
    result = {
        "mean": float(np.mean(values)),
        "median": float(percentiles[2]),
        "population_sd": float(np.std(values, ddof=0)),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    if semantic == "bias_m":
        absolute = np.abs(values)
        result.update(
            {
                "median_absolute": float(np.median(absolute)),
                "p95_absolute": float(np.percentile(absolute, 95, method=PERCENTILE_METHOD)),
            }
        )
    return result


def per_seed_summary_rows(
    dataset: str,
    architecture: str,
    seed: int,
    outputs: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for semantic, values in outputs.items():
        row: dict[str, object] = {
            "dataset": dataset,
            "architecture": architecture,
            "seed": seed,
            "output": semantic,
            "row_count": int(values.size),
            **distribution(values, semantic),
        }
        if architecture == "TDL-BW" and semantic == "bias_m":
            row["fraction_exactly_zero"] = float(np.mean(values == 0.0))
            row["fraction_positive"] = float(np.mean(values > 0.0))
        rows.append(row)
    return rows


def across_seed_summary_rows(per_seed_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in per_seed_rows:
        groups[(str(row["dataset"]), str(row["architecture"]), str(row["output"]))].append(row)
    identifier_fields = {"dataset", "architecture", "seed", "output", "row_count"}
    result: list[dict[str, object]] = []
    for (dataset, architecture, semantic), rows in sorted(groups.items()):
        if sorted(int(row["seed"]) for row in rows) != list(range(10)):
            raise RuntimeError(f"incomplete seed summary for {dataset}/{architecture}/{semantic}")
        metrics = sorted(set.intersection(*(set(row) for row in rows)) - identifier_fields)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            q25, median, q75 = np.percentile(values, [25, 50, 75], method=PERCENTILE_METHOD)
            result.append(
                {
                    "dataset": dataset,
                    "architecture": architecture,
                    "output": semantic,
                    "per_seed_metric": metric,
                    "seed_count": 10,
                    "mean": float(np.mean(values)),
                    "median": float(median),
                    "population_sd": float(np.std(values, ddof=0)),
                    "iqr": float(q75 - q25),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def relationship_rows(
    dataset: FeatureDataset,
    architecture: str,
    seed: int,
    outputs: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    correlation_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    for semantic, output in outputs.items():
        for feature_name, (feature_index, edges, unit) in FEATURE_RELATIONSHIPS.items():
            feature = dataset.features[:, feature_index]
            if feature_name == "abs_ols_residual":
                feature = np.abs(feature)
            correlation_rows.append(
                {
                    "dataset": dataset.name,
                    "architecture": architecture,
                    "seed": seed,
                    "output": semantic,
                    "feature": feature_name,
                    "row_count": int(feature.size),
                    "pearson": _correlation(feature, output),
                    "spearman": _correlation(_average_ranks(feature), _average_ranks(output)),
                }
            )
            indices = np.searchsorted(edges, feature, side="right") - 1
            indices[feature == edges[-1]] = len(edges) - 2
            if np.any((indices < 0) | (indices >= len(edges) - 1)):
                raise RuntimeError(f"fixed {feature_name} bins do not cover {dataset.name}")
            for index in range(len(edges) - 1):
                selected = output[indices == index]
                if selected.size:
                    p05, median, p95 = np.percentile(
                        selected, [5, 50, 95], method=PERCENTILE_METHOD
                    )
                    bin_rows.append(
                        {
                            "dataset": dataset.name,
                            "architecture": architecture,
                            "seed": seed,
                            "output": semantic,
                            "feature": feature_name,
                            "feature_unit": unit,
                            "bin_index": index,
                            "left": float(edges[index]),
                            "right": float(edges[index + 1]),
                            "right_inclusive": index == len(edges) - 2,
                            "count": int(selected.size),
                            "median": float(median),
                            "p05": float(p05),
                            "p95": float(p95),
                        }
                    )
    return correlation_rows, bin_rows


def _ks_statistic(first: np.ndarray, second: np.ndarray) -> float:
    values = np.sort(np.concatenate((first, second)))
    first_sorted = np.sort(first)
    second_sorted = np.sort(second)
    first_cdf = np.searchsorted(first_sorted, values, side="right") / first.size
    second_cdf = np.searchsorted(second_sorted, values, side="right") / second.size
    return float(np.max(np.abs(first_cdf - second_cdf)))


def comparison_rows(
    datasets: Sequence[FeatureDataset],
    output_lookup: Mapping[tuple[str, str, int, str], np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    by_name = {dataset.name: dataset for dataset in datasets}
    comparisons: list[dict[str, object]] = []
    matched_residual: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        for semantic in OUTPUT_SEMANTICS[architecture]:
            for seed in range(10):
                ibiza = output_lookup[("Ibiza", architecture, seed, semantic)]
                for reference in ("KLT1", "KLT2"):
                    klt = output_lookup[(reference, architecture, seed, semantic)]
                    comparisons.append(
                        {
                            "architecture": architecture,
                            "output": semantic,
                            "seed": seed,
                            "reference": reference,
                            "ibiza_minus_reference_mean": float(np.mean(ibiza) - np.mean(klt)),
                            "ibiza_minus_reference_median": float(np.median(ibiza) - np.median(klt)),
                            "ks_statistic": _ks_statistic(ibiza, klt),
                        }
                    )
                    edges = FEATURE_RELATIONSHIPS["abs_ols_residual"][1]
                    ibiza_abs = np.abs(by_name["Ibiza"].features[:, 2])
                    klt_abs = np.abs(by_name[reference].features[:, 2])
                    for index in range(len(edges) - 1):
                        ibiza_mask = (ibiza_abs >= edges[index]) & (
                            (ibiza_abs <= edges[index + 1])
                            if index == len(edges) - 2
                            else (ibiza_abs < edges[index + 1])
                        )
                        klt_mask = (klt_abs >= edges[index]) & (
                            (klt_abs <= edges[index + 1])
                            if index == len(edges) - 2
                            else (klt_abs < edges[index + 1])
                        )
                        if np.count_nonzero(ibiza_mask) < 20 or np.count_nonzero(klt_mask) < 20:
                            continue
                        matched_residual.append(
                            {
                                "architecture": architecture,
                                "output": semantic,
                                "seed": seed,
                                "reference": reference,
                                "abs_residual_left_m": float(edges[index]),
                                "abs_residual_right_m": float(edges[index + 1]),
                                "ibiza_count": int(np.count_nonzero(ibiza_mask)),
                                "reference_count": int(np.count_nonzero(klt_mask)),
                                "ibiza_minus_reference_median": float(
                                    np.median(ibiza[ibiza_mask]) - np.median(klt[klt_mask])
                                ),
                                "ks_statistic": _ks_statistic(ibiza[ibiza_mask], klt[klt_mask]),
                            }
                        )

    ks_by_architecture: dict[str, list[float]] = defaultdict(list)
    for row in comparisons:
        ks_by_architecture[str(row["architecture"])].append(float(row["ks_statistic"]))
    architecture_shift = {
        architecture: {
            "mean_ks_over_outputs_seeds_and_klt_references": float(np.mean(values)),
            "population_sd": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for architecture, values in ks_by_architecture.items()
    }
    winner = max(architecture_shift, key=lambda item: architecture_shift[item]["mean_ks_over_outputs_seeds_and_klt_references"])

    high_cn0: list[dict[str, object]] = []
    nominal: list[dict[str, object]] = []
    pooled_features = np.concatenate([dataset.features for dataset in datasets])
    thresholds = {
        "high_cn0": float(np.percentile(pooled_features[:, 0], 75, method=PERCENTILE_METHOD)),
        "high_elevation_rad": float(np.percentile(pooled_features[:, 1], 75, method=PERCENTILE_METHOD)),
        "low_abs_residual_m": float(np.percentile(np.abs(pooled_features[:, 2]), 25, method=PERCENTILE_METHOD)),
    }
    for dataset in datasets:
        high_mask = dataset.features[:, 0] > HIGH_CN0_KLT3_MAX
        nominal_mask = (
            (dataset.features[:, 0] >= thresholds["high_cn0"])
            & (dataset.features[:, 1] >= thresholds["high_elevation_rad"])
            & (np.abs(dataset.features[:, 2]) <= thresholds["low_abs_residual_m"])
        )
        for architecture in ARCHITECTURES:
            for semantic in OUTPUT_SEMANTICS[architecture]:
                for seed in range(10):
                    values = output_lookup[(dataset.name, architecture, seed, semantic)]
                    if np.any(high_mask):
                        high_cn0.append(
                            {
                                "dataset": dataset.name,
                                "architecture": architecture,
                                "output": semantic,
                                "seed": seed,
                                "criterion": f"C/N0 > KLT3 training maximum ({HIGH_CN0_KLT3_MAX:g})",
                                "count": int(np.count_nonzero(high_mask)),
                                **distribution(values[high_mask], semantic),
                            }
                        )
                    if np.any(nominal_mask):
                        nominal.append(
                            {
                                "dataset": dataset.name,
                                "architecture": architecture,
                                "output": semantic,
                                "seed": seed,
                                "count": int(np.count_nonzero(nominal_mask)),
                                **distribution(values[nominal_mask], semantic),
                            }
                        )
    interpretation = {
        "comparison_metric": (
            "Two-sample empirical KS distance, computed per seed and output. Architecture "
            "scores average those distances over outputs and the two held-out references; "
            "no seeds or observations are treated as independent replicates."
        ),
        "architecture_shift_scores": architecture_shift,
        "largest_shift_architecture": winner,
        "substantial_is_not_a_prespecified_inferential_threshold": True,
        "high_cn0_boundary_snr0_div_1000": HIGH_CN0_KLT3_MAX,
        "nominal_stratum_thresholds": thresholds,
        "nominal_stratum_definition": (
            "same pooled-feature thresholds for every dataset: upper-quartile C/N0, "
            "upper-quartile elevation, and lower-quartile absolute OLS residual"
        ),
        "causal_positioning_claim_made": False,
    }
    return comparisons + high_cn0 + nominal, matched_residual, interpretation


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure: object, path: Path) -> None:
    figure.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "validation.domain_shift.model_response_probe"},
    )


def make_plots(
    datasets: Sequence[FeatureDataset],
    output_lookup: Mapping[tuple[str, str, int, str], np.ndarray],
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir()
    created: list[Path] = []
    colors = {"KLT1": "#1f77b4", "KLT2": "#ff7f0e", "Ibiza": "#2ca02c"}
    for architecture in ARCHITECTURES:
        for semantic in OUTPUT_SEMANTICS[architecture]:
            figure, axis = plt.subplots(figsize=(7.4, 4.7))
            for dataset in datasets:
                seed_outputs = [
                    output_lookup[(dataset.name, architecture, seed, semantic)]
                    for seed in range(10)
                ]
                for seed, values in enumerate(seed_outputs):
                    ordered = np.sort(values)
                    y = np.arange(1, ordered.size + 1, dtype=np.float64) / ordered.size
                    axis.step(
                        ordered,
                        y,
                        where="post",
                        color=colors[dataset.name],
                        alpha=0.18,
                        linewidth=0.65,
                        label=dataset.name if seed == 0 else None,
                    )
            axis.set(
                xlabel="predicted bias [m]" if semantic == "bias_m" else "predicted weight",
                ylabel="ECDF",
                ylim=(0.0, 1.01),
                title=f"{architecture} {semantic}: per-seed output distributions",
            )
            axis.grid(alpha=0.2)
            axis.legend()
            path = plots / f"{ARCHITECTURE_SLUGS[architecture]}_{semantic}_ecdf.png"
            _save_figure(figure, path)
            plt.close(figure)
            created.append(path)

            for feature_name, (feature_index, _edges, unit) in FEATURE_RELATIONSHIPS.items():
                figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), sharey=True)
                for axis, dataset in zip(axes, datasets, strict=True):
                    feature = dataset.features[:, feature_index]
                    if feature_name == "abs_ols_residual":
                        feature = np.abs(feature)
                    ensemble_median = np.median(
                        np.stack(
                            [
                                output_lookup[(dataset.name, architecture, seed, semantic)]
                                for seed in range(10)
                            ]
                        ),
                        axis=0,
                    )
                    image = axis.hexbin(
                        feature,
                        ensemble_median,
                        gridsize=55,
                        bins="log",
                        mincnt=1,
                        cmap="viridis",
                    )
                    axis.set(xlabel=f"{feature_name} [{unit}]", title=dataset.name)
                    axis.grid(alpha=0.15)
                    figure.colorbar(image, ax=axis, label="row count (log colour scale)")
                axes[0].set_ylabel(
                    "across-seed median predicted bias [m]"
                    if semantic == "bias_m"
                    else "across-seed median predicted weight"
                )
                figure.suptitle(f"{architecture} {semantic} versus {feature_name}")
                path = plots / (
                    f"{ARCHITECTURE_SLUGS[architecture]}_{semantic}_vs_{feature_name}.png"
                )
                _save_figure(figure, path)
                plt.close(figure)
                created.append(path)
    return created


def _schema(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {"dtype": str(values.dtype), "shape": list(values.shape)}
        for name, values in arrays.items()
    }


def _artifact_records(paths: Iterable[Path], root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def run_probe(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    klt1_path: Path = DEFAULT_KLT1,
    klt2_path: Path = DEFAULT_KLT2,
    ibiza_path: Path = DEFAULT_IBIZA,
    checkpoint_manifest: Path = DEFAULT_CHECKPOINT_MANIFEST,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    datasets = load_feature_datasets(klt1_path, klt2_path, ibiza_path)
    input_hashes_before = {dataset.name: sha256_file(dataset.source_path) for dataset in datasets}
    inventory = validate_checkpoint_inventory(manifest_path=checkpoint_manifest.resolve())
    if not np.array_equal(np.asarray(inventory.common_mean), FROZEN_MEAN):
        raise RuntimeError("checkpoint normalization mean changed")
    if not np.array_equal(np.asarray(inventory.common_std), FROZEN_STD):
        raise RuntimeError("checkpoint normalization std changed")
    checkpoint_hashes_before = {
        record.checkpoint_path: sha256_file(REPOSITORY_ROOT / record.checkpoint_path)
        for record in inventory.records
    }

    with tempfile.TemporaryDirectory(prefix="model-response-", dir=output_dir.parent) as temporary:
        work = Path(temporary) / "publish"
        rows_dir = work / "rows"
        work.mkdir()
        rows_dir.mkdir()
        per_seed_rows: list[dict[str, object]] = []
        correlation_rows: list[dict[str, object]] = []
        bin_rows: list[dict[str, object]] = []
        forward_checks: list[dict[str, object]] = []
        output_lookup: dict[tuple[str, str, int, str], np.ndarray] = {}
        row_artifacts: list[Path] = []
        schemas: dict[str, object] = {}

        for architecture in ARCHITECTURES:
            for seed in range(10):
                frozen = load_frozen_model(
                    architecture,
                    seed,
                    manifest_path=checkpoint_manifest.resolve(),
                )
                loaded_mean = frozen.model.state_dict()["seq.0.mean"].detach().cpu().numpy()
                loaded_std = frozen.model.state_dict()["seq.0.std"].detach().cpu().numpy()
                if not np.array_equal(loaded_mean, FROZEN_MEAN) or not np.array_equal(loaded_std, FROZEN_STD):
                    raise RuntimeError(f"{architecture} seed {seed} normalization differs")
                for dataset in datasets:
                    result = forward_only(frozen, dataset.features)
                    arrays = row_output_arrays(dataset, frozen, result)
                    filename = (
                        f"{dataset.name.lower()}_{ARCHITECTURE_SLUGS[architecture]}_seed_{seed:02d}.npz"
                    )
                    path = rows_dir / filename
                    write_deterministic_npz(path, arrays)
                    row_artifacts.append(path)
                    schemas[f"{dataset.name}/{architecture}"] = _schema(arrays)
                    per_seed_rows.extend(
                        per_seed_summary_rows(dataset.name, architecture, seed, result.outputs)
                    )
                    correlations, bins = relationship_rows(
                        dataset, architecture, seed, result.outputs
                    )
                    correlation_rows.extend(correlations)
                    bin_rows.extend(bins)
                    for semantic, values in result.outputs.items():
                        output_lookup[(dataset.name, architecture, seed, semantic)] = values
                    forward_checks.append(
                        {
                            "dataset": dataset.name,
                            "architecture": architecture,
                            "seed": seed,
                            "row_count": dataset.features.shape[0],
                            "checkpoint_sha256": frozen.record.sha256,
                            "model_eval_active": result.eval_mode,
                            "gradients_disabled_inside_forward": result.gradients_disabled,
                            "state_dict_unchanged": result.state_dict_unchanged,
                            "state_sha256_before": result.state_sha256_before,
                            "state_sha256_after": result.state_sha256_after,
                            "repeated_forward_exact": result.repeated_forward_exact,
                        }
                    )

        across_seed_rows = across_seed_summary_rows(per_seed_rows)
        comparison_data, matched_residual_rows, interpretation = comparison_rows(
            datasets, output_lookup
        )
        summaries = {
            "per_seed_summary.csv": per_seed_rows,
            "across_seed_summary.csv": across_seed_rows,
            "feature_output_correlations.csv": correlation_rows,
            "feature_output_fixed_bins.csv": bin_rows,
            "dataset_comparisons_and_strata.csv": comparison_data,
            "matched_abs_residual_comparisons.csv": matched_residual_rows,
        }
        for filename, rows in summaries.items():
            _write_csv(work / filename, rows)
        _json_dump(work / "per_seed_summary.json", per_seed_rows)
        _json_dump(work / "across_seed_summary.json", across_seed_rows)
        _json_dump(work / "scientific_interpretation.json", interpretation)
        plot_paths = make_plots(datasets, output_lookup, work)

        checkpoint_hashes_after = {
            record.checkpoint_path: sha256_file(REPOSITORY_ROOT / record.checkpoint_path)
            for record in inventory.records
        }
        input_hashes_after = {dataset.name: sha256_file(dataset.source_path) for dataset in datasets}
        if input_hashes_before != input_hashes_after:
            raise RuntimeError("a source feature artifact changed during the probe")
        if checkpoint_hashes_before != checkpoint_hashes_after:
            raise RuntimeError("a checkpoint changed during the probe")
        if not all(
            row["model_eval_active"]
            and row["gradients_disabled_inside_forward"]
            and row["state_dict_unchanged"]
            and row["repeated_forward_exact"]
            for row in forward_checks
        ):
            raise RuntimeError("one or more forward-only controls failed")

        generated_without_manifest = row_artifacts + plot_paths + [
            work / filename for filename in summaries
        ] + [
            work / "per_seed_summary.json",
            work / "across_seed_summary.json",
            work / "scientific_interpretation.json",
        ]
        artifact_records = _artifact_records(generated_without_manifest, work)
        _write_csv(work / "artifact_hashes.csv", artifact_records)
        manifest = {
            "schema_version": 1,
            "status": "complete_forward_only_model_response_probe",
            "scientific_scope": "descriptive frozen-network responses; no positioning or causal claim",
            "repository": {
                "commit": _git_value("rev-parse", "HEAD"),
                "dirty_at_generation": bool(_git_value("status", "--porcelain")),
            },
            "inputs": {
                dataset.name: {
                    "path": str(dataset.source_path),
                    "sha256_before": input_hashes_before[dataset.name],
                    "sha256_after": input_hashes_after[dataset.name],
                    "row_count": int(dataset.features.shape[0]),
                    "epoch_count": int(np.unique(dataset.epoch_index).size),
                    "constellation_counts": {
                        str(label): int(count)
                        for label, count in zip(
                            *np.unique(dataset.constellations, return_counts=True), strict=True
                        )
                    },
                }
                for dataset in datasets
            },
            "feature_order": [
                {"index": index, "name": name, "unit": unit}
                for index, (name, unit) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS, strict=True))
            ],
            "normalization": {
                "source": "each checkpoint's embedded seq.0 StandardizeLayer buffers",
                "mean": FROZEN_MEAN.tolist(),
                "std": FROZEN_STD.tolist(),
                "fitted_or_recomputed": False,
                "all_30_bit_identical": True,
            },
            "checkpoint_manifest": {
                "path": str(checkpoint_manifest.resolve()),
                "sha256": sha256_file(checkpoint_manifest.resolve()),
                "records": [
                    {
                        "architecture": record.architecture,
                        "seed": record.seed,
                        "path": record.checkpoint_path,
                        "expected_sha256": record.sha256,
                        "actual_sha256_before": checkpoint_hashes_before[record.checkpoint_path],
                        "actual_sha256_after": checkpoint_hashes_after[record.checkpoint_path],
                    }
                    for record in inventory.records
                ],
            },
            "output_semantics": {
                "TDL-B": "raw signed predicted bias in metres; no final activation",
                "TDL-W": "released sigmoid output multiplied by 10 and clamped to [0, 10]",
                "TDL-BW": (
                    "released tuple order (weight, bias): sigmoid/clamp weight in [0,1], "
                    "ReLU non-negative bias in metres"
                ),
            },
            "percentiles": {"implementation": "numpy.percentile", "method": PERCENTILE_METHOD},
            "fixed_bins": {
                name: {"edges": edges.tolist(), "unit": unit}
                for name, (_index, edges, unit) in FEATURE_RELATIONSHIPS.items()
            },
            "controls": {
                "execution_boundary": "raw feature -> frozen normalization -> frozen network -> neural output -> STOP",
                "feature_hashes_unchanged": True,
                "all_30_checkpoint_hashes_match_and_unchanged": True,
                "model_eval_active_for_all_90_forwards": True,
                "gradients_disabled_for_all_90_forwards": True,
                "state_dict_unchanged_for_all_90_forwards": True,
                "repeated_full_forward_exact_for_all_90_forwards": True,
                "row_counts_match_inputs": True,
                "ground_truth_read": False,
                "positioning_or_wls_called": False,
                "optimizer_created": False,
                "backward_called": False,
                "training_performed": False,
                "normalization_fitted": False,
            },
            "forward_checks": forward_checks,
            "row_artifact_count": len(row_artifacts),
            "row_schemas": schemas,
            "summary_row_counts": {
                filename: len(rows) for filename, rows in summaries.items()
            },
            "scientific_interpretation": interpretation,
            "generated_files_excluding_manifest_and_hash_index": artifact_records,
            "artifact_hash_index": {
                "path": "artifact_hashes.csv",
                "sha256": sha256_file(work / "artifact_hashes.csv"),
            },
        }
        _json_dump(work / "model_response_manifest.json", manifest)
        os.rename(work, output_dir)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--klt1", type=Path, default=DEFAULT_KLT1)
    parser.add_argument("--klt2", type=Path, default=DEFAULT_KLT2)
    parser.add_argument("--ibiza", type=Path, default=DEFAULT_IBIZA)
    parser.add_argument("--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_probe(
        output_dir=args.output,
        klt1_path=args.klt1,
        klt2_path=args.klt2,
        ibiza_path=args.ibiza,
        checkpoint_manifest=args.checkpoint_manifest,
    )
    print(f"Output: {args.output.resolve()}")
    print(f"Row artifacts: {manifest['row_artifact_count']}")
    print(f"Largest descriptive KS shift: {manifest['scientific_interpretation']['largest_shift_architecture']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
