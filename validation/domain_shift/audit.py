#!/usr/bin/env python3
"""Audit frozen paper-era KLT3 and Ibiza feature domains without inference.

The percentile convention is NumPy's deterministic ``method="linear"``.
This module never imports model implementations, deserializes checkpoints, or
uses ground truth.  Checkpoints and frozen inference files are read only as
byte streams for SHA-256 validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KLT3 = REPOSITORY_ROOT / "validation/paper_weightnet/klt3_features.npz"
DEFAULT_KLT3_MANIFEST = (
    REPOSITORY_ROOT / "validation/paper_weightnet/klt3_feature_manifest.json"
)
DEFAULT_IBIZA = (
    REPOSITORY_ROOT.parent
    / "external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz"
)
DEFAULT_IBIZA_MANIFEST = (
    REPOSITORY_ROOT / "validation/ibiza_generalization/ibiza_preprocessed_manifest.json"
)
DEFAULT_CHECKPOINT_MANIFEST = (
    REPOSITORY_ROOT / "validation/ibiza_generalization/frozen_checkpoint_manifest.json"
)
DEFAULT_IBIZA_RESULTS = (
    REPOSITORY_ROOT.parent
    / "external_data/ibiza_2025_01_01/results/frozen_tdl"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT.parent / "external_data/domain_shift/klt3_vs_ibiza"
KLT_HELD_OUT_RESULT_DIRECTORIES = {
    "TDL-B": REPOSITORY_ROOT / "results/paper_biasnet_seed_sensitivity",
    "TDL-W": REPOSITORY_ROOT / "results/paper_weightnet_seed_sensitivity",
    "TDL-BW": REPOSITORY_ROOT / "results/paper_hybrid_seed_sensitivity",
}

FEATURE_NAMES = ("C/N0", "elevation", "OLS residual")
FEATURE_SLUGS = ("cn0", "elevation", "ols_residual")
FEATURE_UNITS = ("SNR[0]/1000", "radian", "metre")
KLT3_SHA256 = "ae1457a933992f57e850a0abda64c1150f4586640d311b1880f00ed1262b5815"
IBIZA_SHA256 = "edb0189e9eadf3266d75984e3041a90306dd44b3ebecc0101eb188f933dc88c5"
KLT3_MEAN = np.asarray(
    [29.084903717041016, 0.8471900224685669, -1.7873233559839719e-07],
    dtype=np.float64,
)
KLT3_STD = np.asarray(
    [5.891841888427734, 0.2879891097545624, 4.3018035888671875],
    dtype=np.float64,
)
PERCENTILE_METHOD = "linear"
ELEVATION_BIN_EDGES_RAD = np.deg2rad(np.arange(0.0, 105.0, 15.0))
CN0_BIN_EDGES = np.arange(0.0, 65.0, 5.0)


@dataclass(frozen=True)
class Dataset:
    name: str
    features: np.ndarray
    constellations: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": actual}


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def load_datasets(
    klt3_path: Path,
    ibiza_path: Path,
    ibiza_manifest_path: Path,
) -> tuple[Dataset, Dataset]:
    verify_hash(klt3_path, KLT3_SHA256, "KLT3 feature artifact")
    verify_hash(ibiza_path, IBIZA_SHA256, "Ibiza feature artifact")
    with np.load(klt3_path, allow_pickle=False) as source:
        klt_features = np.asarray(source["features"], dtype=np.float64)
        satellite_ids = np.asarray(source["satellite_ids"])
    if klt_features.shape != (8857, 3) or satellite_ids.shape != (8857,):
        raise RuntimeError("unexpected KLT3 feature or satellite-label shape")
    klt_constellations = np.asarray([item[0] for item in satellite_ids], dtype="U1")

    ibiza_manifest = _load_json(ibiza_manifest_path)
    raw_mapping = ibiza_manifest["state_layout"]["constellation_code"]  # type: ignore[index]
    code_mapping = {int(key): str(value) for key, value in raw_mapping.items()}  # type: ignore[union-attr]
    with np.load(ibiza_path, allow_pickle=False) as source:
        ibiza_features = np.asarray(source["features"], dtype=np.float64)
        codes = np.asarray(source["constellation_code"], dtype=np.int64)
        columns = np.column_stack(
            (source["cn0_snr0_div_1000"], source["elevation_rad"], source["ols_residual_m"])
        )
    if ibiza_features.shape != (73204, 3) or codes.shape != (73204,):
        raise RuntimeError("unexpected Ibiza feature or constellation-label shape")
    if not np.array_equal(ibiza_features, columns):
        raise RuntimeError("Ibiza feature matrix does not equal its three named raw columns")
    try:
        ibiza_constellations = np.asarray([code_mapping[int(code)] for code in codes], dtype="U1")
    except KeyError as error:
        raise RuntimeError(f"Ibiza contains an unmapped constellation code: {error}") from error
    for label, values in (("KLT3", klt_features), ("Ibiza", ibiza_features)):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{label} features contain non-finite values")
    return (
        Dataset("KLT3", klt_features, klt_constellations),
        Dataset("Ibiza", ibiza_features, ibiza_constellations),
    )


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("distribution requires at least one value")
    p05, p25, median, p75, p95 = np.percentile(
        values, [5, 25, 50, 75, 95], method=PERCENTILE_METHOD
    )
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(median),
        "population_std": float(np.std(values, ddof=0)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "p05": float(p05),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
        "iqr": float(p75 - p25),
    }


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64).reshape(-1), kind="mergesort")
    if x.size == 0:
        raise ValueError("ECDF requires at least one value")
    y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
    return x, y


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("correlation inputs must have the same length >= 2")
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(_rankdata_average(x), _rankdata_average(y))[0, 1])
    return pearson, spearman


def feature_summary(datasets: Sequence[Dataset]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for index, (feature, unit) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS, strict=True)):
            rows.append({"dataset": dataset.name, "feature": feature, "unit": unit, **distribution(dataset.features[:, index])})
    return rows


def zscore_summary(
    datasets: Sequence[Dataset], klt_min: np.ndarray, klt_max: np.ndarray
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        z = (dataset.features - KLT3_MEAN) / KLT3_STD
        for index, (feature, unit) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS, strict=True)):
            values = z[:, index]
            p05, median, p95 = np.percentile(
                values, [5, 50, 95], method=PERCENTILE_METHOD
            )
            raw = dataset.features[:, index]
            below = raw < klt_min[index]
            above = raw > klt_max[index]
            rows.append(
                {
                    "dataset": dataset.name,
                    "feature": feature,
                    "raw_unit": unit,
                    "count": int(values.size),
                    "z_mean": float(np.mean(values)),
                    "z_median": float(median),
                    "z_population_std": float(np.std(values, ddof=0)),
                    "z_p05": float(p05),
                    "z_p95": float(p95),
                    "z_minimum": float(np.min(values)),
                    "z_maximum": float(np.max(values)),
                    "fraction_abs_z_gt_1": float(np.mean(np.abs(values) > 1.0)),
                    "fraction_abs_z_gt_2": float(np.mean(np.abs(values) > 2.0)),
                    "fraction_abs_z_gt_3": float(np.mean(np.abs(values) > 3.0)),
                    "fraction_below_klt3_raw_min": float(np.mean(below)),
                    "fraction_above_klt3_raw_max": float(np.mean(above)),
                    "fraction_outside_klt3_raw_range": float(np.mean(below | above)),
                }
            )
    return rows


def constellation_summary(datasets: Sequence[Dataset]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for constellation in sorted(np.unique(dataset.constellations).tolist()):
            selected = dataset.features[dataset.constellations == constellation]
            for index, (feature, unit) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS, strict=True)):
                rows.append(
                    {
                        "dataset": dataset.name,
                        "constellation": constellation,
                        "feature": feature,
                        "unit": unit,
                        **distribution(selected[:, index]),
                    }
                )
    return rows


def binned_relationships(datasets: Sequence[Dataset]) -> list[dict[str, object]]:
    definitions = (
        ("abs_residual_vs_elevation", 1, "elevation", "radian", ELEVATION_BIN_EDGES_RAD),
        ("abs_residual_vs_cn0", 0, "C/N0", "SNR[0]/1000", CN0_BIN_EDGES),
    )
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        absolute_residual = np.abs(dataset.features[:, 2])
        for relationship, feature_index, binned_feature, unit, edges in definitions:
            values = dataset.features[:, feature_index]
            for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                if bin_index == len(edges) - 2:
                    selected = (values >= left) & (values <= right)
                else:
                    selected = (values >= left) & (values < right)
                residual = absolute_residual[selected]
                rows.append(
                    {
                        "dataset": dataset.name,
                        "relationship": relationship,
                        "binned_feature": binned_feature,
                        "bin_unit": unit,
                        "bin_index": bin_index,
                        "bin_left_inclusive": float(left),
                        "bin_right": float(right),
                        "right_edge_inclusive": bin_index == len(edges) - 2,
                        "count": int(residual.size),
                        "median_abs_residual_m": "" if residual.size == 0 else float(np.percentile(residual, 50, method=PERCENTILE_METHOD)),
                        "p95_abs_residual_m": "" if residual.size == 0 else float(np.percentile(residual, 95, method=PERCENTILE_METHOD)),
                    }
                )
    return rows


def correlation_summary(datasets: Sequence[Dataset]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        absolute_residual = np.abs(dataset.features[:, 2])
        for index, feature in ((1, "elevation"), (0, "C/N0")):
            pearson, spearman = correlation(dataset.features[:, index], absolute_residual)
            rows.append(
                {
                    "dataset": dataset.name,
                    "x_feature": feature,
                    "y_feature": "abs(OLS residual)",
                    "count": int(absolute_residual.size),
                    "pearson": pearson,
                    "spearman": spearman,
                    "interpretation": "descriptive_only",
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": "domain_shift.audit"})  # type: ignore[attr-defined]


def make_plots(datasets: Sequence[Dataset], output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created: list[Path] = []
    colors = {"KLT3": "#1f77b4", "Ibiza": "#d62728"}
    for index, (feature, slug, unit) in enumerate(
        zip(FEATURE_NAMES, FEATURE_SLUGS, FEATURE_UNITS, strict=True)
    ):
        combined = np.concatenate([dataset.features[:, index] for dataset in datasets])
        edges = np.histogram_bin_edges(combined, bins=80)
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        for dataset in datasets:
            axis.hist(dataset.features[:, index], bins=edges, density=True, histtype="step", linewidth=1.6, label=dataset.name, color=colors[dataset.name])
        axis.set(xlabel=f"{feature} [{unit}]", ylabel="density", title=f"{feature}: raw feature histogram")
        axis.legend()
        axis.grid(alpha=0.2)
        path = output / "plots" / f"feature_{slug}_histogram.png"
        _save_figure(figure, path)
        plt.close(figure)
        created.append(path)

        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        for dataset in datasets:
            x, y = ecdf(dataset.features[:, index])
            axis.step(x, y, where="post", linewidth=1.5, label=dataset.name, color=colors[dataset.name])
        axis.set(xlabel=f"{feature} [{unit}]", ylabel="ECDF", ylim=(0.0, 1.01), title=f"{feature}: raw feature ECDF")
        axis.legend()
        axis.grid(alpha=0.2)
        path = output / "plots" / f"feature_{slug}_ecdf.png"
        _save_figure(figure, path)
        plt.close(figure)
        created.append(path)

    common = sorted(set(datasets[0].constellations) & set(datasets[1].constellations))
    line_styles = {"KLT3": "-", "Ibiza": "--"}
    for index, (feature, slug, unit) in enumerate(
        zip(FEATURE_NAMES, FEATURE_SLUGS, FEATURE_UNITS, strict=True)
    ):
        figure, axis = plt.subplots(figsize=(7.8, 4.8))
        for dataset in datasets:
            for constellation in common:
                x, y = ecdf(dataset.features[dataset.constellations == constellation, index])
                axis.step(x, y, where="post", linewidth=1.2, linestyle=line_styles[dataset.name], label=f"{dataset.name} {constellation}")
        axis.set(xlabel=f"{feature} [{unit}]", ylabel="ECDF", ylim=(0.0, 1.01), title=f"{feature}: within-constellation ECDF")
        axis.legend(ncol=2, fontsize=8)
        axis.grid(alpha=0.2)
        path = output / "plots" / f"constellation_{slug}_ecdf.png"
        _save_figure(figure, path)
        plt.close(figure)
        created.append(path)

    relationships = ((1, "elevation", "radian"), (0, "cn0", "SNR[0]/1000"))
    for feature_index, slug, unit in relationships:
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
        maximum_residual = max(float(np.max(np.abs(item.features[:, 2]))) for item in datasets)
        residual_ticks_m = np.asarray(
            [value for value in (0.0, 1.0, 10.0, 100.0, 1000.0) if value <= maximum_residual],
            dtype=np.float64,
        )
        for axis, dataset in zip(axes, datasets, strict=True):
            log_residual = np.log10(1.0 + np.abs(dataset.features[:, 2]))
            image = axis.hexbin(dataset.features[:, feature_index], log_residual, gridsize=60, bins="log", mincnt=1, cmap="viridis")
            axis.set_yticks(
                np.log10(1.0 + residual_ticks_m),
                [f"{value:g}" for value in residual_ticks_m],
            )
            axis.set(xlabel=f"{FEATURE_NAMES[feature_index]} [{unit}]", title=dataset.name)
            axis.grid(alpha=0.15)
            figure.colorbar(image, ax=axis, label="bin count (log colour scale)")
        axes[0].set_ylabel("abs(OLS residual) [metre]")
        figure.suptitle(f"abs(OLS residual) versus {FEATURE_NAMES[feature_index]}")
        path = output / "plots" / f"relationship_abs_residual_vs_{slug}.png"
        _save_figure(figure, path)
        plt.close(figure)
        created.append(path)
    return created


def frozen_snapshot(
    checkpoint_manifest_path: Path, ibiza_results_dir: Path
) -> dict[str, object]:
    checkpoint_manifest = _load_json(checkpoint_manifest_path)
    records = checkpoint_manifest.get("checkpoints")
    if not isinstance(records, list) or len(records) != 30:
        raise RuntimeError("frozen checkpoint manifest must contain exactly 30 records")
    checkpoints: list[dict[str, object]] = []
    for record in records:
        path = REPOSITORY_ROOT / str(record["checkpoint_path"])
        checkpoints.append(
            {
                "architecture": record["architecture"],
                "seed": record["seed"],
                **verify_hash(path, str(record["sha256"]), "frozen checkpoint"),
            }
        )
    run_manifest_path = ibiza_results_dir / "run_manifest.json"
    run_manifest = _load_json(run_manifest_path)
    jobs = run_manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 30:
        raise RuntimeError("Ibiza run manifest must contain exactly 30 jobs")
    results: list[dict[str, object]] = []
    for job in jobs:
        path = ibiza_results_dir / str(job["result_file"])
        results.append(
            {
                "architecture": job["architecture"],
                "seed": job["seed"],
                **verify_hash(path, str(job["result_file_sha256"]), "frozen Ibiza inference JSONL"),
            }
        )
    held_out_results: list[dict[str, object]] = []
    for architecture, directory in KLT_HELD_OUT_RESULT_DIRECTORIES.items():
        for seed in range(10):
            path = directory / f"seed_{seed}.json"
            document = _load_json(path)
            held_out = document.get("held_out")
            if not isinstance(held_out, dict) or set(held_out) != {"KLT1", "KLT2"}:
                raise RuntimeError(f"unexpected held-out structure: {path}")
            measurement_counts = {
                dataset: int(values["retained_measurement_count"])
                for dataset, values in held_out.items()
            }
            if measurement_counts != {"KLT1": 4676, "KLT2": 4914}:
                raise RuntimeError(f"unexpected held-out measurement counts: {path}")
            held_out_results.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "retained_measurement_counts": measurement_counts,
                    "stored_fields": {
                        dataset: sorted(values) for dataset, values in held_out.items()
                    },
                }
            )
    return {
        "checkpoint_manifest": verify_hash(
            checkpoint_manifest_path, sha256_file(checkpoint_manifest_path), "checkpoint manifest"
        ),
        "checkpoints": checkpoints,
        "ibiza_run_manifest": verify_hash(
            run_manifest_path, sha256_file(run_manifest_path), "Ibiza run manifest"
        ),
        "ibiza_inference_jsonl": results,
        "klt_held_out_seed_results": held_out_results,
    }


def _source_provenance(
    klt_manifest: dict[str, object],
    ibiza_manifest: dict[str, object],
    klt_manifest_path: Path,
    ibiza_manifest_path: Path,
) -> dict[str, object]:
    return {
        "klt3": {
            "tracked_manifest": {
                "path": str(klt_manifest_path.resolve()),
                "size_bytes": klt_manifest_path.stat().st_size,
                "sha256": sha256_file(klt_manifest_path),
            },
            "feature_artifact": klt_manifest["cache"],
            "reference_commit": klt_manifest["reference_commit"],
            "pyrtklib_version": klt_manifest["pyrtklib_version"],
            "pyrtklib_hypothesis_commit": klt_manifest["pyrtklib_hypothesis_commit"],
            "dataset_archive": klt_manifest["dataset_archive"],
            "rover_observation": klt_manifest["rover_observation"],
            "navigation_files": klt_manifest["navigation_files"],
            "ground_truth_provenance_only": klt_manifest["ground_truth"],
            "ground_truth_note": "Ground truth is provenance only and was not loaded by this audit.",
            "row_count": 8857,
        },
        "ibiza": {
            "tracked_manifest": {
                "path": str(ibiza_manifest_path.resolve()),
                "size_bytes": ibiza_manifest_path.stat().st_size,
                "sha256": sha256_file(ibiza_manifest_path),
            },
            "feature_artifact": ibiza_manifest["output"],
            "inputs": ibiza_manifest["inputs"],
            "tdl_gnss_reference_commit": ibiza_manifest["provenance"]["tdl_gnss_reference_commit"],  # type: ignore[index]
            "row_count": 73204,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--klt3", type=Path, default=DEFAULT_KLT3)
    parser.add_argument("--klt3-manifest", type=Path, default=DEFAULT_KLT3_MANIFEST)
    parser.add_argument("--ibiza", type=Path, default=DEFAULT_IBIZA)
    parser.add_argument("--ibiza-manifest", type=Path, default=DEFAULT_IBIZA_MANIFEST)
    parser.add_argument("--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST)
    parser.add_argument("--ibiza-results", type=Path, default=DEFAULT_IBIZA_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    datasets = load_datasets(args.klt3.resolve(), args.ibiza.resolve(), args.ibiza_manifest.resolve())
    klt3, ibiza = datasets
    frozen_before = frozen_snapshot(args.checkpoint_manifest.resolve(), args.ibiza_results.resolve())
    source_hashes_before = {
        "klt3": sha256_file(args.klt3),
        "ibiza": sha256_file(args.ibiza),
    }

    feature_rows = feature_summary(datasets)
    z_rows = zscore_summary(datasets, klt3.features.min(axis=0), klt3.features.max(axis=0))
    constellation_rows = constellation_summary(datasets)
    relationship_rows = binned_relationships(datasets)
    correlation_rows = correlation_summary(datasets)
    csv_outputs = {
        "feature_summary.csv": feature_rows,
        "zscore_summary.csv": z_rows,
        "out_of_range_summary.csv": [
            {key: row[key] for key in ("dataset", "feature", "count", "fraction_below_klt3_raw_min", "fraction_above_klt3_raw_max", "fraction_outside_klt3_raw_range")}
            for row in z_rows
        ],
        "constellation_summary.csv": constellation_rows,
        "feature_relationship_summary.csv": relationship_rows,
        "feature_correlation_summary.csv": correlation_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in csv_outputs.items():
        _write_csv(output / name, rows)
    with (output / "feature_summary.csv").open(encoding="utf-8", newline="") as stream:
        written_feature_rows = list(csv.DictReader(stream))
    expected_counts = {"KLT3": 8857, "Ibiza": 73204}
    if any(
        int(row["count"]) != expected_counts[row["dataset"]]
        for row in written_feature_rows
    ):
        raise RuntimeError("feature_summary.csv counts do not match source rows")
    with (output / "constellation_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        written_constellation_rows = list(csv.DictReader(stream))
    for dataset_name, expected in expected_counts.items():
        for feature in FEATURE_NAMES:
            total = sum(
                int(row["count"])
                for row in written_constellation_rows
                if row["dataset"] == dataset_name and row["feature"] == feature
            )
            if total != expected:
                raise RuntimeError(
                    "constellation_summary.csv partition count mismatch for "
                    f"{dataset_name} {feature}: {total} != {expected}"
                )
    plot_paths = make_plots(datasets, output)

    blocker = {
        "status": "blocked_before_new_inference",
        "reason": "Required per-satellite model outputs are not stored.",
        "ibiza_missing": [
            "individual predicted bias for every retained satellite row and seed for TDL-B and TDL-BW",
            "individual predicted weight for every retained satellite row and seed for TDL-W and TDL-BW",
        ],
        "ibiza_available_but_insufficient": "Each frozen JSONL row stores only per-epoch count/mean/std/min/max and selected fractions.",
        "klt_held_out_missing": [
            "complete KLT1 and KLT2 raw feature rows and constellation labels in evaluation row order",
            "per-satellite TDL-B biases for seeds 0-9",
            "per-satellite TDL-W weights for seeds 0-9",
            "per-satellite TDL-BW weights and biases for seeds 0-9",
        ],
        "klt_available_but_insufficient": "Seed result JSON files retain only held-out positioning aggregates (203 epochs/4676 measurements for KLT1; 209 epochs/4914 measurements for KLT2).",
        "consequence": "Sections E and F cannot be computed exactly; no inference was run and no observation-level output was reconstructed from aggregate moments.",
    }
    blocker_path = output / "model_response_blocker.json"
    blocker_path.write_text(json.dumps(blocker, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    frozen_after = frozen_snapshot(args.checkpoint_manifest.resolve(), args.ibiza_results.resolve())
    source_hashes_after = {"klt3": sha256_file(args.klt3), "ibiza": sha256_file(args.ibiza)}
    if source_hashes_before != source_hashes_after or frozen_before != frozen_after:
        raise RuntimeError("a frozen source changed during the read-only audit")

    klt_constellations = set(klt3.constellations.tolist())
    ibiza_constellations = set(ibiza.constellations.tolist())
    generated = [output / name for name in csv_outputs] + plot_paths + [blocker_path]
    klt_manifest = _load_json(args.klt3_manifest.resolve())
    ibiza_manifest = _load_json(args.ibiza_manifest.resolve())
    manifest = {
        "schema_version": 1,
        "status": "input_domain_audit_complete_model_response_blocked",
        "scientific_scope": "descriptive domain-shift analysis; no causal inference",
        "source_artifacts": _source_provenance(
            klt_manifest,
            ibiza_manifest,
            args.klt3_manifest.resolve(),
            args.ibiza_manifest.resolve(),
        ),
        "frozen_source_snapshot": frozen_after,
        "feature_order": [
            {"index": index, "name": name, "unit": unit}
            for index, (name, unit) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS, strict=True))
        ],
        "normalization": {
            "source": "frozen KLT3 StandardizeLayer",
            "mean": KLT3_MEAN.tolist(),
            "std": KLT3_STD.tolist(),
            "ibiza_fitted_normalization": False,
        },
        "percentiles": {"implementation": "numpy.percentile", "method": PERCENTILE_METHOD},
        "relationship_bins": {
            "elevation_rad": ELEVATION_BIN_EDGES_RAD.tolist(),
            "elevation_degrees_for_readability": np.rad2deg(ELEVATION_BIN_EDGES_RAD).tolist(),
            "cn0_snr0_div_1000": CN0_BIN_EDGES.tolist(),
            "semantics": "left-closed/right-open, except final bin includes its right edge",
        },
        "constellations": {
            "common": sorted(klt_constellations & ibiza_constellations),
            "klt3_only": sorted(klt_constellations - ibiza_constellations),
            "ibiza_only": sorted(ibiza_constellations - klt_constellations),
        },
        "controls": {
            "ground_truth_loaded": False,
            "model_implementation_imported": False,
            "checkpoint_deserialized": False,
            "model_parameters_modified": False,
            "inference_run": False,
            "training_run": False,
            "seed_and_observation_aggregation_mixed": False,
        },
        "validation": {
            "ibiza_npz_unchanged": source_hashes_before["ibiza"] == source_hashes_after["ibiza"] == IBIZA_SHA256,
            "klt3_npz_unchanged": source_hashes_before["klt3"] == source_hashes_after["klt3"] == KLT3_SHA256,
            "all_30_checkpoint_hashes_match": len(frozen_after["checkpoints"]) == 30,  # type: ignore[arg-type]
            "all_30_ibiza_jsonl_hashes_match": len(frozen_after["ibiza_inference_jsonl"]) == 30,  # type: ignore[arg-type]
            "feature_order_verified": True,
            "ibiza_scaler_fit_performed": False,
            "klt3_zscore_uses_frozen_scaler": True,
            "summary_counts_match_source_rows": all(
                int(row["count"]) == (8857 if row["dataset"] == "KLT3" else 73204)
                for row in feature_rows
            ),
            "ecdf_monotonic_and_terminal": all(
                np.all(np.diff(ecdf(dataset.features[:, index])[1]) >= 0.0)
                and ecdf(dataset.features[:, index])[1][-1] == 1.0
                for dataset in datasets for index in range(3)
            ),
            "constellation_partitions_match_totals": all(
                sum(int(np.count_nonzero(dataset.constellations == item)) for item in np.unique(dataset.constellations)) == dataset.features.shape[0]
                for dataset in datasets
            ),
        },
        "model_response_blocker": blocker,
        "generated_files": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in generated
        ],
    }
    manifest_path = output / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"KLT3 rows: {klt3.features.shape[0]}")
    print(f"Ibiza rows: {ibiza.features.shape[0]}")
    print(f"Output: {output}")
    print("Model-response status: blocked before new inference (see model_response_blocker.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
