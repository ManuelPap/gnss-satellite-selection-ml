from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from validation.tdl_3feature_paper_analysis.audit import (
    DEFAULT_CHECKPOINT_MANIFEST,
    DEFAULT_IBIZA,
    DEFAULT_IBIZA_MANIFEST,
    DEFAULT_IBIZA_RESULTS,
    DEFAULT_KLT3,
    FEATURE_NAMES,
    IBIZA_SHA256,
    KLT3_MEAN,
    KLT3_STD,
    Dataset,
    constellation_summary,
    distribution,
    ecdf,
    feature_summary,
    frozen_snapshot,
    load_datasets,
    sha256_file,
    zscore_summary,
)
from validation.paper_weightnet.core import StandardizeLayer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SOURCE = REPOSITORY_ROOT / "validation/tdl_3feature_paper_analysis/audit.py"


def test_frozen_ibiza_and_all_model_artifacts_match_manifests() -> None:
    assert sha256_file(DEFAULT_IBIZA) == IBIZA_SHA256
    snapshot = frozen_snapshot(DEFAULT_CHECKPOINT_MANIFEST, DEFAULT_IBIZA_RESULTS)
    assert len(snapshot["checkpoints"]) == 30
    assert len(snapshot["ibiza_inference_jsonl"]) == 30
    assert len(snapshot["klt_held_out_seed_results"]) == 30


def test_feature_order_named_columns_and_constellation_partitions() -> None:
    klt3, ibiza = load_datasets(DEFAULT_KLT3, DEFAULT_IBIZA, DEFAULT_IBIZA_MANIFEST)
    assert FEATURE_NAMES == ("C/N0", "elevation", "OLS residual")
    assert klt3.features.shape == (8857, 3)
    assert ibiza.features.shape == (73204, 3)
    assert sum(np.count_nonzero(klt3.constellations == value) for value in np.unique(klt3.constellations)) == 8857
    assert sum(np.count_nonzero(ibiza.constellations == value) for value in np.unique(ibiza.constellations)) == 73204


def test_frozen_scaler_matches_standardize_layer_without_ibiza_fit() -> None:
    klt3, ibiza = load_datasets(DEFAULT_KLT3, DEFAULT_IBIZA, DEFAULT_IBIZA_MANIFEST)
    sample = np.vstack((klt3.features[:4], ibiza.features[:4]))
    expected = (sample - KLT3_MEAN) / KLT3_STD
    layer = StandardizeLayer(
        torch.tensor(KLT3_MEAN, dtype=torch.float64),
        torch.tensor(KLT3_STD, dtype=torch.float64),
    )
    actual = layer(torch.tensor(sample, dtype=torch.float64)).detach().numpy()
    np.testing.assert_array_equal(actual, expected)


def test_summary_counts_ecdf_and_out_of_range_are_exact() -> None:
    klt3, ibiza = load_datasets(DEFAULT_KLT3, DEFAULT_IBIZA, DEFAULT_IBIZA_MANIFEST)
    summaries = feature_summary((klt3, ibiza))
    assert [row["count"] for row in summaries] == [8857] * 3 + [73204] * 3
    z_rows = zscore_summary((klt3, ibiza), klt3.features.min(0), klt3.features.max(0))
    assert all(row["fraction_outside_klt3_raw_range"] == 0.0 for row in z_rows[:3])
    for dataset in (klt3, ibiza):
        for index in range(3):
            x, y = ecdf(dataset.features[:, index])
            assert np.all(np.diff(x) >= 0.0)
            assert np.all(np.diff(y) > 0.0)
            assert y[-1] == 1.0


def test_percentiles_use_numpy_linear_convention() -> None:
    values = np.asarray([0.0, 1.0, 10.0, 100.0])
    summary = distribution(values)
    expected = np.percentile(values, [5, 25, 50, 75, 95], method="linear")
    assert [summary[key] for key in ("p05", "p25", "median", "p75", "p95")] == expected.tolist()


def test_constellation_counts_repeat_source_partition_counts() -> None:
    toy = Dataset(
        "toy",
        np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]),
        np.asarray(["G", "G", "E"]),
    )
    rows = constellation_summary((toy,))
    counts = {(row["constellation"], row["feature"]): row["count"] for row in rows}
    assert counts[("G", "C/N0")] == 2
    assert counts[("E", "OLS residual")] == 1


def test_analysis_has_no_model_or_ground_truth_data_path() -> None:
    tree = ast.parse(AUDIT_SOURCE.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imported
    assert not any("inference" in name or "held_out" in name for name in imported)
    source = AUDIT_SOURCE.read_text(encoding="utf-8")
    assert "ground_truth_geodetic_deg_m\"]" not in source
    assert "fit_transform" not in source
