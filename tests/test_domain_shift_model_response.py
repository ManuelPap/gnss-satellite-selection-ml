from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from validation.domain_shift.model_response_probe import (
    DEFAULT_OUTPUT,
    EXPECTED_ROWS,
    FEATURE_NAMES,
    FROZEN_MEAN,
    FROZEN_STD,
    INPUT_SHA256,
    OUTPUT_SEMANTICS,
    REPOSITORY_ROOT,
    across_seed_summary_rows,
    distribution,
    forward_only,
    load_feature_datasets,
    named_array_sha256,
    per_seed_summary_rows,
    row_output_arrays,
    sha256_file,
    write_deterministic_npz,
)
from validation.ibiza_generalization.checkpoints import (
    ARCHITECTURES,
    load_frozen_model,
    validate_checkpoint_inventory,
)


PROBE_SOURCE = REPOSITORY_ROOT / "validation/domain_shift/model_response_probe.py"


@pytest.fixture(scope="module")
def datasets():
    return load_feature_datasets()


def test_all_three_exact_feature_artifacts_are_hash_verified_and_aligned(datasets) -> None:
    assert {dataset.name: sha256_file(dataset.source_path) for dataset in datasets} == INPUT_SHA256
    for dataset in datasets:
        assert dataset.features.shape == (EXPECTED_ROWS[dataset.name], 3)
        assert dataset.features.dtype == np.float64
        assert np.array_equal(dataset.row_index, np.arange(EXPECTED_ROWS[dataset.name]))
        assert dataset.epoch_index.shape == dataset.row_index.shape
        assert dataset.satellite_ids.shape == dataset.row_index.shape
        assert dataset.constellations.shape == dataset.row_index.shape
    assert FEATURE_NAMES == ("C/N0", "elevation", "OLS residual")


def test_all_30_checkpoint_hashes_and_embedded_normalization_match() -> None:
    inventory = validate_checkpoint_inventory()
    assert len(inventory.records) == inventory.distinct_hash_count == 30
    assert set((record.architecture, record.seed) for record in inventory.records) == {
        (architecture, seed) for architecture in ARCHITECTURES for seed in range(10)
    }
    assert np.array_equal(np.asarray(inventory.common_mean), FROZEN_MEAN)
    assert np.array_equal(np.asarray(inventory.common_std), FROZEN_STD)
    for record in inventory.records:
        assert sha256_file(REPOSITORY_ROOT / record.checkpoint_path) == record.sha256


def test_forward_boundary_is_eval_gradient_free_state_preserving_and_deterministic(datasets) -> None:
    sample = datasets[-1].features[:37]
    for architecture in ARCHITECTURES:
        frozen = load_frozen_model(architecture, 0)
        result = forward_only(frozen, sample)
        assert result.eval_mode
        assert result.gradients_disabled
        assert result.state_dict_unchanged
        assert result.repeated_forward_exact
        assert result.state_sha256_before == result.state_sha256_after
        assert tuple(result.outputs) == OUTPUT_SEMANTICS[architecture]
        assert all(values.shape == (37,) for values in result.outputs.values())
        assert np.array_equal(
            result.normalized_features,
            (sample - FROZEN_MEAN) / FROZEN_STD,
        )
        assert all(not value.requires_grad for value in frozen.model.parameters())


def test_released_output_semantics_are_preserved(datasets) -> None:
    sample = datasets[-1].features[:19]
    bias = forward_only(load_frozen_model("TDL-B", 0), sample).outputs
    weight = forward_only(load_frozen_model("TDL-W", 0), sample).outputs
    hybrid = forward_only(load_frozen_model("TDL-BW", 0), sample).outputs
    assert tuple(bias) == ("bias_m",)
    assert np.any(bias["bias_m"] < 0.0)  # released TDL-B has no final activation
    assert tuple(weight) == ("weight",)
    assert np.all((weight["weight"] >= 0.0) & (weight["weight"] <= 10.0))
    assert tuple(hybrid) == ("weight", "bias_m")
    assert np.all((hybrid["weight"] >= 0.0) & (hybrid["weight"] <= 1.0))
    assert np.all(hybrid["bias_m"] >= 0.0)


def test_row_schema_preserves_exact_source_identity_and_count(datasets) -> None:
    dataset = datasets[0]
    frozen = load_frozen_model("TDL-BW", 1)
    result = forward_only(frozen, dataset.features)
    arrays = row_output_arrays(dataset, frozen, result)
    assert arrays["features"].shape == (EXPECTED_ROWS["KLT1"], 3)
    assert arrays["normalized_features"].shape == arrays["features"].shape
    for name in ("row_index", "epoch_index", "satellite_ids", "constellations"):
        np.testing.assert_array_equal(arrays[name], getattr(dataset, name))
    np.testing.assert_array_equal(arrays["features"], dataset.features)
    assert arrays["predicted_weight"].shape == dataset.row_index.shape
    assert arrays["predicted_bias_m"].shape == dataset.row_index.shape


def test_npz_writer_and_content_hash_are_deterministic(tmp_path: Path) -> None:
    arrays = {
        "features": np.arange(12, dtype=np.float64).reshape(4, 3),
        "ids": np.asarray(["G01", "E02", "R03", "C04"]),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    assert named_array_sha256(arrays.items()) == named_array_sha256(arrays.items())


def test_per_seed_and_across_seed_summaries_reconcile_exactly() -> None:
    all_rows = []
    for seed in range(10):
        values = np.asarray([seed - 1.0, seed + 2.0, seed + 7.0], dtype=np.float64)
        rows = per_seed_summary_rows("toy", "TDL-B", seed, {"bias_m": values})
        row = rows[0]
        expected = distribution(values, "bias_m")
        for key, value in expected.items():
            assert row[key] == value
        all_rows.extend(rows)
    across = across_seed_summary_rows(all_rows)
    mean_row = next(row for row in across if row["per_seed_metric"] == "mean")
    per_seed_means = np.asarray([float(row["mean"]) for row in all_rows])
    assert mean_row["mean"] == float(np.mean(per_seed_means))
    assert mean_row["population_sd"] == float(np.std(per_seed_means, ddof=0))
    assert mean_row["seed_count"] == 10


def test_probe_has_no_positioning_ground_truth_training_or_scaler_fit_route() -> None:
    source = PROBE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    direct_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("inference" in name or "ground_truth" in name for name in direct_imports)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden_calls = {
        "solve_paper_weighted_position",
        "solve_paper_bias_position",
        "solve_paper_hybrid_position",
        "evaluate_epoch",
        "evaluate_architecture",
        "prepare_dataset",
        "load_ground_truth_window",
        "backward",
        "train",
        "fit",
        "fit_transform",
    }
    assert calls.isdisjoint(forbidden_calls)
    assert "torch.optim" not in source
    assert "optimizer" not in calls
    loader_tree = ast.parse(ast.get_source_segment(source, next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_one"
    )) or "")
    string_literals = {
        node.value for node in ast.walk(loader_tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not any("ground_truth" in value for value in string_literals)


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="external generated probe is not present")
def test_published_manifest_controls_and_summaries_match_all_row_outputs() -> None:
    manifest_path = DEFAULT_OUTPUT / "model_response_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_artifact_count"] == 90
    controls = manifest["controls"]
    assert controls["feature_hashes_unchanged"]
    assert controls["all_30_checkpoint_hashes_match_and_unchanged"]
    assert controls["model_eval_active_for_all_90_forwards"]
    assert controls["gradients_disabled_for_all_90_forwards"]
    assert controls["state_dict_unchanged_for_all_90_forwards"]
    assert controls["repeated_full_forward_exact_for_all_90_forwards"]
    assert not controls["ground_truth_read"]
    assert not controls["positioning_or_wls_called"]
    assert not controls["optimizer_created"]
    assert not controls["backward_called"]
    assert not controls["training_performed"]
    assert not controls["normalization_fitted"]
    datasets_by_name = {dataset.name: dataset for dataset in load_feature_datasets()}
    checked_artifacts = set()
    with (DEFAULT_OUTPUT / "per_seed_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 120
    for row in rows:
        path = DEFAULT_OUTPUT / "rows" / (
            f"{row['dataset'].lower()}_{row['architecture'].lower().replace('-', '_')}_seed_{int(row['seed']):02d}.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            values = archive[f"predicted_{row['output']}"]
            if path not in checked_artifacts:
                source = datasets_by_name[row["dataset"]]
                np.testing.assert_array_equal(archive["row_index"], source.row_index)
                np.testing.assert_array_equal(archive["epoch_index"], source.epoch_index)
                np.testing.assert_array_equal(archive["satellite_ids"], source.satellite_ids)
                np.testing.assert_array_equal(archive["constellations"], source.constellations)
                np.testing.assert_array_equal(archive["features"], source.features)
                np.testing.assert_array_equal(
                    archive["normalized_features"],
                    (source.features - FROZEN_MEAN) / FROZEN_STD,
                )
                checked_artifacts.add(path)
        expected = distribution(values, row["output"])
        assert int(row["row_count"]) == EXPECTED_ROWS[row["dataset"]]
        for metric, value in expected.items():
            assert float(row[metric]) == value
    assert len(checked_artifacts) == 90

    with (DEFAULT_OUTPUT / "artifact_hashes.csv").open(encoding="utf-8", newline="") as stream:
        artifact_rows = list(csv.DictReader(stream))
    assert len(artifact_rows) == 119
    for row in artifact_rows:
        path = DEFAULT_OUTPUT / row["path"]
        assert path.stat().st_size == int(row["size_bytes"])
        assert sha256_file(path) == row["sha256"]

    with (DEFAULT_OUTPUT / "across_seed_summary.csv").open(encoding="utf-8", newline="") as stream:
        across_rows = list(csv.DictReader(stream))
    typed_rows = [
        {key: value for key, value in row.items() if value != ""}
        for row in rows
    ]
    recomputed = across_seed_summary_rows(typed_rows)
    assert len(across_rows) == len(recomputed) == 126
    for written, expected in zip(across_rows, recomputed, strict=True):
        for key, value in expected.items():
            if isinstance(value, float):
                assert float(written[key]) == value
            elif isinstance(value, int):
                assert int(written[key]) == value
            else:
                assert written[key] == value
