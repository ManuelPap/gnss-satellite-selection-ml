from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from validation.domain_shift.freeze_klt_features import (
    DEFAULT_OUTPUT,
    EXPECTED_COUNTS,
    FEATURE_COLUMNS,
    HISTORICAL_ARRAY_MAP,
    HISTORICAL_ARRAY_MEANINGS,
    IBIZA_FEATURE_SHA256,
    KLT1_PAPER_TRACE_SHA256,
    KLT1_WEIGHT_TRACE_SHA256,
    OUTPUT_FILENAMES,
    array_sha256,
    build_cache_arrays,
    named_array_sha256,
    sha256,
    validate_cache_arrays,
    write_deterministic_npz,
)
from validation.paper_weightnet import held_out
from validation.paper_weightnet.held_out import (
    DATASET_SPECS,
    FEATURE_NAMES,
    FEATURE_UNITS,
    FeatureInputPaths,
    PreparedFeatureDataset,
    PreparedFeatureEpoch,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SOURCE = (
    REPOSITORY_ROOT / "validation/domain_shift/freeze_klt_features.py"
)
MANIFEST = DEFAULT_OUTPUT / "feature_cache_manifest.json"


def _epoch(
    index: int, ids: tuple[str, ...], raw_snr: tuple[float, ...]
) -> PreparedFeatureEpoch:
    count = len(ids)
    snr = np.asarray(raw_snr, dtype=np.float64)
    elevation = np.arange(count, dtype=np.float64) / 10.0 + 0.4
    residual = np.arange(count, dtype=np.float64) - 1.0
    return PreparedFeatureEpoch(
        valid_epoch_index=index,
        candidate_epoch_index=index,
        split_epoch_index=100 + index,
        epoch_time=1000.25 + index,
        satellite_ids=np.asarray(ids, dtype="U3"),
        satellite_numbers=np.arange(1, count + 1, dtype=np.int64) + index * 10,
        raw_pseudorange_m=np.arange(count, dtype=np.float64) + 20_000_000.0,
        raw_snr_units=snr,
        features=np.column_stack((snr / 1000.0, elevation, residual)),
        satellite_positions_ecef_m=np.arange(count * 3, dtype=np.float64).reshape(
            count, 3
        ),
        satellite_clock_bias_s=np.arange(count, dtype=np.float64) * 1.0e-6,
        corrected_pseudorange_m=np.arange(count, dtype=np.float64) + 19_999_999.0,
        system_clock_indices=np.full(count, 3, dtype=np.int64),
        initial_ols_state=np.arange(7, dtype=np.float64) + index,
    )


def _prepared() -> PreparedFeatureDataset:
    inputs = FeatureInputPaths(
        observation=Path("observation.obs"),
        ephemeris_patterns=("navigation.*",),
        tdl_dir=Path("tdl"),
        pyrtklib_site=Path("pyrtklib"),
    )
    return PreparedFeatureDataset(
        spec=DATASET_SPECS["KLT1"],
        inputs=inputs,
        raw_split_epoch_count=2,
        candidate_epoch_count=2,
        epochs=(
            _epoch(0, ("G01", "E13"), (36_000.0, 42_000.0)),
            _epoch(1, ("R09",), (31_000.0,)),
        ),
        invalid_epochs=(),
        input_provenance={},
    )


def test_actual_historical_feature_construction_order_is_explicit() -> None:
    source = inspect.getsource(held_out.prepare_feature_dataset)
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "construct_features"
    ]
    assert len(calls) == 1
    assert [argument.id for argument in calls[0].args] == [  # type: ignore[union-attr]
        "snr",
        "elevation",
        "residual",
    ]
    assert tuple(column["source"] for column in FEATURE_COLUMNS) == (
        "SNR[0]/1000",
        "historical final OLS azel[:,1]",
        "historical equal-weight OLS residual",
    )
    assert FEATURE_NAMES == ("C/N0", "elevation", "OLS residual")
    assert FEATURE_UNITS == ("dB-Hz-like SNR[0]/1000", "radian", "metre")


def test_feature_preparation_has_no_ground_truth_route() -> None:
    signature = inspect.signature(held_out.resolve_feature_input_paths)
    assert "ground_truth" not in signature.parameters
    assert "ground_truth" not in FeatureInputPaths.__dataclass_fields__
    tree = ast.parse(inspect.getsource(held_out.prepare_feature_dataset))
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "ground_truth" not in names
    assert "load_ground_truth_window" not in names
    assert "nearest_ground_truth" not in names


def test_cache_schema_feature_columns_counts_and_constellations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    monkeypatch.setitem(EXPECTED_COUNTS, "KLT1", {"epochs": 2, "rows": 3})
    arrays = build_cache_arrays(prepared)
    validate_cache_arrays("KLT1", prepared, arrays)

    assert arrays["features"].shape == (3, 3)
    assert arrays["features"].dtype == np.float64
    np.testing.assert_array_equal(
        arrays["features"],
        np.column_stack(
            (
                arrays["cn0_snr0_div_1000"],
                arrays["elevation_rad"],
                arrays["ols_residual_m"],
            )
        ),
    )
    np.testing.assert_array_equal(
        arrays["features"][:, 0], arrays["raw_snr_units"] / 1000.0
    )
    np.testing.assert_array_equal(arrays["epoch_offsets"], [0, 2, 3])
    np.testing.assert_array_equal(arrays["epoch_row_counts"], [2, 1])
    np.testing.assert_array_equal(arrays["epoch_index"], [0, 0, 1])
    np.testing.assert_array_equal(arrays["row_index_within_epoch"], [0, 1, 0])
    assert arrays["constellations"].tolist() == ["G", "E", "R"]
    assert np.all(np.isfinite(arrays["features"]))


def test_npz_and_content_hashes_are_deterministic(tmp_path: Path) -> None:
    arrays = build_cache_arrays(_prepared())
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, arrays)

    assert first.read_bytes() == second.read_bytes()
    assert sha256(first) == sha256(second)
    assert named_array_sha256(arrays.items()) == named_array_sha256(arrays.items())
    with np.load(first, allow_pickle=False) as cache:
        assert cache.files == list(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(cache[name], expected)


def test_established_array_hash_is_shape_dtype_and_value_sensitive() -> None:
    values = np.asarray([1.0, 2.0], dtype=np.float64)
    assert array_sha256(values) != array_sha256(values.astype(np.float32))
    assert array_sha256(values) != array_sha256(values.reshape(1, 2))
    assert array_sha256(values) != array_sha256(np.asarray([1.0, 3.0]))


def test_generator_has_no_inference_training_or_learned_wls_calls() -> None:
    tree = ast.parse(GENERATOR_SOURCE.read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden = {
        "load_frozen_weightnet",
        "load_state_dict",
        "torch.load",
        "forward",
        "infer_weights",
        "evaluate_epoch",
        "evaluate_prepared_dataset",
        "solve_paper_weighted_position",
        "backward",
        "step",
        "normalize_with_frozen_klt3",
        "load_ground_truth_window",
        "prepare_dataset",
    }
    assert called.isdisjoint(forbidden)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imports


def test_same_stage_authority_names_exactly_nine_non_ground_truth_arrays() -> None:
    assert tuple(HISTORICAL_ARRAY_MAP) == (
        "epoch_offsets",
        "epoch_times",
        "features",
        "satellite_ids",
        "satellite_positions",
        "satellite_clock_bias",
        "corrected_pseudorange",
        "system_clock_indices",
        "initial_states",
    )
    assert set(HISTORICAL_ARRAY_MEANINGS) == set(HISTORICAL_ARRAY_MAP)
    assert "gt_times" not in HISTORICAL_ARRAY_MAP
    assert "ground_truth" not in HISTORICAL_ARRAY_MAP


@pytest.mark.skipif(not MANIFEST.is_file(), reason="generated KLT caches absent")
def test_generated_caches_and_manifest_satisfy_frozen_contract() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["ground_truth"] == {
        "ground_truth_file_read": False,
        "ground_truth_path_resolved": False,
        "ground_truth_present_in_cache": False,
        "ground_truth_used_for_feature_generation": False,
        "separation": manifest["ground_truth"]["separation"],
    }
    boundary = manifest["execution_boundary"]
    assert boundary["historical_equal_weight_ols_invoked"] is True
    assert all(
        boundary[name] is False
        for name in (
            "checkpoint_loaded",
            "neural_model_constructed",
            "neural_forward_pass_invoked",
            "optimizer_constructed",
            "backward_invoked",
            "training_invoked",
            "learned_wls_invoked",
            "learned_position_computed",
        )
    )
    assert manifest["feature_contract"]["normalization_fitted"] is False
    assert manifest["feature_contract"]["normalization_applied"] is False
    assert manifest["immutability"]["source_and_frozen_files_unchanged"] is True
    assert manifest["immutability"]["historical_results_unchanged"] is True
    assert (
        manifest["immutability"]["klt1_weight_trace"]["sha256"]
        == KLT1_WEIGHT_TRACE_SHA256
    )
    assert (
        manifest["immutability"]["klt1_paper_trace"]["sha256"]
        == KLT1_PAPER_TRACE_SHA256
    )
    assert (
        manifest["immutability"]["ibiza_feature_artifact"]["sha256"]
        == IBIZA_FEATURE_SHA256
    )

    for dataset, expected in EXPECTED_COUNTS.items():
        record = manifest["datasets"][dataset]
        assert record["epoch_count"] == expected["epochs"]
        assert record["row_count"] == expected["rows"]
        assert sum(record["constellation_counts"].values()) == expected["rows"]
        assert record["historical_cross_check"][
            "all_available_non_ground_truth_array_hashes_match"
        ] is True
        assert record["historical_cross_check"]["matched_array_count"] == 9
        authority = record["historical_cross_check"]["same_stage_authority"]
        assert authority["support"] == "complete retained multi-constellation epoch"
        assert authority["ground_truth_arrays_excluded"] == [
            "gt_times",
            "ground_truth",
        ]
        for comparison in record["historical_cross_check"][
            "array_comparisons"
        ].values():
            assert comparison["matches"] is True
            assert comparison["expected_sha256"] == comparison["actual_sha256"]
        artifact = DEFAULT_OUTPUT / OUTPUT_FILENAMES[dataset]
        assert sha256(artifact) == record["artifact"]["sha256"]
        with np.load(artifact, allow_pickle=False) as cache:
            assert cache["features"].shape == (expected["rows"], 3)
            assert cache["epoch_offsets"].shape == (expected["epochs"] + 1,)
            assert int(cache["epoch_offsets"][-1]) == expected["rows"]
            assert int(cache["epoch_row_counts"].sum()) == expected["rows"]
            assert np.all(np.isfinite(cache["features"]))
            np.testing.assert_array_equal(
                cache["features"],
                np.column_stack(
                    (
                        cache["cn0_snr0_div_1000"],
                        cache["elevation_rad"],
                        cache["ols_residual_m"],
                    )
                ),
            )
            assert (
                named_array_sha256((name, cache[name]) for name in cache.files)
                == record["artifact"]["content_sha256"]
            )

    deterministic = manifest["determinism"]
    assert deterministic["preprocessing_runs"] == 2
    assert deterministic["fresh_temporary_output_locations"] == 2
    for record in deterministic["datasets"].values():
        assert record["arrays_exactly_equal"] is True
        assert record["npz_bytes_identical"] is True
        assert record["first_npz_sha256"] == record["second_npz_sha256"]
        assert record["first_content_sha256"] == record["second_content_sha256"]

    supplementary = manifest["supplementary_gps_only_diagnostics"]
    assert supplementary["used_as_same_stage_regression_authority"] is False
    assert supplementary["full_held_out_support_count"] == 17
    assert supplementary["gps_only_support_count"] == 8
    assert "not numerical error" in supplementary["reason"]
    trace = supplementary["klt1_nn_weight_trace"]
    assert trace["cn0_exactly_equal"] is True
    assert trace["rows_compared"] == 8
    assert trace["ols_residuals_expected_to_match"] is False
    assert trace["ols_states_expected_to_match"] is False
    assert trace["ols_states_exactly_equal"] is False
    assert trace["maximum_absolute_ols_residual_difference_m"] > 0.0
    assert all(
        supplementary["paper_epoch_trace"][
            "supplementary_common_input_checks"
        ].values()
    )
