from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import validation.tdl_3feature_paper_analysis.positioning_ablation as ablation


SOURCE = ablation.REPOSITORY_ROOT / "validation/tdl_3feature_paper_analysis/positioning_ablation.py"


@pytest.fixture(scope="module")
def authoritative() -> ablation.AuthoritativeInputs:
    return ablation.validate_authoritative_inputs()


def test_authoritative_ibiza_responses_frozen_results_and_checkpoints_match(
    authoritative: ablation.AuthoritativeInputs,
) -> None:
    assert ablation.sha256_file(ablation.DEFAULT_IBIZA) == ablation.IBIZA_SHA256
    assert len(authoritative.responses) == 30
    assert authoritative.dataset["features"].shape == (73204, 3)
    assert authoritative.dataset["epoch_offsets"].shape == (2857,)
    assert authoritative.provenance["verified_ibiza_response_files"] == 30
    checkpoint = authoritative.provenance["checkpoint_manifest"]
    assert checkpoint["sha256"] == ablation.CHECKPOINT_MANIFEST_SHA256
    assert checkpoint["checkpoint_count"] == 30
    assert checkpoint["checkpoints_deserialized"] is False
    assert authoritative.frozen_validation.run_manifest_sha256 == (
        ablation.FROZEN_RESULT_MANIFEST_SHA256
    )


def test_all_stored_outputs_map_one_to_one_to_exact_ibiza_rows(
    authoritative: ablation.AuthoritativeInputs,
) -> None:
    dataset = authoritative.dataset
    for architecture in ablation.ARCHITECTURES:
        for seed in ablation.SEEDS:
            outputs = authoritative.responses[(architecture, seed)]
            assert all(values.shape == (73204,) for values in outputs.values())
            if "weight" in outputs:
                assert np.all(outputs["weight"] > 0.0)
    assert np.array_equal(dataset["epoch_index"], np.repeat(
        np.arange(2856), np.diff(dataset["epoch_offsets"])
    ))


def test_recomputed_original_subset_matches_published_positions_and_diagnostics(
    authoritative: ablation.AuthoritativeInputs,
) -> None:
    dataset = authoritative.dataset
    for architecture in ablation.ARCHITECTURES:
        output = authoritative.responses[(architecture, 0)]
        if architecture == "TDL-B":
            series = ablation.solve_series(
                dataset, route="bias", bias_m=output["bias_m"], epoch_indices=(0, 1, 2)
            )
        elif architecture == "TDL-W":
            series = ablation.solve_series(
                dataset, route="weight", weight=output["weight"], epoch_indices=(0, 1, 2)
            )
        else:
            series = ablation.solve_series(
                dataset,
                route="hybrid",
                bias_m=output["bias_m"],
                weight=output["weight"],
                epoch_indices=(0, 1, 2),
            )
        path = ablation.DEFAULT_FROZEN_RESULTS_DIR / ablation.result_filename(
            architecture, 0
        )
        expected = [json.loads(line) for line in path.read_text().splitlines()[:3]]
        for index, row in enumerate(expected):
            np.testing.assert_allclose(
                series.state[index],
                row["estimated_receiver_state"],
                rtol=0.0,
                atol=ablation.STATE_REGRESSION_ATOL,
            )
            assert series.solved[index]
            assert series.rank[index] == row["rank"]
            assert series.iteration_count[index] == row["iteration_count"]


def test_uniform_positive_weight_scaling_is_numerically_invariant(
    authoritative: ablation.AuthoritativeInputs,
) -> None:
    dataset = authoritative.dataset
    rows = dataset["features"].shape[0]
    indices = (0, 1000, 2855)
    states = []
    for scale in (0.25, 1.0, 7.0):
        result = ablation.solve_series(
            dataset,
            route="weight",
            weight=np.full(rows, scale, dtype=np.float64),
            epoch_indices=indices,
        )
        assert np.all(result.solved)
        states.append(result.state)
    np.testing.assert_allclose(states[0], states[1], rtol=0.0, atol=ablation.UNIFORM_SCALE_ATOL)
    np.testing.assert_allclose(states[2], states[1], rtol=0.0, atol=ablation.UNIFORM_SCALE_ATOL)


def test_ablation_plan_changes_only_frozen_bias_and_weight_inputs() -> None:
    assert ablation.VARIANTS == {
        "TDL-B": ("B-neutral", "B-original"),
        "TDL-W": ("W-neutral", "W-original"),
        "TDL-BW": (
            "BW-neutral",
            "BW-bias-only",
            "BW-weight-only",
            "BW-full",
        ),
    }
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "load_frozen_model" not in calls
    assert "infer_neural_outputs" not in calls
    assert "backward" not in calls
    assert "train" not in calls
    assert "fit" not in calls
    assert "fit_transform" not in calls
    assert "torch.optim" not in source
    assert "solve_paper_weighted_position" in calls
    assert "solve_paper_bias_position" in calls
    assert "solve_paper_hybrid_position" in calls


def test_ground_truth_has_no_positioning_stage_route() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    positioning_names = {
        "_solve_epoch",
        "solve_series",
        "compute_original_positions",
        "compute_ablation_positions",
        "validate_uniform_weight_scaling",
    }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in positioning_names:
            segment = ast.get_source_segment(source, node) or ""
            assert "load_reference_manifest" not in segment
            assert "reference_ecef" not in segment
            assert "evaluate_ecef_positions" not in segment
    solver_signature = ast.get_source_segment(
        source,
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_solve_epoch"),
    )
    assert solver_signature is not None and "ground_truth" not in solver_signature


def _toy_series(solved: np.ndarray, values: np.ndarray) -> ablation.EvaluatedSeries:
    count = len(solved)
    positioning = ablation.PositionSeries(
        state=np.zeros((count, 7)),
        solved=solved,
        status=np.where(solved, "solved", "failed"),
        rank=np.full(count, 4),
        condition_number=np.ones(count),
        iteration_count=np.ones(count, dtype=np.int64),
    )
    return ablation.EvaluatedSeries(
        positioning=positioning,
        enu_error_m=np.column_stack((values, values, values)),
        e2d_m=values,
        abs_up_m=values + 1.0,
        e3d_m=values + 2.0,
    )


def test_paired_rows_use_only_common_solved_epochs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ablation, "ARCHITECTURES", ("TDL-B",))
    monkeypatch.setattr(ablation, "SEEDS", (0,))
    monkeypatch.setitem(
        ablation.PAIRED_COMPARISONS, "TDL-B", (("B-original", "B-neutral"),)
    )
    first = _toy_series(np.asarray([True, False, True]), np.asarray([4.0, np.nan, 8.0]))
    second = _toy_series(np.asarray([True, True, False]), np.asarray([1.0, 2.0, np.nan]))
    path = tmp_path / "paired.csv"
    summaries = ablation.write_paired_epoch_and_summaries(
        path,
        {
            ("TDL-B", 0, "B-original"): first,
            ("TDL-B", 0, "B-neutral"): second,
        },
    )
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 1
    assert int(rows[0]["accepted_epoch_index"]) == 0
    assert len(summaries) == 3
    assert {row["common_solved_epochs"] for row in summaries} == {1}


def test_factorial_interaction_formula_is_exact() -> None:
    full = np.asarray([10.0, 30.0, 5.0])
    bias = np.asarray([2.0, 7.0, 1.0])
    weight = np.asarray([3.0, 11.0, 8.0])
    neutral = np.asarray([4.0, 13.0, 6.0])
    expected = full - bias - weight + neutral
    np.testing.assert_array_equal(
        ablation.factorial_interaction(full, bias, weight, neutral), expected
    )


@pytest.mark.skipif(not ablation.DEFAULT_OUTPUT.is_dir(), reason="external ablation is not published")
def test_published_ablation_reconciles_counts_hashes_and_controls() -> None:
    root = ablation.DEFAULT_OUTPUT
    manifest = json.loads((root / "positioning_ablation_manifest.json").read_text())
    assert manifest["original_position_regression"]["all_30_passed"]
    assert len(manifest["original_position_regression"]["records"]) == 30
    assert manifest["counts"]["per_seed_metric_rows"] == 80
    controls = manifest["controls"]
    assert controls["ground_truth_read_during_positioning"] is False
    assert controls["ground_truth_passed_to_solver"] is False
    assert controls["neural_inference_rerun_for_ablation"] is False
    assert controls["validation_only_single_epoch_forward_performed_before_ablation"]
    assert controls["checkpoint_deserialized_for_ablation"] is False
    assert controls["same_epoch_satellite_support_all_variants"]
    assert controls["same_initial_states_all_variants"]
    assert controls["only_frozen_bias_and_or_weight_inputs_differ"]
    with (root / "positioning_ablation_artifact_hashes.csv").open(newline="") as stream:
        artifact_rows = list(csv.DictReader(stream))
    for row in artifact_rows:
        path = root / row["path"]
        assert path.stat().st_size == int(row["size_bytes"])
        assert ablation.sha256_file(path) == row["sha256"]
    with (root / "positioning_ablation_per_seed.csv").open(newline="") as stream:
        per_seed = list(csv.DictReader(stream))
    assert len(per_seed) == 80
    assert {int(row["solved_epochs"]) + int(row["failed_epochs"]) for row in per_seed} == {2856}
    with np.load(root / "positioning_ablation_epoch_results.npz", allow_pickle=False) as archive:
        assert archive["accepted_epoch_index"].shape == (80 * 2856,)
        assert archive["estimated_receiver_state"].shape == (80 * 2856, 7)
        for row in per_seed:
            mask = (
                (archive["architecture"] == row["architecture"])
                & (archive["seed"] == int(row["seed"]))
                & (archive["ablation"] == row["ablation"])
            )
            assert np.count_nonzero(mask) == 2856
            solved = archive["solved"][mask].astype(bool)
            assert np.count_nonzero(solved) == int(row["solved_epochs"])
            assert float(row["e3d_mean_m"]) == float(np.mean(archive["e3d_m"][mask][solved]))
