from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from validation.rtklib_ibiza_controlled_baseline import core
from validation.tdl_9feature_tasgnss_analysis import evaluate_ibiza


def _solutions(
    timestamps: list[float], positions: list[list[float]] | None = None
) -> core.RtklibSolutions:
    count = len(timestamps)
    if positions is None:
        reference = np.asarray([4967979.25220, 125663.49615, 3984693.03705])
        positions = [(reference + [index + 1, 2, 3]).tolist() for index in range(count)]
    return core.RtklibSolutions(
        timestamps_gpst_s=np.asarray(timestamps, dtype=np.float64),
        ecef_m=np.asarray(positions, dtype=np.float64).reshape((-1, 3)),
        quality=np.ones(count, dtype=np.int16),
        satellite_count=np.full(count, 9, dtype=np.int16),
    )


def _support(timestamps: list[float]) -> core.FrozenSupport:
    values = np.asarray(timestamps, dtype=np.float64)
    ids = np.arange(values.size, dtype=np.int64)
    return core.FrozenSupport(ids, values, ids, values, "support-hash", {})


def _serialized_arrays() -> dict[str, np.ndarray]:
    reference = np.asarray([4967979.25220, 125663.49615, 3984693.03705])
    return {
        "timestamp_gpst_like_s": np.asarray([10.0, 20.0, 30.0]),
        "position_ecef_m": np.vstack(
            (reference + [1, 2, 3], reference + [2, 1, -1], reference + [3, -1, 2])
        ),
        "solution_quality": np.ones(3, dtype=np.int16),
        "satellite_count": np.full(3, 10, dtype=np.int16),
        "required_epoch_id": np.arange(3, dtype=np.int64),
        "required_timestamp_gpst_like_s": np.asarray([10.0, 20.0, 30.0]),
        "matched_solution_index": np.arange(3, dtype=np.int64),
        "support_complete": np.asarray([1], dtype=np.uint8),
        "support_sha256": np.asarray(["support"]),
        "config_sha256": np.asarray(["config"]),
        "observation_sha256": np.asarray(["observation"]),
        "navigation_sha256": np.asarray(["navigation"]),
        "raw_output_sha256": np.asarray(["raw"]),
        "ground_truth_available": np.asarray([0], dtype=np.uint8),
    }


def test_frozen_observation_navigation_and_reference_hashes() -> None:
    hashes = core.verify_scientific_inputs(include_reference=True)
    assert hashes == {
        "observation": core.OBSERVATION_SHA256,
        "navigation": core.NAVIGATION_SHA256,
        "ground_truth": core.REFERENCE_SHA256,
    }


def test_frozen_support_is_2849_and_identical_across_all_ten_seeds() -> None:
    support = core.load_frozen_primary_support()
    assert support.candidate_epoch_ids.shape == (2880,)
    assert support.required_epoch_ids.shape == (2849,)
    assert len(support.seed_artifact_sha256) == 10
    assert np.array_equal(
        support.required_epoch_ids,
        support.candidate_epoch_ids[
            np.isin(support.candidate_epoch_ids, support.required_epoch_ids)
        ],
    )


def test_support_matching_uses_timestamp_not_row_number() -> None:
    support = _support([10.0, 20.0, 30.0])
    solutions = _solutions([30.0, 999.0, 10.0, 20.0])
    match = core.match_required_support(solutions, support)
    np.testing.assert_array_equal(match["matched_solution_indices"], [2, 3, 0])
    assert match["exact_timestamp_match"] is True
    assert match["extra_rtklib_epoch_count"] == 1


def test_no_ground_truth_error_threshold_can_change_support() -> None:
    source = inspect.getsource(core.match_required_support)
    assert "ecef" not in source.lower()
    assert "error" not in source.lower()
    evaluation_source = inspect.getsource(core._evaluate_with_reference)
    policy = evaluation_source[evaluation_source.index("inclusion_policy") :]
    assert "no ground-truth error threshold" in policy


def test_ground_truth_is_unavailable_to_rtklib_execution() -> None:
    execute_source = inspect.getsource(core.execute_rtklib)
    solve_source = inspect.getsource(core.solve_full_interval)
    assert "reference" not in execute_source.lower()
    assert "include_reference=False" in solve_source
    assert "REFERENCE" not in solve_source
    assert "read_text" not in solve_source
    assert "json.load" not in solve_source


def test_perturbed_ground_truth_leaves_raw_positions_and_status_unchanged(
    tmp_path: Path,
) -> None:
    arrays = _serialized_arrays()
    position_path = tmp_path / "positions.npz"
    core.write_deterministic_npz(position_path, arrays)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "trusted_reference_ecef_m": {
                    "x": 4967979.25220,
                    "y": 125663.49615,
                    "z": 3984693.03705,
                }
            }
        ),
        encoding="utf-8",
    )
    before = core.raw_position_fingerprints(core.load_pre_ground_truth(position_path))
    primary, _comparison, leakage = core.evaluate_serialized_positions(
        position_path, reference_path
    )
    after = core.raw_position_fingerprints(core.load_pre_ground_truth(position_path))
    assert before == after
    assert primary["status"] == "complete_exact_support"
    assert leakage["raw_ecef_positions_unchanged"] is True
    assert leakage["raw_solution_timestamps_unchanged"] is True
    assert leakage["solver_status_unchanged"] is True
    assert leakage["enu_and_2d_3d_metrics_changed"] is True


def test_metric_implementation_agrees_with_existing_ibiza_evaluator() -> None:
    reference = np.asarray([4967979.25220, 125663.49615, 3984693.03705])
    positions = _serialized_arrays()["position_ecef_m"]
    assert core.solution_metrics(positions, reference) == evaluate_ibiza._solution_metrics(
        positions, reference
    )


def test_all_requested_metrics_are_independently_recomputed() -> None:
    reference = np.asarray([4967979.25220, 125663.49615, 3984693.03705])
    positions = _serialized_arrays()["position_ecef_m"]
    metrics = core.solution_metrics(positions, reference)
    enu = evaluate_ibiza.enu_errors(positions, reference)
    for dimension, values in (
        ("2d", np.sqrt(enu[:, 0] ** 2 + enu[:, 1] ** 2)),
        ("3d", np.sqrt(np.sum(enu**2, axis=1))),
    ):
        independently = {
            "mean": float(sum(values) / len(values)),
            "median": float(np.sort(values)[len(values) // 2]),
            "rms": float(np.sqrt(sum(value * value for value in values) / len(values))),
            "p68": float(np.quantile(values, 0.68, method="linear")),
            "p95": float(np.quantile(values, 0.95, method="linear")),
            "max": float(max(values)),
        }
        assert metrics[dimension] == pytest.approx(independently)
    assert metrics["east_rms"] == pytest.approx(np.sqrt(np.mean(enu[:, 0] ** 2)))
    assert metrics["north_rms"] == pytest.approx(np.sqrt(np.mean(enu[:, 1] ** 2)))
    assert metrics["up_rms"] == pytest.approx(np.sqrt(np.mean(enu[:, 2] ** 2)))


def test_exact_support_cannot_silently_shrink() -> None:
    support = _support([10.0, 20.0, 30.0])
    match = core.match_required_support(_solutions([10.0, 30.0]), support)
    assert match["missing_required_timestamps_gpst_s"] == [20.0]
    assert match["rtklib_solution_count_on_required_support"] == 2
    with pytest.raises(RuntimeError, match="incomplete"):
        core.require_complete_primary_support(match)
    arrays = _serialized_arrays()
    arrays["support_complete"] = np.asarray([0], dtype=np.uint8)
    result = core._evaluate_with_reference(
        arrays, np.asarray([4967979.25220, 125663.49615, 3984693.03705])
    )
    assert result["status"] == "incomplete_exact_support"
    assert result["metrics"] is None


def test_duplicate_and_missing_rtklib_timestamps_fail_validation(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.pos"
    duplicate.write_text(
        "2025/01/01 00:00:00.000 1 2 3 5 8\n"
        "2025/01/01 00:00:00.000 4 5 6 5 9\n",
        encoding="ascii",
    )
    with pytest.raises(RuntimeError, match="duplicate RTKLIB timestamps"):
        core.parse_rtklib_pos(duplicate)
    match = core.match_required_support(_solutions([10.0]), _support([10.0, 20.0]))
    with pytest.raises(RuntimeError, match="incomplete"):
        core.require_complete_primary_support(match)


def test_input_hashes_remain_unchanged_across_support_audit() -> None:
    before = core.verify_scientific_inputs(include_reference=True)
    core.load_frozen_primary_support()
    after = core.verify_scientific_inputs(include_reference=True)
    assert before == after


def test_official_rtklib_scientific_source_is_clean_and_checkout_is_unchanged() -> None:
    before = core.verify_rtklib_identity()
    scientific_changes = [
        row
        for row in before["status_short"]
        if row[3:].startswith("src/") or row[3:].endswith("rnx2rtkp.c")
    ]
    assert scientific_changes == []
    after = core.verify_rtklib_identity()
    core.assert_checkout_unchanged(before, after)


def test_deterministic_replay_has_identical_positions_and_metrics(tmp_path: Path) -> None:
    arrays = _serialized_arrays()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    core.write_deterministic_npz(first, arrays)
    core.write_deterministic_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    first_arrays = core.load_pre_ground_truth(first)
    second_arrays = core.load_pre_ground_truth(second)
    for name in first_arrays:
        np.testing.assert_array_equal(first_arrays[name], second_arrays[name])
    reference = np.asarray([4967979.25220, 125663.49615, 3984693.03705])
    assert core.solution_metrics(first_arrays["position_ecef_m"], reference) == (
        core.solution_metrics(second_arrays["position_ecef_m"], reference)
    )


def test_configuration_is_conventional_spp_and_frozen_before_full_run() -> None:
    text = core.CONFIG.read_text(encoding="utf-8")
    assert "pos1-posmode       =single" in text
    assert "pos1-frequency     =l1" in text
    assert "pos1-navsys        =13" in text
    elevation_lines = [
        line for line in text.splitlines() if line.strip().startswith("pos1-elmask")
    ]
    assert elevation_lines == ["pos1-elmask        =7"]
    assert float(elevation_lines[0].split("=", 1)[1]) == 7.0
    assert "pos1-ionoopt       =brdc" in text
    assert "pos1-tropopt       =saas" in text
    assert "pos1-sateph        =brdc" in text
    for forbidden in ("=ppp", "=dgps", "=kinematic", "=static"):
        assert forbidden not in text
    audit = json.loads(core.CONFIGURATION_AUDIT.read_text(encoding="utf-8"))
    assert audit["chosen_configuration"]["pos1_elmask_degrees"] == 7.0
    assert "predeclared scientific configuration choice" in audit["decision_policy"]


def test_smoke_requires_rtklib_header_to_confirm_seven_degree_mask() -> None:
    from validation.rtklib_ibiza_controlled_baseline import __main__ as command

    source = inspect.getsource(command.run_smoke)
    assert "% elev mask : 7.0 deg" in source
    assert '"elevation_mask_degrees": 7.0' in source


def test_controlled_comparison_uses_frozen_ten_seed_summary() -> None:
    reference = np.asarray([4967979.25220, 125663.49615, 3984693.03705])
    rtklib = core.solution_metrics(_serialized_arrays()["position_ecef_m"], reference)
    comparison = core.controlled_comparison(rtklib)
    learned = comparison["columns"]["HybridShareSysNet + TASGNSS"]
    neutral = comparison["columns"]["Neutral TASGNSS"]
    assert "ten seed-level metric summaries" in learned["aggregation"]
    assert neutral["aggregation"] == "single deterministic solution"
    assert comparison["support"].startswith("exact frozen 2,849")
