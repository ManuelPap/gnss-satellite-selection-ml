from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from validation.tasgnss_comparison.core import (
    IBIZA_NPZ_SHA256,
    accuracy_summary,
    enu_errors,
    historical_neutral_rows,
    paired_delta_summary,
    phase_a_compatibility,
    sha256_file,
    trusted_reference_ecef,
)


ROOT = Path(__file__).resolve().parents[1]
PHD_ROOT = ROOT.parent
IBIZA_NPZ = PHD_ROOT / "external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz"
REFERENCE = ROOT / "validation/ibiza_generalization/ibiz00esp_reference.json"
AUDIT = ROOT / "validation/tasgnss_comparison/solver_audit.json"
RUNNER = ROOT / "validation/tasgnss_comparison/run_current_stack.py"
HISTORICAL = PHD_ROOT / "external_data/domain_shift/positioning_ablation/positioning_ablation_epoch_results.npz"
OUTPUT = PHD_ROOT / "external_data/tasgnss_comparison/ibiza_neutral"


def test_frozen_ibiza_hash_is_unchanged() -> None:
    assert sha256_file(IBIZA_NPZ) == IBIZA_NPZ_SHA256


def test_phase_a_gate_rejects_missing_fixed_atmosphere() -> None:
    with np.load(IBIZA_NPZ, allow_pickle=False) as data:
        gate = phase_a_compatibility(data.files)
    assert gate["passed"] is False
    assert gate["decision"] == "stop_phase_a"
    assert gate["missing_core_arrays"] == []
    assert gate["missing_fixed_correction_arrays"] == [
        "tasgnss_broadcast_ionosphere_m",
        "tasgnss_saastamoinen_troposphere_m",
    ]
    assert gate["sagnac_exactly_derivable_from_frozen_geometry"] is True


def test_current_commit_backend_and_external_status_are_recorded() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["current_tasgnss"]["commit"] == "fdd7e8ebc0019ad9b7c73f31363de066290d057a"
    assert audit["current_tasgnss"]["status_short"] == []
    assert audit["selected_current_backend"]["version"] == "0.2.7"
    assert audit["selected_current_backend"]["source_status_short"] == []
    assert audit["classification"].startswith("TASGNSS is a later refactor")


def test_paper_state_has_exact_current_active_clock_embedding() -> None:
    with np.load(IBIZA_NPZ, allow_pickle=False) as data:
        assert np.array_equal(
            data["clock_state_index_by_constellation_code"], np.array([0, 3, 6, 5, 4])
        )
        assert set(np.unique(data["system_clock_index"])) <= {3, 4, 5, 6}
    current_clock_order = ("G", "C", "E", "R", "J", "I", "1")
    assert tuple(current_clock_order.index(name) + 3 for name in ("G", "C", "E", "R")) == (
        3,
        4,
        5,
        6,
    )


def test_runner_contains_no_training_or_neural_deserialization_calls() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                calls.append(ast.unparse(node.func))
            except Exception:
                pass
    assert not any(name.endswith(".backward") for name in calls)
    assert not any("optimizer" in name.lower() for name in calls)
    assert not any(name in {"torch.load", "torch.jit.load"} for name in calls)
    source = RUNNER.read_text(encoding="utf-8")
    assert "enable_torch=False" in source
    assert "w=1" in source
    assert "b=None" in source


def test_ground_truth_open_occurs_after_position_serialization() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    save_offset = source.index("_save_npz(positions_path, arrays)")
    reference_offset = source.index("args.reference.read_text")
    assert save_offset < reference_offset
    solver_call = source.index("tas.wls_pnt_pos(")
    assert solver_call < reference_offset


def test_metric_implementation_reconciles_frozen_paper_ols() -> None:
    reference = trusted_reference_ecef(json.loads(REFERENCE.read_text(encoding="utf-8")))
    with np.load(IBIZA_NPZ, allow_pickle=False) as data:
        enu = enu_errors(data["epoch_ols_initial_state"][:, :3], reference)
    summary = accuracy_summary(enu, total_epochs=enu.shape[0])
    assert summary["solved_epochs"] == 2856
    assert summary["e2d_rms_m"] == pytest.approx(8.373581796576715, abs=1e-12)
    assert summary["e3d_rms_m"] == pytest.approx(16.503387033950712, abs=1e-11)


def test_historical_neutral_reference_reconciles_without_a_model() -> None:
    reference = trusted_reference_ecef(json.loads(REFERENCE.read_text(encoding="utf-8")))
    rows = historical_neutral_rows(HISTORICAL)
    assert rows["solved"].all()
    enu = enu_errors(rows["state"][:, :3], reference)
    summary = accuracy_summary(enu, total_epochs=enu.shape[0])
    assert summary["e3d_rms_m"] == pytest.approx(35.740774945422885, abs=1e-11)


def test_paired_delta_sign_and_fractions() -> None:
    reference = np.array([[3.0, 4.0, 2.0], [0.0, 2.0, -3.0], [0.0, 1.0, 1.0]])
    candidate = np.array([[0.0, 4.0, 1.0], [0.0, 2.0, -3.0], [0.0, 2.0, 4.0]])
    rows = {row["metric"]: row for row in paired_delta_summary(candidate, reference)}
    assert rows["e2d"]["fraction_improved"] == pytest.approx(1 / 3)
    assert rows["e2d"]["fraction_worsened"] == pytest.approx(1 / 3)
    assert rows["e2d"]["fraction_effectively_equal"] == pytest.approx(1 / 3)
    assert rows["abs_up"]["mean_delta_m"] == pytest.approx(2 / 3)


@pytest.mark.skipif(not (OUTPUT / "tasgnss_comparison_manifest.json").is_file(), reason="Phase-B artifacts absent")
def test_completed_run_records_controls_and_repository_immutability() -> None:
    manifest = json.loads((OUTPUT / "tasgnss_comparison_manifest.json").read_text())
    controls = manifest["controls"]
    assert manifest["phase_a"]["status"] == "stopped_at_compatibility_gate"
    assert manifest["phase_b"]["status"] == "completed"
    assert controls["neural_checkpoint_loaded"] is False
    assert controls["learned_bias_applied"] is False
    assert controls["learned_weights_applied"] is False
    assert controls["uniform_weights"] is True
    assert controls["zero_bias"] is True
    assert controls["ground_truth_passed_to_solver"] is False
    assert controls["training_run"] is False
    assert controls["optimizer_created"] is False
    assert controls["backward_called"] is False
    assert controls["input_hashes_unchanged"] is True
    assert controls["external_repositories_unchanged"] is True
    assert controls["uniform_global_weight_scaling_invariant"] is True
    assert controls["numpy_torch_neutral_consistency"] is True


@pytest.mark.skipif(not (OUTPUT / "current_stack_epoch_results.npz").is_file(), reason="Phase-B artifacts absent")
def test_completed_epoch_metrics_reconcile_with_manifest() -> None:
    manifest = json.loads((OUTPUT / "tasgnss_comparison_manifest.json").read_text())
    with np.load(OUTPUT / "current_stack_epoch_results.npz", allow_pickle=False) as data:
        mask = data["solved"].astype(bool)
        summary = accuracy_summary(data["enu_error_m"][mask], total_epochs=mask.size)
        assert data["iteration_count"].shape == (mask.size,)
        assert np.all(data["iteration_count"] == -1)  # upstream does not expose it
    recorded = next(
        row
        for row in manifest["accuracy"]
        if row["solution"] == "current_tasgnss_neutral_all_raw_epochs"
    )
    for key, value in summary.items():
        assert recorded[key] == pytest.approx(value, abs=1e-12)


@pytest.mark.skipif(not (OUTPUT / "tasgnss_comparison_manifest.json").is_file(), reason="Phase-B artifacts absent")
def test_phase_a_row_mapping_is_not_claimed_for_end_to_end_baseline() -> None:
    manifest = json.loads((OUTPUT / "tasgnss_comparison_manifest.json").read_text())
    support = manifest["phase_b"]["support_comparison_with_frozen_paper_epochs"]
    assert manifest["strict_solver_only_comparison_completed"] is False
    assert support["accepted_epochs"] == 2856
    assert support["exact_same_order_support_epochs"] <= 2856


@pytest.mark.skipif(not (OUTPUT / "tasgnss_comparison_manifest.json").is_file(), reason="Phase-B artifacts absent")
def test_accuracy_table_separates_common_and_all_raw_denominators() -> None:
    manifest = json.loads((OUTPUT / "tasgnss_comparison_manifest.json").read_text())
    rows = {row["solution"]: row for row in manifest["accuracy"]}
    assert rows["current_tasgnss_neutral_common_frozen_epochs"]["total_epochs"] == 2856
    assert rows["current_tasgnss_neutral_all_raw_epochs"]["total_epochs"] == 2880
    rejected = manifest["phase_b"]["paper_rejected_epoch_slice"]
    assert rejected["epochs"] == 24
    assert rejected["current_solved_epochs"] == 24
