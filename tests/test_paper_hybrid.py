import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from validation.paper_hybrid.check_exported_metrics import (
    aggregate_csv,
    compare_summary,
)
from validation.paper_hybrid.core import (
    BIAS_SIGN_CONVENTION,
    BIAS_UNIT,
    OUTPUT_ORDER,
    RAW_COLUMN_TO_OUTPUT,
    WEIGHT_UNIT,
    HybridShareNet,
    construct_features,
    solve_paper_hybrid_position,
)
from validation.paper_hybrid.held_out import (
    DATASET_SPECS,
    FROZEN_MODEL_MEAN,
    FROZEN_MODEL_STD,
    evaluate_epoch,
    infer_hybrid,
    load_frozen_hybrid,
    normalize_with_frozen_klt3,
    prepare_dataset,
    resolve_input_paths,
)
from validation.paper_weightnet.held_out import historical_position_error


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "validation/paper_hybrid"
CHECKPOINT = ROOT / "checkpoints/paper_hybrid/hybrid_share_3d.pth"
TRAINING = ARTIFACT_DIR / "training_metrics.json"
LOCAL_KLT_OBSERVATION = Path(
    "/tmp/gnss-weightnet-repro/extracted/data/0610_KLT/COM38_210610_025603.obs"
)


def test_released_hybrid_architecture_is_shared_and_exact() -> None:
    model = HybridShareNet().double()
    linears = [layer for layer in model.seq if isinstance(layer, nn.Linear)]
    relus = [layer for layer in model.seq if isinstance(layer, nn.ReLU)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (3, 64), (64, 128), (128, 64), (64, 2)
    ]
    assert len(relus) == 3
    assert not hasattr(model, "weightNet")
    assert not hasattr(model, "biasNet")
    assert OUTPUT_ORDER == ("weight", "bias")
    assert RAW_COLUMN_TO_OUTPUT == {0: "weight", 1: "bias"}


def test_output_order_transformations_ranges_and_units() -> None:
    raw = torch.tensor(
        [[0.0, 2.0], [2.0, -3.0], [-2.0, 0.0]], dtype=torch.float64
    )
    weight, bias = HybridShareNet.transform_raw(raw)
    torch.testing.assert_close(weight, torch.sigmoid(raw[:, 0]))
    torch.testing.assert_close(
        bias, torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
    )
    assert torch.all((weight > 0.0) & (weight < 1.0))
    assert torch.all(bias >= 0.0)
    assert BIAS_UNIT == "metre"
    assert WEIGHT_UNIT == "dimensionless relative WLS coefficient"


def test_positive_bias_is_subtracted_in_hybrid_solver() -> None:
    # A deterministic physical sign check uses a valid multi-constellation
    # geometry and compares the solution wrapper's explicit corrected vector.
    positions = np.asarray(
        [[20e6, 0, 0], [0, 20e6, 0], [0, 0, 20e6], [-20e6, -20e6, -20e6]],
        dtype=np.float64,
    )
    state = np.asarray([1e6, 2e6, 3e6, 0, 0, 0, 0], dtype=np.float64)
    ranges = np.linalg.norm(positions - state[:3], axis=1)
    solution = solve_paper_hybrid_position(
        positions,
        np.zeros(4),
        ranges + np.asarray([5.0, 6.0, 7.0, 8.0]),
        np.full(4, 3),
        np.asarray([0.2, 0.4, 0.6, 0.8]),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        state,
        maximum_iterations=1,
    )
    np.testing.assert_allclose(
        solution.bias_corrected_pseudorange_m.numpy(),
        ranges + np.asarray([4.0, 4.0, 4.0, 4.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert BIAS_SIGN_CONVENTION.endswith("pseudorange_m - predicted_bias_m")


def test_features_preserve_rows_and_have_no_ground_truth_route() -> None:
    snr = np.asarray([31.0, 27.0, 19.0])
    elevation = np.asarray([0.4, 0.8, 1.1])
    residual = np.asarray([-2.0, 0.5, 3.0])
    first = construct_features(snr, elevation, residual)
    _unrelated_ground_truth = np.asarray([22.3, 114.2, 4.0])
    second = construct_features(snr, elevation, residual)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.column_stack((snr, elevation, residual)))


def test_shared_klt3_normalization_and_no_test_refit() -> None:
    manifest = json.loads(
        (ROOT / "validation/paper_weightnet/klt3_feature_manifest.json").read_text()
    )
    np.testing.assert_array_equal(
        np.asarray(manifest["features"]["mean"], dtype=np.float32).astype(np.float64),
        FROZEN_MODEL_MEAN,
    )
    np.testing.assert_array_equal(
        np.asarray(manifest["features"]["population_std"], dtype=np.float32).astype(np.float64),
        FROZEN_MODEL_STD,
    )
    features = np.asarray([[20.0, 0.4, -2.0], [40.0, 1.1, 3.0]])
    first = normalize_with_frozen_klt3(features)
    _ = (features * 1000.0 + 50.0).mean(axis=0)
    second = normalize_with_frozen_klt3(features)
    np.testing.assert_array_equal(first, second)


def test_hybrid_gt_cardinality_alignment_has_no_duplicate_defect() -> None:
    audit = json.loads((ARTIFACT_DIR / "gt_alignment_audit.json").read_text())
    cardinality = audit["cardinality"]
    assert audit["status"] == "passed"
    assert cardinality["gnss_epochs_retained_by_strict_time_window"] == 405
    assert cardinality["hybrid_gt_entries_built"] == 405
    assert cardinality["unique_consumed_gt_list_indices"] == 405
    assert audit["source_proof"]["second_append_present"] is False
    assert audit["defect_result"]["corrected_target_mode_required"] is False
    assert [row["gnss_epoch_index"] for row in audit["selected_mappings"]] == [
        0, 1, 2, 10, 100, 200, 400
    ]
    for row in audit["selected_mappings"]:
        assert row["hybrid_gt_list_index_consumed"] == row["gnss_epoch_index"]
        assert abs(row["time_difference_gt_minus_gnss_s"]) <= 0.0041


def test_archived_controlled_forward_equivalence_and_row_alignment() -> None:
    result = json.loads((ARTIFACT_DIR / "forward_equivalence.json").read_text())
    assert result["status"] == "passed"
    assert result["output_order"] == ["weight", "bias"]
    assert result["epoch"]["satellites"] == [
        "G01", "G03", "G07", "G14", "G21", "G22", "G28", "G30",
        "R09", "E13", "E15", "E21", "E26", "E27", "C07", "C11", "C13",
    ]
    assert all(
        value <= 1.0e-8
        for value in result["maximum_absolute_discrepancies"].values()
    )
    assert all(
        value <= 1.0e-8
        for value in result["transformation_maximum_absolute_discrepancies"].values()
    )
    assert result["final_state_maximum_absolute_discrepancy"] <= 1.0e-8


def test_dual_gradient_paths_updates_and_finite_differences() -> None:
    result = json.loads((ARTIFACT_DIR / "gradient_sanity.json").read_text())
    assert result["status"] == "passed"
    assert result["all_outputs_finite"] is True
    assert result["all_wls_states_finite"] is True
    assert result["bias_output_gradient_l2_norm"] > 0.0
    assert result["weight_output_gradient_l2_norm"] > 0.0
    assert result["bias_path_shared_parameter_gradient_l2_norm"] > 0.0
    assert result["weight_path_shared_parameter_gradient_l2_norm"] > 0.0
    assert result["gradient_additivity"]["maximum_scaled_discrepancy"] <= 1.0e-10
    assert result["every_expected_trainable_tensor_changed"] is True
    assert result["maximum_finite_difference_scaled_relative_error"] <= 5.0e-3
    audit = result["conventional_backward_gradient_audit"]
    assert audit["every_trainable_parameter_has_gradient"] is True
    assert audit["every_trainable_parameter_gradient_finite"] is True


def test_training_configuration_and_curve_are_released_behavior() -> None:
    training = json.loads(TRAINING.read_text())
    configuration = training["configuration"]
    assert training["status"] == "passed"
    assert configuration["training_epochs"] == 100
    assert configuration["optimizer"] == "Adam"
    assert configuration["learning_rate"] == 0.01
    assert configuration["shuffle"] is False
    assert configuration["maximum_wls_iterations"] == 10
    assert configuration["wls_tolerance"] == 1.0e-4
    assert configuration["initialization"].startswith("from scratch")
    assert len(training["epochs"]) == 100
    assert all(row["duration_seconds"] > 0.0 for row in training["epochs"])
    assert training["checkpoint"]["all_tensors_finite_after_reload"] is True


def test_independent_csv_checker_and_enu_formulas(tmp_path: Path) -> None:
    import pymap3d as p3d

    ground_truth = np.asarray([22.3, 114.2, 10.0])
    estimated = np.asarray(p3d.enu2ecef(3.0, 4.0, 12.0, *ground_truth))
    _geodetic, enu, error_2d, error_3d = historical_position_error(
        estimated, ground_truth
    )
    np.testing.assert_allclose(enu, [3.0, 4.0, 12.0], rtol=0.0, atol=2.0e-9)
    assert error_2d == pytest.approx(5.0, abs=2.0e-9)
    assert error_3d == pytest.approx(13.0, abs=2.0e-9)
    csv_path = tmp_path / "epochs.csv"
    summary_path = tmp_path / "summary.json"
    rows = [
        {"error_2d_m": "5", "error_3d_m": "13", "ols_error_2d_m": "10", "ols_error_3d_m": "20"},
        {"error_2d_m": "7", "error_3d_m": "15", "ols_error_2d_m": "14", "ols_error_3d_m": "24"},
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(json.dumps({
        "per_epoch_csv": {"rows": 2},
        "tdl_bw": {"mean_2d_error_m": 6.0, "mean_3d_error_m": 14.0},
        "equal_weight_ols_sanity_baseline": {"mean_2d_error_m": 12.0, "mean_3d_error_m": 22.0},
    }))
    metrics = aggregate_csv(csv_path)
    compare_summary(metrics, summary_path, 1.0e-12)
    assert metrics["rows"] == 2
    assert metrics["error_2d_diagnostics"]["median_m"] == 6.0


@pytest.mark.skipif(
    not (CHECKPOINT.is_file() and TRAINING.is_file()),
    reason="local trained hybrid checkpoint absent",
)
def test_frozen_inference_no_grad_finite_aligned_and_immutable() -> None:
    model = load_frozen_hybrid(CHECKPOINT, TRAINING)
    features = np.asarray(
        [[33.0, 0.8, 0.7], [20.0, 0.6, 2.9], [36.0, 0.5, -1.7]]
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    weight, bias = infer_hybrid(model, features)
    assert weight.shape == bias.shape == (3,)
    assert not weight.requires_grad and not bias.requires_grad
    assert torch.all(torch.isfinite(weight)) and torch.all(torch.isfinite(bias))
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not (CHECKPOINT.is_file() and TRAINING.is_file() and LOCAL_KLT_OBSERVATION.is_file()),
    reason="local hybrid checkpoint or audited KLT archive absent",
)
def test_one_fixed_klt1_epoch_rows_and_outputs_are_reproducible() -> None:
    spec = DATASET_SPECS["KLT1"]
    prepared = prepare_dataset(spec, resolve_input_paths(spec))
    epoch = prepared.epochs[0]
    result = evaluate_epoch(load_frozen_hybrid(CHECKPOINT, TRAINING), epoch)
    assert epoch.epoch_time == 1623296154.005
    assert result.weights.shape == result.predicted_bias_m.shape == (17,)
    np.testing.assert_allclose(
        result.bias_corrected_pseudorange_m,
        epoch.corrected_pseudorange_m - result.predicted_bias_m,
        rtol=0.0,
        atol=0.0,
    )
    assert result.normal_matrix_rank == 7
    assert result.historical_wls_status == "converged"
    np.testing.assert_allclose(
        result.estimated_ecef_m,
        [-2417838.7673911173, 5384760.699278383, 2408305.2699847673],
        rtol=0.0,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        result.enu_error_m,
        [3.648848873998122, -1.036504836102979, -20.82156434608651],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert result.error_2d_m == pytest.approx(3.79320977281012, abs=1.0e-12)
    assert result.error_3d_m == pytest.approx(21.1642619100871, abs=1.0e-12)
    summary = json.loads((ARTIFACT_DIR / "held_out_evaluation_summary.json").read_text())
    representative = summary["datasets"]["KLT1"]["representative_epochs"][0]
    assert representative["epoch_index"] == 0
    assert [row["PRN"] for row in representative["rows"]] == epoch.satellite_ids.tolist()
