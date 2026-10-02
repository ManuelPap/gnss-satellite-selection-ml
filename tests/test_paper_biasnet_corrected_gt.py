import json
from pathlib import Path

import numpy as np
import pytest
import torch

from validation.paper_biasnet.held_out import infer_biases, load_frozen_biasnet
from validation.paper_biasnet.train_paper_biasnet import DEFAULT_SEED
from validation.paper_biasnet_corrected_gt.experiment import (
    HELD_OUT_DATASETS,
    TRAINING_DATASETS,
    corrected_target_indices,
    historical_target_indices,
    initialize_controlled_ab,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "validation/paper_biasnet_corrected_gt"
ALIGNMENT = ARTIFACT_DIR / "gt_alignment_audit.json"
SMOKE = ARTIFACT_DIR / "smoke_metrics.json"
TRAINING = ARTIFACT_DIR / "training_metrics.json"
HELD_OUT = ARTIFACT_DIR / "held_out_evaluation_summary.json"
CHECKPOINT = ROOT / "checkpoints/paper_biasnet_corrected_gt/biasnet_3d.pth"


def test_corrected_mapping_has_exactly_one_unique_target_per_epoch() -> None:
    corrected = corrected_target_indices(405)
    historical = historical_target_indices(405)
    np.testing.assert_array_equal(corrected, np.arange(405))
    np.testing.assert_array_equal(historical[:8], [0, 0, 1, 1, 2, 2, 3, 3])
    assert corrected.size == 405
    assert np.unique(corrected).size == 405
    assert corrected[-1] == 404
    assert historical[-1] == 202
    assert not np.array_equal(corrected, historical)


def test_raw_timestamp_audit_proves_alignment_and_no_growing_error() -> None:
    audit = json.loads(ALIGNMENT.read_text())
    contract = audit["correction_contract"]
    statistics = audit["corrected_alignment_statistics_seconds"]
    historical_statistics = audit["historical_alignment_statistics_seconds"]
    growth = audit["offset_growth_comparison"]
    assert audit["status"] == "passed"
    assert audit["released_defect_proof"]["historical_gt_list_length"] == 810
    assert contract["training_epoch_count"] == 405
    assert contract["training_gt_target_count"] == 405
    assert contract["unique_target_index_count"] == 405
    assert contract["final_target_source_epoch_index"] == 404
    assert statistics["maximum_absolute_mismatch"] <= statistics[
        "expected_maximum_tolerance"
    ]
    assert statistics["median_absolute_mismatch"] <= 0.0041
    assert statistics["p95_absolute_mismatch"] <= 0.0041
    assert historical_statistics["maximum_absolute_mismatch"] > 200.0
    assert historical_statistics["median_absolute_mismatch"] > 100.0
    assert historical_statistics["p95_absolute_mismatch"] > 190.0
    assert growth["historical_last_absolute_offset_s"] > 200.0
    assert growth["corrected_last_absolute_offset_s"] <= 0.0041
    assert growth["corrected_error_grows_with_epoch_index"] is False
    for section in ("first_10", "middle_10", "last_10"):
        rows = audit["alignment_samples"][section]
        assert len(rows) == 10
        assert all(row["absolute_time_difference_after_alignment_s"] <= 0.0041 for row in rows)


def test_ab_seed_initialization_is_exactly_identical() -> None:
    mean = np.asarray([29.084904595235407, 0.8471899979658503, -1.7873233713548635e-07])
    std = np.asarray([5.89184205821295, 0.28798909935383876, 4.3018036109624145])
    _model, _optimizer, proof = initialize_controlled_ab(
        mean, std, seed=DEFAULT_SEED
    )
    assert proof["same_theta_0"] is True
    assert proof["historical_initial_model_sha256"] == proof[
        "corrected_initial_model_sha256"
    ]
    assert proof["historical_optimizer"] == proof["corrected_optimizer"]
    assert proof["corrected_optimizer"]["learning_rate"] == 0.01


def test_only_gt_target_tensor_differs_in_controlled_setup() -> None:
    corrected = json.loads(SMOKE.read_text())
    historical = json.loads(
        (ROOT / "validation/paper_biasnet/training_metrics.json").read_text()
    )
    control = corrected["ab_control"]
    assert control["same_theta_0"] is True
    assert control["same_optimizer_configuration"] is True
    assert control["same_feature_array"] is True
    assert control["same_normalization"] is True
    assert control["target_tensors_differ"] is True
    assert control["only_gt_target_mapping_differs"] is True
    assert corrected["dataset"]["feature_cache_sha256"] == historical["dataset"][
        "feature_cache_sha256"
    ]
    np.testing.assert_array_equal(
        corrected["dataset"]["feature_mean_float64"],
        historical["dataset"]["feature_mean_float64"],
    )
    np.testing.assert_array_equal(
        corrected["dataset"]["feature_population_std_float64"],
        historical["dataset"]["feature_population_std_float64"],
    )
    assert corrected["configuration"]["optimizer"] == historical["configuration"][
        "optimizer"
    ]
    assert corrected["configuration"]["learning_rate"] == historical[
        "configuration"
    ]["learning_rate"]
    assert corrected["configuration"]["loss"] == historical["configuration"]["loss"]


def test_no_held_out_data_leaks_into_training() -> None:
    smoke = json.loads(SMOKE.read_text())
    control = smoke["ab_control"]
    assert TRAINING_DATASETS == ("KLT3",)
    assert HELD_OUT_DATASETS == ("KLT1", "KLT2")
    assert control["training_datasets"] == ["KLT3"]
    assert control["held_out_datasets"] == ["KLT1", "KLT2"]
    assert control["held_out_data_used_for_training"] is False


def test_corrected_gradient_smoke_is_finite_and_updates_parameters() -> None:
    smoke = json.loads(SMOKE.read_text())
    audit = smoke["first_gradient_audit"]
    assert smoke["status"] == "passed"
    assert smoke["dataset"]["epoch_count"] == 405
    assert smoke["dataset"]["gt_target_count"] == 405
    assert smoke["dataset"]["measurement_count"] == 8857
    assert audit["every_trainable_parameter_has_gradient"] is True
    assert audit["every_trainable_parameter_gradient_finite"] is True
    assert all(
        change > 0.0 for change in smoke["first_parameter_max_abs_changes"].values()
    )


def test_full_training_keeps_frozen_configuration_and_records_500_epochs() -> None:
    training = json.loads(TRAINING.read_text())
    configuration = training["configuration"]
    assert training["status"] == "passed"
    assert configuration["training_epochs"] == 500
    assert configuration["optimizer"] == "Adam"
    assert configuration["learning_rate"] == 0.01
    assert configuration["shuffle"] is False
    assert configuration["maximum_wls_iterations"] == 10
    assert configuration["wls_tolerance"] == 1.0e-4
    assert configuration["bias_output"] == "linear, unbounded, metres"
    assert len(training["epochs"]) == 500
    assert [row["epoch"] for row in training["epochs"]] == list(range(1, 501))
    assert all(row["duration_seconds"] > 0.0 for row in training["epochs"])
    assert all(np.isfinite(row["loss_sum_3d_m"]) for row in training["epochs"])
    assert all(np.isfinite(row["gradient_l2_norm"]) for row in training["epochs"])
    assert training["checkpoint"]["all_tensors_finite_after_reload"] is True


@pytest.mark.skipif(
    not (CHECKPOINT.is_file() and TRAINING.is_file()),
    reason="local corrected 500-epoch checkpoint absent",
)
def test_corrected_checkpoint_is_finite_and_inference_is_immutable() -> None:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert all(torch.all(torch.isfinite(value)) for value in state.values())
    model = load_frozen_biasnet(CHECKPOINT, TRAINING)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    features = np.asarray(
        [[33.0, 0.8, 0.7], [20.0, 0.6, 2.9], [36.0, 0.5, -1.7]]
    )
    output = infer_biases(model, features)
    assert output.requires_grad is False
    assert output.grad_fn is None
    assert torch.all(torch.isfinite(output))
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)


@pytest.mark.skipif(not HELD_OUT.is_file(), reason="held-out corrected summary absent")
def test_independent_held_out_metric_checker_agrees() -> None:
    summary = json.loads(HELD_OUT.read_text())
    assert summary["status"] == "passed"
    assert summary["held_out_data_used_for_training"] is False
    expected = {"KLT1": (203, 4676), "KLT2": (209, 4914)}
    for name, (epochs, measurements) in expected.items():
        dataset = summary["datasets"][name]
        assert dataset["cardinality"]["released_code_valid_evaluation_epochs"] == epochs
        assert dataset["cardinality"]["retained_satellite_measurements"] == measurements
        checker = dataset["independent_csv_checker"]
        assert checker["rows"] == epochs
        assert checker["agreed_with_evaluator_summary"] is True
