import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from validation.paper_biasnet.check_exported_metrics import (
    aggregate_csv,
    compare_summary,
)
from validation.paper_biasnet.core import (
    BIAS_SIGN_CONVENTION,
    BIAS_UNIT,
    BiasNet,
    construct_features,
    subtract_predicted_bias,
)
from validation.paper_biasnet.held_out import (
    DATASET_SPECS,
    FROZEN_MODEL_MEAN,
    FROZEN_MODEL_STD,
    KLT3_FEATURE_MEAN,
    KLT3_FEATURE_POPULATION_STD,
    evaluate_epoch,
    historical_position_error,
    infer_biases,
    load_frozen_biasnet,
    normalize_with_frozen_klt3,
    prepare_dataset,
    repository_root,
    resolve_input_paths,
)


ROOT = repository_root()
ARTIFACT_DIR = ROOT / "validation/paper_biasnet"
CHECKPOINT = ROOT / "checkpoints/paper_biasnet/biasnet_3d.pth"
LOCAL_KLT_OBSERVATION = Path(
    "/tmp/gnss-weightnet-repro/extracted/data/0610_KLT/COM38_210610_025603.obs"
)


def test_released_biasnet_architecture_and_unbounded_linear_output() -> None:
    model = BiasNet()
    model.double()
    linears = [layer for layer in model.seq if isinstance(layer, nn.Linear)]
    relus = [layer for layer in model.seq if isinstance(layer, nn.ReLU)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (3, 64),
        (64, 128),
        (128, 1),
    ]
    assert len(relus) == 2
    assert isinstance(model.seq[-1], nn.Linear)
    output = model(torch.zeros((5, 3), dtype=torch.float64))
    assert output.shape == (5, 1)
    assert torch.all(torch.isfinite(output))


def test_bias_sign_convention_and_units_are_metres() -> None:
    pseudorange = torch.tensor([20_000_000.0, 21_000_000.0], dtype=torch.float64)
    bias = torch.tensor([3.5, -2.0], dtype=torch.float64)
    corrected = subtract_predicted_bias(pseudorange, bias)
    torch.testing.assert_close(
        corrected,
        torch.tensor([19_999_996.5, 21_000_002.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert BIAS_UNIT == "metre"
    assert BIAS_SIGN_CONVENTION.endswith("pseudorange_m - predicted_bias_m")


def test_features_have_no_ground_truth_route_and_preserve_rows() -> None:
    snr = np.asarray([31.0, 27.0, 19.0])
    elevation = np.asarray([0.4, 0.8, 1.1])
    residual = np.asarray([-2.0, 0.5, 3.0])
    first_gt = np.asarray([22.3, 114.2, 4.0])
    second_gt = first_gt + np.asarray([1.0, -1.0, 1000.0])
    _ = first_gt
    first = construct_features(snr, elevation, residual)
    _ = second_gt
    second = construct_features(snr, elevation, residual)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 0], snr)
    np.testing.assert_array_equal(first[:, 1], elevation)
    np.testing.assert_array_equal(first[:, 2], residual)


def test_gradient_and_full_dataset_optimizer_smoke_artifacts() -> None:
    sanity = json.loads((ARTIFACT_DIR / "gradient_sanity.json").read_text())
    smoke = json.loads((ARTIFACT_DIR / "smoke_metrics.json").read_text())
    assert sanity["status"] == "passed"
    assert sanity["predicted_bias_all_finite"] is True
    assert sanity["predicted_bias_gradient_all_finite"] is True
    assert sanity["wls_state_all_finite"] is True
    assert sanity["every_expected_trainable_tensor_changed"] is True
    assert sanity["maximum_scaled_relative_error"] <= 1.0e-4
    assert smoke["status"] == "passed"
    assert smoke["dataset"]["epoch_count"] == 405
    assert smoke["dataset"]["measurement_count"] == 8857
    assert all(
        change > 0.0
        for change in smoke["first_parameter_max_abs_changes"].values()
    )


def test_real_epoch_forward_equivalence_artifact() -> None:
    comparison = json.loads((ARTIFACT_DIR / "forward_equivalence.json").read_text())
    assert comparison["status"] == "passed"
    assert comparison["bias_units"] == "metres"
    assert comparison["iterations"] == 2
    assert comparison["weighting"] == "identity W (equal weight)"
    assert all(
        value <= 1.0e-8
        for value in comparison["maximum_absolute_discrepancies"].values()
    )
    assert comparison["final_state_maximum_absolute_discrepancy"] <= 1.0e-8


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
    summary_path.write_text(
        json.dumps(
            {
                "per_epoch_csv": {"rows": 2},
                "tdl_b": {"mean_2d_error_m": 6.0, "mean_3d_error_m": 14.0},
                "equal_weight_ols_sanity_baseline": {
                    "mean_2d_error_m": 12.0,
                    "mean_3d_error_m": 22.0,
                },
            }
        )
    )
    metrics = aggregate_csv(csv_path)
    compare_summary(metrics, summary_path, 1.0e-12)
    assert metrics["rows"] == 2


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="local trained BiasNet absent")
def test_frozen_normalization_finite_inference_no_grad_and_immutability() -> None:
    model = load_frozen_biasnet(CHECKPOINT)
    np.testing.assert_array_equal(
        model.seq[0].mean.detach().numpy(),
        KLT3_FEATURE_MEAN.astype(np.float32).astype(np.float64),
    )
    np.testing.assert_array_equal(
        model.seq[0].std.detach().numpy(),
        KLT3_FEATURE_POPULATION_STD.astype(np.float32).astype(np.float64),
    )
    features = np.asarray(
        [[33.0, 0.8, 0.7], [20.0, 0.6, 2.9], [36.0, 0.5, -1.7]]
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    biases = infer_biases(model, features)
    assert model.training is False
    assert biases.shape == (3,)
    assert biases.requires_grad is False
    assert biases.grad_fn is None
    assert torch.all(torch.isfinite(biases))
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)


def test_test_dataset_statistics_cannot_change_frozen_normalization() -> None:
    features = np.asarray([[20.0, 0.4, -2.0], [40.0, 1.1, 3.0]])
    before_mean = FROZEN_MODEL_MEAN.copy()
    before_std = FROZEN_MODEL_STD.copy()
    normalized = normalize_with_frozen_klt3(features)
    _ = (features * 1000.0 + 9876.0).mean(axis=0)
    normalized_again = normalize_with_frozen_klt3(features)
    np.testing.assert_array_equal(FROZEN_MODEL_MEAN, before_mean)
    np.testing.assert_array_equal(FROZEN_MODEL_STD, before_std)
    np.testing.assert_array_equal(normalized, normalized_again)


@pytest.mark.skipif(
    not (CHECKPOINT.is_file() and LOCAL_KLT_OBSERVATION.is_file()),
    reason="local trained BiasNet or audited KLT archive absent",
)
def test_one_frozen_klt1_inference_epoch_row_alignment_and_subtraction() -> None:
    spec = DATASET_SPECS["KLT1"]
    prepared = prepare_dataset(spec, resolve_input_paths(spec))
    epoch = prepared.epochs[0]
    result = evaluate_epoch(load_frozen_biasnet(CHECKPOINT), epoch)
    assert epoch.epoch_time == 1623296154.005
    assert result.predicted_bias_m.shape == epoch.corrected_pseudorange_m.shape
    assert result.predicted_bias_m.shape[0] == epoch.satellite_ids.shape[0]
    np.testing.assert_allclose(
        result.bias_corrected_pseudorange_m,
        epoch.corrected_pseudorange_m - result.predicted_bias_m,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.estimated_ecef_m,
        [-2417941.9884433025, 5385052.465354332, 2408558.9084076877],
        rtol=0.0,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        result.enu_error_m,
        [-21.69923415751893, 116.38689474952768, 360.86377489002626],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert result.error_2d_m == pytest.approx(118.392423881178, abs=1.0e-12)
    assert result.error_3d_m == pytest.approx(379.78866499718, abs=1.0e-12)
