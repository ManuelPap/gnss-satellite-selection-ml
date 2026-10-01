import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from validation.paper_weightnet.check_exported_metrics import (
    aggregate_csv,
    compare_summary,
)
from validation.paper_weightnet.core import construct_features
from validation.paper_weightnet.held_out import (
    CHECKPOINT_SHA256,
    DATASET_SPECS,
    FROZEN_MODEL_MEAN,
    FROZEN_MODEL_STD,
    KLT3_FEATURE_MEAN,
    KLT3_FEATURE_POPULATION_STD,
    evaluate_epoch,
    historical_position_error,
    infer_weights,
    load_frozen_weightnet,
    normalize_with_frozen_klt3,
    prepare_dataset,
    repository_root,
    resolve_input_paths,
    verify_checkpoint,
)


ROOT = repository_root()
CHECKPOINT = ROOT / "checkpoints/paper_weightnet/weightnet_3d.pth"
LOCAL_KLT_OBSERVATION = Path(
    "/tmp/gnss-weightnet-repro/extracted/data/0610_KLT/COM38_210610_025603.obs"
)


def test_checkpoint_hash_and_frozen_klt3_normalization() -> None:
    assert verify_checkpoint(CHECKPOINT) == CHECKPOINT_SHA256
    model = load_frozen_weightnet(CHECKPOINT)

    np.testing.assert_array_equal(
        model.seq[0].mean.detach().numpy(), FROZEN_MODEL_MEAN
    )
    np.testing.assert_array_equal(model.seq[0].std.detach().numpy(), FROZEN_MODEL_STD)
    np.testing.assert_array_equal(
        FROZEN_MODEL_MEAN, KLT3_FEATURE_MEAN.astype(np.float32).astype(np.float64)
    )
    np.testing.assert_array_equal(
        FROZEN_MODEL_STD,
        KLT3_FEATURE_POPULATION_STD.astype(np.float32).astype(np.float64),
    )


def test_test_dataset_statistics_cannot_change_inference_normalization() -> None:
    first = np.asarray([[20.0, 0.4, -2.0], [40.0, 1.1, 3.0]])
    unrelated_shifted_dataset = first * 1000.0 + 9876.0
    before_mean = FROZEN_MODEL_MEAN.copy()
    before_std = FROZEN_MODEL_STD.copy()

    normalized = normalize_with_frozen_klt3(first)
    _ = unrelated_shifted_dataset.mean(axis=0)
    _ = unrelated_shifted_dataset.std(axis=0)
    normalized_again = normalize_with_frozen_klt3(first)

    np.testing.assert_array_equal(FROZEN_MODEL_MEAN, before_mean)
    np.testing.assert_array_equal(FROZEN_MODEL_STD, before_std)
    np.testing.assert_array_equal(normalized, normalized_again)
    expected = (
        first.astype(np.float32).astype(np.float64) - FROZEN_MODEL_MEAN
    ) / FROZEN_MODEL_STD
    np.testing.assert_array_equal(normalized, expected)


def test_inference_is_no_grad_finite_aligned_and_parameter_immutable() -> None:
    model = load_frozen_weightnet(CHECKPOINT)
    features = np.asarray(
        [[33.0, 0.8, 0.7], [20.0, 0.6, 2.9], [36.0, 0.5, -1.7]],
        dtype=np.float64,
    )
    satellite_rows = ("G01", "G03", "G07")
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    weights = infer_weights(model, features)

    assert model.training is False
    assert weights.requires_grad is False
    assert weights.grad_fn is None
    assert tuple(zip(satellite_rows, weights.tolist(), strict=True))[0][0] == "G01"
    assert weights.shape == (len(satellite_rows),)
    assert torch.all(torch.isfinite(weights))
    assert torch.all((weights > 0.0) & (weights < 10.0))
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name], rtol=0.0, atol=0.0)


def test_enu_2d_3d_formulas() -> None:
    # Use a local ENU displacement to create an ECEF estimate, then run the
    # exact historical ECEF->geodetic->ENU evaluation path.
    import pymap3d as p3d

    ground_truth = np.asarray([22.3, 114.2, 10.0], dtype=np.float64)
    estimated_ecef = np.asarray(
        p3d.enu2ecef(3.0, 4.0, 12.0, *ground_truth), dtype=np.float64
    )
    _geodetic, enu, error_2d, error_3d = historical_position_error(
        estimated_ecef, ground_truth
    )

    np.testing.assert_allclose(enu, [3.0, 4.0, 12.0], rtol=0.0, atol=2.0e-9)
    assert error_2d == pytest.approx(5.0, abs=2.0e-9)
    assert error_3d == pytest.approx(13.0, abs=2.0e-9)


def test_ground_truth_has_no_route_into_feature_construction() -> None:
    snr = np.asarray([31.0, 22.0])
    elevation = np.asarray([0.5, 1.0])
    residual = np.asarray([-3.0, 4.0])
    first = construct_features(snr, elevation, residual)
    fictitious_ground_truth = np.asarray([90.0, -180.0, 1.0e9])
    _ = fictitious_ground_truth
    second = construct_features(snr, elevation, residual)
    np.testing.assert_array_equal(first, second)


def test_independent_csv_summary_checker(tmp_path: Path) -> None:
    csv_path = tmp_path / "epochs.csv"
    summary_path = tmp_path / "summary.json"
    rows = [
        {
            "error_2d_m": "5.0",
            "error_3d_m": "13.0",
            "ols_error_2d_m": "10.0",
            "ols_error_3d_m": "20.0",
        },
        {
            "error_2d_m": "7.0",
            "error_3d_m": "15.0",
            "ols_error_2d_m": "14.0",
            "ols_error_3d_m": "24.0",
        },
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(
        json.dumps(
            {
                "per_epoch_csv": {"rows": 2},
                "tdl_w": {"mean_2d_error_m": 6.0, "mean_3d_error_m": 14.0},
                "equal_weight_ols_sanity_baseline": {
                    "mean_2d_error_m": 12.0,
                    "mean_3d_error_m": 22.0,
                },
            }
        )
    )
    metrics = aggregate_csv(csv_path)
    compare_summary(metrics, summary_path, 1.0e-12)
    assert metrics == {
        "rows": 2,
        "mean_2d_error_m": 6.0,
        "mean_3d_error_m": 14.0,
        "ols_mean_2d_error_m": 12.0,
        "ols_mean_3d_error_m": 22.0,
    }


@pytest.mark.skipif(
    not LOCAL_KLT_OBSERVATION.is_file(), reason="local audited public KLT archive absent"
)
def test_frozen_klt1_epoch_zero_is_numerically_reproducible() -> None:
    spec = DATASET_SPECS["KLT1"]
    inputs = resolve_input_paths(spec)
    prepared = prepare_dataset(spec, inputs)
    epoch = prepared.epochs[0]
    result = evaluate_epoch(load_frozen_weightnet(CHECKPOINT), epoch)

    assert epoch.epoch_time == 1623296154.005
    assert tuple(epoch.satellite_ids) == (
        "G01",
        "G03",
        "G07",
        "G14",
        "G21",
        "G22",
        "G28",
        "G30",
        "R09",
        "E13",
        "E15",
        "E21",
        "E26",
        "E27",
        "C07",
        "C11",
        "C13",
    )
    assert len(result.solution.iterations) == 3
    assert result.historical_wls_status == "converged"
    np.testing.assert_allclose(
        result.estimated_ecef_m,
        [-2417845.713594851, 5384767.019720683, 2408313.3513801843],
        rtol=0.0,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        result.enu_error_m,
        [7.396617393474187, 3.167048847108238, -9.78566580413113],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert result.error_2d_m == pytest.approx(8.04612622728568, abs=1.0e-12)
    assert result.error_3d_m == pytest.approx(12.6688358776786, abs=1.0e-12)
