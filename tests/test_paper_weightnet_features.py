import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from validation.paper_weightnet.core import WeightNet, construct_features


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPOSITORY_ROOT / "validation/paper_weightnet"


def test_released_weightnet_architecture_and_output_range() -> None:
    model = WeightNet()
    model.double()
    linear_layers = [layer for layer in model.seq if isinstance(layer, nn.Linear)]
    sigmoid_layers = [layer for layer in model.seq if isinstance(layer, nn.Sigmoid)]

    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (3, 64),
        (64, 128),
        (128, 64),
        (64, 1),
    ]
    assert len(sigmoid_layers) == 4
    output = model(torch.zeros((5, 3), dtype=torch.float64))
    assert output.shape == (5, 1)
    assert torch.all(torch.isfinite(output))
    assert torch.all((output >= 0.0) & (output <= 10.0))


def test_features_are_observation_only_and_ground_truth_independent() -> None:
    snr = np.array([31.0, 27.0, 19.0])
    elevation = np.array([0.4, 0.8, 1.1])
    residual = np.array([-2.0, 0.5, 3.0])
    first_ground_truth = np.array([22.3, 114.2, 4.0])
    second_ground_truth = first_ground_truth + np.array([1.0, -1.0, 1000.0])

    first = construct_features(snr, elevation, residual)
    # Ground truth deliberately has no route into construct_features.
    _ = first_ground_truth
    second = construct_features(snr, elevation, residual)
    _ = second_ground_truth

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 0], snr)
    np.testing.assert_array_equal(first[:, 1], elevation)
    np.testing.assert_array_equal(first[:, 2], residual)


def test_machine_readable_klt3_cardinality_and_feature_validation() -> None:
    manifest = json.loads((ARTIFACT_DIR / "klt3_feature_manifest.json").read_text())
    discrepancy = manifest["cardinality_discrepancy"]

    assert manifest["status"] == "passed"
    assert manifest["valid_epoch_count"] == 405
    assert manifest["satellite_measurement_count"] == 8857
    assert discrepancy["published_metadata"] == {
        "epoch_count": 404,
        "satellite_measurement_count": 8857,
    }
    assert discrepancy["released_code_reproduction"] == {
        "epoch_count": 405,
        "satellite_measurement_count": 8857,
    }
    assert discrepancy["drop_first_epoch"]["satellite_measurement_count"] == 8836
    assert discrepancy["drop_last_epoch"]["satellite_measurement_count"] == 8835
    assert discrepancy["undocumented_epoch_removal_applied"] is False
    assert manifest["configured_interval"]["first_retained_timestamp_gpst_like"] == 1623297151.006
    assert manifest["configured_interval"]["last_retained_timestamp_gpst_like"] == 1623297555.006
    assert manifest["features"]["all_finite"] is True
    assert manifest["features"]["ground_truth_independence_bitwise"] is True
    assert len(manifest["satellite_counts_per_epoch"]) == 405
    assert sum(manifest["satellite_counts_per_epoch"]) == 8857


def test_full_dataset_smoke_gradients_and_optimizer_update_passed() -> None:
    smoke = json.loads((ARTIFACT_DIR / "smoke_metrics.json").read_text())
    audit = smoke["first_gradient_audit"]

    assert smoke["status"] == "passed"
    assert smoke["dataset"]["epoch_count"] == 405
    assert smoke["dataset"]["measurement_count"] == 8857
    assert audit["every_trainable_parameter_has_gradient"] is True
    assert audit["every_trainable_parameter_gradient_finite"] is True
    assert audit["at_least_one_nonzero_gradient"] is True
    assert audit["global_l2_norm"] > 0.0
    assert all(
        record["present"] and record["finite"] and record["nonzero"]
        for record in audit["per_parameter"].values()
    )
    assert any(
        change > 0.0
        for change in smoke["first_parameter_max_abs_changes"].values()
    )

