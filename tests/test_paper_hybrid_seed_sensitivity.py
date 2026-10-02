import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from validation.paper_hybrid.train_paper_hybrid import DEFAULT_SEED
from validation.paper_hybrid_seed_sensitivity import experiment
from validation.paper_hybrid_seed_sensitivity.experiment import (
    build_initial_snapshot,
    create_model,
    gradient_diagnostics,
    instrumented_forward,
    normalization_identity,
    parameter_hash,
    scientific_configuration,
    training_data_identity,
)
from validation.paper_hybrid_seed_sensitivity.run_seed import parse_args
from validation.paper_hybrid_seed_sensitivity.summarize_seeds import (
    BIAS_STATUS_DEFINITIONS,
    build_summary,
    classify_bias_status,
    load_results,
    markdown_table,
)


def small_dataset() -> dict[str, np.ndarray]:
    return {
        "features": np.asarray(
            [
                [18.0, 0.2, -4.0],
                [25.0, 0.5, -1.0],
                [31.0, 0.8, 0.5],
                [38.0, 1.0, 2.0],
                [44.0, 1.2, 5.0],
            ],
            dtype=np.float64,
        ),
        "ground_truth_geodetic_deg_m": np.asarray(
            [[22.30, 114.20, 10.0], [22.31, 114.21, 11.0]], dtype=np.float64
        ),
        "epoch_offsets": np.asarray([0, 2, 5], dtype=np.int64),
        "satellite_ids": np.asarray(["G01", "G02", "E01", "C01", "R01"]),
        "split_epoch_indices": np.asarray([10, 11], dtype=np.int64),
        "epoch_times_gpst_like": np.asarray([100.0, 101.0], dtype=np.float64),
        "corrected_pseudorange_m": np.arange(5, dtype=np.float64),
    }


def manifest_for(dataset: dict[str, np.ndarray]) -> dict[str, object]:
    return {"cache": {"sha256": "synthetic-cache-sha256"}}


def test_seed_argument_and_seed_call_precede_model_construction(monkeypatch) -> None:
    assert parse_args(["--seed", "7"]).seed == 7
    assert DEFAULT_SEED == 20_260_929
    events: list[tuple[str, object]] = []
    sentinel = object()

    def record_seed(seed: int) -> None:
        events.append(("seed", seed))

    def record_construction(mean, std, *, device):
        events.append(("construct", device))
        return sentinel

    monkeypatch.setattr(experiment, "set_seed", record_seed)
    monkeypatch.setattr(
        experiment, "instantiate_released_hybrid", record_construction
    )
    result = experiment.create_model(small_dataset(), 7, "cpu")
    assert result is sentinel
    assert events == [("seed", 7), ("construct", "cpu")]


def test_same_seed_same_initialization_and_different_seed_differs() -> None:
    dataset = small_dataset()
    first = create_model(dataset, 3, "cpu")
    repeated = create_model(dataset, 3, "cpu")
    different = create_model(dataset, 4, "cpu")
    assert parameter_hash(first) == parameter_hash(repeated)
    assert parameter_hash(first) != parameter_hash(different)
    features = torch.as_tensor(dataset["features"], dtype=torch.float32)
    with torch.no_grad():
        first_outputs = first(features)
        repeated_outputs = repeated(features)
    for left, right in zip(first_outputs, repeated_outputs, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_instrumentation_preserves_forward_outputs_gradients_and_parameters() -> None:
    dataset = small_dataset()
    model = create_model(dataset, 6, "cpu")
    features = torch.as_tensor(dataset["features"], dtype=torch.float32)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    expected_weight, expected_bias = model(features)
    expected_loss = expected_weight.square().sum() + expected_bias.square().sum()
    expected_loss.backward()
    expected_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    for parameter in model.parameters():
        parameter.grad = None

    raw, weight, bias = instrumented_forward(model, features)
    actual_loss = weight.square().sum() + bias.square().sum()
    actual_loss.backward()

    torch.testing.assert_close(weight, expected_weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(bias, expected_bias, rtol=0.0, atol=0.0)
    torch.testing.assert_close(bias, torch.relu(raw[:, 1]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        weight, torch.clamp(torch.sigmoid(raw[:, 0]), 0.0, 1.0), rtol=0.0, atol=0.0
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        torch.testing.assert_close(
            parameter.grad, expected_gradients[name], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)
    audit = gradient_diagnostics(model)
    assert audit["all_trainable_gradients_present"] is True
    assert audit["all_trainable_gradients_finite"] is True


def test_seed_does_not_change_data_normalization_or_configuration() -> None:
    dataset = small_dataset()
    untouched = {name: values.copy() for name, values in dataset.items()}
    manifest = manifest_for(dataset)
    data_before = training_data_identity(dataset, manifest)
    normalization_before = normalization_identity(dataset)

    first_model, first_optimizer, first = build_initial_snapshot(
        dataset, 0, torch.device("cpu")
    )
    second_model, second_optimizer, second = build_initial_snapshot(
        dataset, 1, torch.device("cpu")
    )

    assert first["initial_parameter_sha256"] != second["initial_parameter_sha256"]
    assert first["configuration_sha256"] == second["configuration_sha256"]
    assert first["normalization_sha256"] == second["normalization_sha256"]
    assert scientific_configuration(first_model, first_optimizer) == (
        scientific_configuration(second_model, second_optimizer)
    )
    assert training_data_identity(dataset, manifest) == data_before
    assert normalization_identity(dataset) == normalization_before
    for name in dataset:
        np.testing.assert_array_equal(dataset[name], untouched[name])


def mock_result(
    seed: int, initial: float, final: float, loss: float
) -> dict[str, object]:
    return {
        "status": "completed",
        "seed": seed,
        "executed_training_epochs": 100,
        "finite": {"all_finite": True},
        "configuration_sha256": "same-config",
        "normalization_sha256": "same-normalization",
        "training_data_identity_sha256": "same-data",
        "initial_positive_bias_preactivation_fraction": initial,
        "final_positive_bias_fraction": final,
        "training_loss": {"final_post_update_sum_3d_m": loss},
        "held_out": {
            "KLT1": {
                "mean_2d_error_m": loss + 1.0,
                "mean_3d_error_m": loss + 2.0,
            },
            "KLT2": {
                "mean_2d_error_m": loss + 3.0,
                "mean_3d_error_m": loss + 4.0,
            },
        },
    }


def test_summary_parsing_and_categories_on_mock_results(tmp_path: Path) -> None:
    results = [
        mock_result(0, 0.0, 0.0, 10.0),
        mock_result(1, 0.2, 0.0, 20.0),
        mock_result(2, 0.0, 0.3, 30.0),
        mock_result(3, 0.2, 0.3, 40.0),
    ]
    # Exercise the exact machine-readable representation used on disk.
    for result in results:
        path = tmp_path / f"seed_{result['seed']}.json"
        path.write_text(json.dumps(result))
    loaded = load_results(tmp_path, [0, 1, 2, 3])
    summary = build_summary(loaded)

    assert [row["seed"] for row in summary["rows"]] == [0, 1, 2, 3]
    assert summary["category_counts"] == {
        category: 1 for category in BIAS_STATUS_DEFINITIONS
    }
    assert summary["aggregate_statistics"][
        "final_training_loss_sum_3d_m"
    ] == pytest.approx(
        {
            "mean": 25.0,
            "median": 25.0,
            "std": np.std([10.0, 20.0, 30.0, 40.0]),
            "minimum": 10.0,
            "maximum": 40.0,
        }
    )
    table = markdown_table(summary)
    assert "Rows are in numeric seed order" in table
    assert "| 3 |" in table
    assert classify_bias_status(0.0, 0.0) == "dead from initialization"
    assert classify_bias_status(0.1, 0.0) == "initially active, later dead"
    assert classify_bias_status(0.0, 0.1) == "initially dead, later active"
    assert classify_bias_status(0.1, 0.1) == "active"


def test_summary_rejects_changed_control_hash() -> None:
    first = mock_result(0, 0.0, 0.0, 10.0)
    second = copy.deepcopy(mock_result(1, 0.0, 0.0, 11.0))
    second["normalization_sha256"] = "changed"
    with pytest.raises(ValueError, match="normalization_sha256"):
        build_summary([first, second])
