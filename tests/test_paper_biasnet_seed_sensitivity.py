import csv
import copy
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
from validation.paper_biasnet.core import subtract_predicted_bias
from validation.paper_biasnet.train_paper_biasnet import DEFAULT_SEED, load_dataset
from validation.paper_biasnet_corrected_gt.experiment import (
    corrected_target_indices,
    model_state_sha256,
    validate_training_dataset,
)
from validation.paper_biasnet_seed_sensitivity import experiment
from validation.paper_biasnet_seed_sensitivity.experiment import (
    build_initial_snapshot,
    create_model,
    gradient_diagnostics,
    hidden_activation_diagnostics,
    instrumented_forward,
    normalization_identity,
    parameter_change_diagnostics,
    parameter_hash,
    parameter_snapshot,
    scientific_configuration,
    training_data_identity,
)
from validation.paper_biasnet_seed_sensitivity.run_seed import parse_args
from validation.paper_biasnet_seed_sensitivity.summarize_seeds import (
    build_summary,
    load_results,
    markdown_table,
    row_from_result,
)


ROOT = Path(__file__).resolve().parents[1]


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


def manifest_for(_dataset: dict[str, np.ndarray]) -> dict[str, object]:
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
    monkeypatch.setattr(experiment, "instantiate_released_biasnet", record_construction)
    assert experiment.create_model(small_dataset(), 7, "cpu") is sentinel
    assert events == [("seed", 7), ("construct", "cpu")]


def test_same_seed_same_parameters_and_outputs_different_seed_differs() -> None:
    dataset = small_dataset()
    first = create_model(dataset, 3, "cpu")
    repeated = create_model(dataset, 3, "cpu")
    different = create_model(dataset, 4, "cpu")
    assert parameter_hash(first) == parameter_hash(repeated)
    assert parameter_hash(first) != parameter_hash(different)
    features = torch.as_tensor(dataset["features"], dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(
            first(features), repeated(features), rtol=0.0, atol=0.0
        )


def test_instrumentation_preserves_output_gradients_and_captures_post_relu() -> None:
    dataset = small_dataset()
    model = create_model(dataset, 6, "cpu")
    features = torch.as_tensor(dataset["features"], dtype=torch.float32)
    parameters_before = parameter_snapshot(model)

    expected_output = model(features)
    expected_output.square().sum().backward()
    expected_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    for parameter in model.parameters():
        parameter.grad = None

    hidden_1, hidden_2, output = instrumented_forward(model, features)
    output.square().sum().backward()
    torch.testing.assert_close(output, expected_output, rtol=0.0, atol=0.0)
    expected_hidden_1 = model.seq[2](model.seq[1](model.seq[0](features)))
    expected_hidden_2 = model.seq[4](model.seq[3](expected_hidden_1))
    torch.testing.assert_close(hidden_1, expected_hidden_1, rtol=0.0, atol=0.0)
    torch.testing.assert_close(hidden_2, expected_hidden_2, rtol=0.0, atol=0.0)
    assert torch.all(hidden_1 >= 0.0)
    assert torch.all(hidden_2 >= 0.0)
    audit = gradient_diagnostics(model)
    assert audit["all_trainable_gradients_present"] is True
    assert audit["all_trainable_gradients_finite"] is True
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter.grad, expected_gradients[name], rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                parameter, parameters_before[name], rtol=0.0, atol=0.0
            )


def test_optimizer_change_instrumentation_is_noninvasive() -> None:
    model = create_model(small_dataset(), 2, "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    features = torch.as_tensor(small_dataset()["features"], dtype=torch.float32)
    before = parameter_snapshot(model)
    optimizer.zero_grad()
    model(features).square().sum().backward()
    optimizer.step()
    changes = parameter_change_diagnostics(before, model, detailed=True)
    assert changes["all_expected_trainable_tensors_changed"] is True
    assert changes["expected_trainable_tensor_count"] == 6


def test_hidden_dead_neuron_definition_uses_all_training_rows() -> None:
    values = np.asarray([[0.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    diagnostics = hidden_activation_diagnostics(values)
    assert diagnostics["dead_neuron_count"] == 1
    assert diagnostics["dead_neuron_fraction"] == pytest.approx(1.0 / 3.0)
    assert diagnostics["fraction_exactly_zero"] == pytest.approx(4.0 / 6.0)
    assert diagnostics["fraction_positive"] == pytest.approx(2.0 / 6.0)


def test_final_output_is_linear_unbounded_and_preserves_negative_bias() -> None:
    model = create_model(small_dataset(), 0, "cpu")
    linears = [module for module in model.seq if isinstance(module, nn.Linear)]
    relus = [module for module in model.seq if isinstance(module, nn.ReLU)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (3, 64),
        (64, 128),
        (128, 1),
    ]
    assert len(relus) == 2
    assert isinstance(model.seq[-1], nn.Linear)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
        model.seq[-1].bias.fill_(-2.5)
    features = torch.as_tensor(small_dataset()["features"], dtype=torch.float32)
    _hidden_1, _hidden_2, output = instrumented_forward(model, features)
    torch.testing.assert_close(output, torch.full_like(output, -2.5))
    corrected = subtract_predicted_bias(
        torch.tensor([100.0], dtype=torch.float64),
        torch.tensor([-2.5], dtype=torch.float64),
    )
    torch.testing.assert_close(corrected, torch.tensor([102.5], dtype=torch.float64))


def test_actual_klt3_corrected_gt_remains_one_to_one() -> None:
    shared = ROOT / "validation/paper_weightnet"
    dataset, _manifest = load_dataset(
        shared / "klt3_features.npz",
        shared / "klt3_feature_manifest.json",
        torch.device("cpu"),
    )
    validate_training_dataset(dataset)
    indices = corrected_target_indices(405)
    np.testing.assert_array_equal(indices, np.arange(405))
    assert dataset["ground_truth_geodetic_deg_m"].shape == (405, 3)
    assert np.unique(indices).size == 405
    historical_seed_model = create_model(dataset, DEFAULT_SEED, "cpu")
    assert model_state_sha256(historical_seed_model) == (
        "ee991c755bbc6083fdf66d082c7eaa99b0e34f12021e424fdf98a5ef73e2ede2"
    )


def test_seed_changes_no_data_normalization_or_configuration() -> None:
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
    assert data_before["duplicated_historical_gt_defect_present"] is False
    for name in dataset:
        np.testing.assert_array_equal(dataset[name], untouched[name])


def mock_result(seed: int, loss: float) -> dict[str, object]:
    hidden_1 = {"dead_neuron_count": seed, "dead_neuron_fraction": seed / 64.0}
    hidden_2 = {"dead_neuron_count": seed + 1, "dead_neuron_fraction": (seed + 1) / 128.0}
    return {
        "status": "completed",
        "scientific_aggregate_eligible": True,
        "seed": seed,
        "executed_training_epochs": 500,
        "finite": {"all_finite": True},
        "configuration_sha256": "same-config",
        "normalization_sha256": "same-normalization",
        "training_data_identity_sha256": "same-data",
        "held_out_data_identity_sha256": "same-held-out",
        "training_loss": {
            "epoch_1_pre_update_sum_3d_m": loss + 10.0,
            "final_post_update_sum_3d_m": loss,
            "minimum_evaluated": {
                "epoch": 500,
                "stage": "final_post_update",
                "sum_3d_m": loss,
            },
            "pre_update_history_sum_3d_m": [loss + 1.0] * 500,
        },
        "final_diagnostics": {
            "hidden_layer_1": hidden_1,
            "hidden_layer_2": hidden_2,
            "bias_output": {"mean": loss / 10.0, "std": loss / 20.0},
        },
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


def test_summary_parser_orders_without_ranking_and_builds_loss_band(tmp_path: Path) -> None:
    results = [mock_result(1, 20.0), mock_result(0, 10.0)]
    for result in results:
        (tmp_path / f"seed_{result['seed']}.json").write_text(json.dumps(result))
    loaded = load_results(tmp_path, [0, 1])
    summary = build_summary(loaded)
    assert [row["seed"] for row in summary["rows"]] == [0, 1]
    assert summary["seed_order"].startswith("numeric")
    assert summary["aggregate_statistics"]["final_training_loss_sum_3d_m"] == pytest.approx(
        {
            "mean": 15.0,
            "median": 15.0,
            "std": 5.0,
            "minimum": 10.0,
            "maximum": 20.0,
        }
    )
    assert len(summary["loss_curves"]["median"]) == 500
    assert summary["historical_reference"]["aggregate_membership"] is False
    table = markdown_table(summary)
    assert "not ranked" in table
    assert "| 0 |" in table


def test_smoke_result_is_excluded_from_scientific_aggregation() -> None:
    smoke = copy.deepcopy(mock_result(0, 10.0))
    smoke["status"] = "smoke_completed_incomplete"
    smoke["scientific_aggregate_eligible"] = False
    smoke["executed_training_epochs"] = 1
    with pytest.raises(ValueError, match="not a completed scientific run"):
        row_from_result(smoke)


def test_summary_rejects_changed_control_hash() -> None:
    first = mock_result(0, 10.0)
    second = mock_result(1, 11.0)
    second["normalization_sha256"] = "changed"
    with pytest.raises(ValueError, match="normalization_sha256"):
        build_summary([first, second])


def test_independent_metric_checker_still_agrees(tmp_path: Path) -> None:
    csv_path = tmp_path / "epochs.csv"
    summary_path = tmp_path / "summary.json"
    rows = [
        {
            "error_2d_m": "5",
            "error_3d_m": "13",
            "ols_error_2d_m": "10",
            "ols_error_3d_m": "20",
        },
        {
            "error_2d_m": "7",
            "error_3d_m": "15",
            "ols_error_2d_m": "14",
            "ols_error_3d_m": "24",
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
