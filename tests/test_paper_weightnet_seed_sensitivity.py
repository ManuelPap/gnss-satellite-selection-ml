import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from validation.paper_weightnet.check_exported_metrics import aggregate_csv
from validation.paper_weightnet_seed_sensitivity import experiment
from validation.paper_weightnet_seed_sensitivity.experiment import (
    architecture_identity,
    create_model,
    instrumented_forward,
    normalization_identity,
    output_diagnostics,
    parameter_hash,
    parameter_snapshot,
    scientific_configuration,
    training_data_identity,
)
from validation.paper_weightnet_seed_sensitivity.run_seed import (
    held_out_metrics,
    parse_args,
    preflight_held_out_inputs,
)
from validation.paper_weightnet_seed_sensitivity.summarize_seeds import (
    build_summary,
    load_results,
    markdown_table,
    row_from_result,
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


def manifest_for() -> dict[str, object]:
    return {"cache": {"sha256": "synthetic-cache-sha256"}}


def test_seed_argument_and_seed_call_precede_model_construction(monkeypatch) -> None:
    assert parse_args(["--seed", "7"]).seed == 7
    events: list[tuple[str, object]] = []
    sentinel = object()

    def record_seed(seed: int) -> None:
        events.append(("seed", seed))

    def record_construction(mean, std, *, device):
        events.append(("construct", device))
        return sentinel

    monkeypatch.setattr(experiment, "set_seed", record_seed)
    monkeypatch.setattr(
        experiment, "instantiate_released_weightnet", record_construction
    )
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


def test_instrumentation_preserves_forward_outputs_and_gradients() -> None:
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

    output, tensors = instrumented_forward(model, features)
    output.square().sum().backward()
    torch.testing.assert_close(output, expected_output, rtol=0.0, atol=0.0)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter.grad, expected_gradients[name], rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                parameter, parameters_before[name], rtol=0.0, atol=0.0
            )
    assert tensors["final_weight"].shape == (len(features), 1)


def test_exact_released_architecture_sigmoids_and_x10_clamp() -> None:
    model = create_model(small_dataset(), 0, "cpu")
    linears = [module for module in model.seq if isinstance(module, nn.Linear)]
    sigmoids = [module for module in model.seq if isinstance(module, nn.Sigmoid)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (3, 64),
        (64, 128),
        (128, 64),
        (64, 1),
    ]
    assert [index for index, layer in enumerate(model.seq) if isinstance(layer, nn.Sigmoid)] == [
        2,
        4,
        6,
        8,
    ]
    assert len(sigmoids) == 4
    identity = architecture_identity(model)
    assert identity["dimensions"] == [3, 64, 128, 64, 1]
    assert identity["output_transform"] == (
        "clamp(final_sigmoid * 10, min=0, max=10)"
    )

    features = torch.as_tensor(small_dataset()["features"], dtype=torch.float32)
    output, tensors = instrumented_forward(model, features)
    expected = torch.clamp(tensors["final_sigmoid"] * 10.0, 0.0, 10.0)
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    assert torch.all((output >= 0.0) & (output <= 10.0))


def test_final_weight_rows_remain_aligned_with_feature_rows() -> None:
    model = create_model(small_dataset(), 5, "cpu")
    features = torch.as_tensor(small_dataset()["features"], dtype=torch.float32)
    output, tensors = instrumented_forward(model, features)
    direct = model(features)
    reversed_output, _ = instrumented_forward(model, torch.flip(features, dims=(0,)))
    torch.testing.assert_close(output, direct, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        reversed_output, torch.flip(output, dims=(0,)), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(output, tensors["final_weight"], rtol=0.0, atol=0.0)


def test_sigmoid_and_gnss_weight_thresholds_are_distinctly_recorded() -> None:
    values = {
        "layer_1_preactivation": np.asarray([-10.0, 0.0, 10.0]),
        "layer_1_activation": np.asarray([0.001, 0.5, 0.999]),
        "layer_2_preactivation": np.asarray([-10.0, 0.0, 10.0]),
        "layer_2_activation": np.asarray([0.001, 0.5, 0.999]),
        "layer_3_preactivation": np.asarray([-10.0, 0.0, 10.0]),
        "layer_3_activation": np.asarray([0.001, 0.5, 0.999]),
        "final_preactivation": np.asarray([-10.0, 0.0, 10.0]),
        "final_sigmoid": np.asarray([0.001, 0.5, 0.999]),
        "final_weight": np.asarray([0.01, 5.0, 9.99]),
    }
    diagnostics = output_diagnostics(values)
    assert diagnostics["final_sigmoid"]["activation"]["fraction_gt_0_99"] == pytest.approx(
        1.0 / 3.0
    )
    assert diagnostics["final_weight"]["fraction_gt_9_9"] == pytest.approx(
        1.0 / 3.0
    )
    assert "saturation" in diagnostics["final_sigmoid"]["activation"][
        "threshold_purpose"
    ]
    assert "learned-weight" in diagnostics["final_weight"][
        "gnss_threshold_purpose"
    ]


def test_seed_changes_no_data_features_gt_normalization_or_configuration() -> None:
    dataset = small_dataset()
    untouched = {name: values.copy() for name, values in dataset.items()}
    data_before = training_data_identity(dataset, manifest_for())
    normalization_before = normalization_identity(dataset)
    first = create_model(dataset, 0, "cpu")
    second = create_model(dataset, 1, "cpu")
    first_optimizer = torch.optim.Adam(first.parameters(), lr=0.01)
    second_optimizer = torch.optim.Adam(second.parameters(), lr=0.01)
    assert parameter_hash(first) != parameter_hash(second)
    assert scientific_configuration(first, first_optimizer) == (
        scientific_configuration(second, second_optimizer)
    )
    assert training_data_identity(dataset, manifest_for()) == data_before
    assert normalization_identity(dataset) == normalization_before
    for name in dataset:
        np.testing.assert_array_equal(dataset[name], untouched[name])


def mock_result(seed: int, loss: float) -> dict[str, object]:
    final_weight = {
        "mean": loss / 10.0,
        "std": loss / 20.0,
        "fraction_lt_0_01": seed / 100.0,
    }
    final_sigmoid = {
        "activation": {
            "fraction_lt_0_01": seed / 100.0,
            "fraction_gt_0_99": seed / 200.0,
        }
    }
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
            "final_weight": final_weight,
            "final_sigmoid": final_sigmoid,
        },
        "gradient_history": [{"final_output_layer_l2_norm": loss / 100.0}],
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
    assert summary["aggregate_statistics"][
        "final_training_loss_sum_3d_m"
    ] == pytest.approx(
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


def test_smoke_output_is_excluded_from_scientific_aggregation() -> None:
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


def test_held_out_evaluator_preserves_model_values_and_csv_checker_agrees(
    monkeypatch, tmp_path: Path
) -> None:
    import validation.paper_weightnet_seed_sensitivity.run_seed as run_seed

    model = create_model(small_dataset(), 8, "cpu")
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    prepared = {
        name: SimpleNamespace(
            spec=SimpleNamespace(name=name), measurement_count=2, epochs=()
        )
        for name in ("KLT1", "KLT2")
    }

    def fake_evaluate(_model, _prepared):
        return [
            SimpleNamespace(error_2d_m=5.0, error_3d_m=13.0),
            SimpleNamespace(error_2d_m=7.0, error_3d_m=15.0),
        ]

    def fake_write(path, _name, evaluations):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "error_2d_m",
                    "error_3d_m",
                    "ols_error_2d_m",
                    "ols_error_3d_m",
                ),
            )
            writer.writeheader()
            for item in evaluations:
                writer.writerow(
                    {
                        "error_2d_m": item.error_2d_m,
                        "error_3d_m": item.error_3d_m,
                        "ols_error_2d_m": 10.0,
                        "ols_error_3d_m": 20.0,
                    }
                )

    monkeypatch.setattr(run_seed, "evaluate_prepared_dataset", fake_evaluate)
    monkeypatch.setattr(run_seed, "write_results_csv", fake_write)
    monkeypatch.setattr(
        run_seed,
        "prepared_dataset_identity",
        lambda value: {
            "dataset": value.spec.name,
            "valid_epoch_count": 2,
            "retained_measurement_count": 2,
            "identity_sha256": f"identity-{value.spec.name}",
        },
    )
    metrics, identities = held_out_metrics(model, prepared)
    assert metrics["KLT1"]["independent_checker"]["agreed"] is True
    assert identities["model_parameter_values_unchanged"] is True
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)

    csv_path = tmp_path / "metrics.csv"
    fake_write(
        csv_path,
        "KLT1",
        [SimpleNamespace(error_2d_m=5.0, error_3d_m=13.0)],
    )
    assert aggregate_csv(csv_path)["mean_3d_error_m"] == 13.0


def test_full_run_preflight_checks_inputs_without_parsing_rows(monkeypatch) -> None:
    import validation.paper_weightnet_seed_sensitivity.run_seed as run_seed

    args = parse_args(["--seed", "0"])
    events: list[tuple[str, str]] = []

    def fake_resolve(spec, **_kwargs):
        events.append(("resolve", spec.name))
        return SimpleNamespace(dataset=spec.name)

    def fake_verify(spec, inputs):
        assert inputs.dataset == spec.name
        events.append(("verify", spec.name))
        return {"dataset": spec.name, "hashes": "audited"}

    def fake_load(inputs):
        events.append(("load_runtime", inputs.dataset))
        return object(), object()

    monkeypatch.setattr(run_seed, "resolve_input_paths", fake_resolve)
    monkeypatch.setattr(run_seed, "_verify_input_files", fake_verify)
    monkeypatch.setattr(run_seed, "load_historical_modules", fake_load)
    records = preflight_held_out_inputs(args)

    assert list(records) == ["KLT1", "KLT2"]
    assert all(record["status"] == "passed" for record in records.values())
    assert all(record["held_out_rows_parsed"] is False for record in records.values())
    assert events == [
        ("resolve", "KLT1"),
        ("verify", "KLT1"),
        ("load_runtime", "KLT1"),
        ("resolve", "KLT2"),
        ("verify", "KLT2"),
        ("load_runtime", "KLT2"),
    ]
