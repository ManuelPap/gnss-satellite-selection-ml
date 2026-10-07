import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from validation.ibiza_generalization.checkpoints import (
    ARCHITECTURES,
    REPOSITORY_ROOT,
    checkpoint_record,
    load_checkpoint_manifest,
    load_frozen_model,
    validate_checkpoint_inventory,
)
from validation.ibiza_generalization.inference import (
    DEFAULT_IBIZA_NPZ,
    IBIZA_NPZ_SHA256,
    SMOKE_EPOCH_INDICES,
    evaluate_architecture,
    evaluate_epoch,
    infer_neural_outputs,
    load_ibiza_dataset,
    run_seed_zero_smoke,
)


FROZEN_MEAN = torch.tensor(
    [29.084903717041016, 0.8471900224685669, -1.7873233559839719e-07],
    dtype=torch.float64,
)
FROZEN_STD = torch.tensor(
    [5.891841888427734, 0.2879891097545624, 4.3018035888671875],
    dtype=torch.float64,
)


def _assert_optional_tensor_equal(
    first: torch.Tensor | None, second: torch.Tensor | None
) -> None:
    if first is None or second is None:
        assert first is second
    else:
        assert torch.equal(first, second)


@pytest.fixture(scope="module")
def ibiza_dataset() -> dict[str, np.ndarray]:
    return load_ibiza_dataset()


@pytest.fixture(scope="module")
def smoke_results():
    return run_seed_zero_smoke()


def test_manifest_inventory_verifies_all_30_frozen_checkpoints() -> None:
    summary = validate_checkpoint_inventory()
    assert summary.architecture_counts == {
        "TDL-B": 10,
        "TDL-W": 10,
        "TDL-BW": 10,
    }
    assert len(summary.records) == summary.distinct_hash_count == 30
    assert set((record.architecture, record.seed) for record in summary.records) == {
        (architecture, seed)
        for architecture in ARCHITECTURES
        for seed in range(10)
    }
    assert torch.equal(torch.tensor(summary.common_mean), FROZEN_MEAN)
    assert torch.equal(torch.tensor(summary.common_std), FROZEN_STD)
    manifest_text = Path(
        "validation/ibiza_generalization/frozen_checkpoint_manifest.json"
    ).read_text(encoding="utf-8")
    assert "/home/" not in manifest_text


def test_checkpoint_sha_mismatch_is_rejected_before_loading(tmp_path: Path) -> None:
    record = checkpoint_record("TDL-B", 0)
    original = REPOSITORY_ROOT / record.checkpoint_path
    corrupted = tmp_path / original.name
    corrupted.write_bytes(original.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="checkpoint SHA-256 mismatch"):
        load_frozen_model("TDL-B", 0, checkpoint_path=corrupted)


def test_manifest_records_exact_state_keys_and_embedded_normalization() -> None:
    records = load_checkpoint_manifest()
    for record in records:
        state = torch.load(
            REPOSITORY_ROOT / record.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        assert tuple(state) == record.expected_state_dict_keys
        assert torch.equal(state["seq.0.mean"], FROZEN_MEAN)
        assert torch.equal(state["seq.0.std"], FROZEN_STD)


def test_checkpoint_normalization_is_loaded_and_not_replaced_by_ibiza_statistics(
    ibiza_dataset: dict[str, np.ndarray],
) -> None:
    # Do not calculate Ibiza statistics even in this control.  The raw matrix
    # is merely present; normalization comes only from checkpoint state.
    assert ibiza_dataset["features"].shape[1] == 3
    inference_source = Path(
        "validation/ibiza_generalization/inference.py"
    ).read_text(encoding="utf-8")
    assert ".mean(" not in inference_source
    assert ".std(" not in inference_source
    for architecture in ARCHITECTURES:
        frozen = load_frozen_model(architecture, 0)
        assert torch.equal(frozen.model.seq[0].mean, FROZEN_MEAN)
        assert torch.equal(frozen.model.seq[0].std, FROZEN_STD)


def test_all_seed_zero_methods_receive_identical_raw_rows_and_ordering(
    smoke_results,
) -> None:
    reference = smoke_results[0]
    assert tuple(epoch.accepted_epoch_index for epoch in reference.epochs) == (
        SMOKE_EPOCH_INDICES
    )
    for candidate in smoke_results[1:]:
        for expected_epoch, actual_epoch in zip(
            reference.epochs, candidate.epochs, strict=True
        ):
            assert actual_epoch.row_identity == expected_epoch.row_identity
            assert actual_epoch.row_start == expected_epoch.row_start
            assert actual_epoch.row_stop == expected_epoch.row_stop
            assert torch.equal(actual_epoch.raw_features, expected_epoch.raw_features)


def test_output_semantics_and_released_hybrid_branches(
    ibiza_dataset: dict[str, np.ndarray], smoke_results
) -> None:
    by_architecture = {result.architecture: result for result in smoke_results}
    bias = by_architecture["TDL-B"].epochs[0].neural_outputs
    assert bias.semantics == ("bias_m",)
    assert bias.predicted_bias_m is not None
    assert bias.predicted_weight is None

    weight = by_architecture["TDL-W"].epochs[0].neural_outputs
    assert weight.semantics == ("weight",)
    assert weight.predicted_bias_m is None
    assert weight.predicted_weight is not None

    hybrid = by_architecture["TDL-BW"].epochs[0].neural_outputs
    assert hybrid.semantics == ("weight", "bias_m")
    assert hybrid.predicted_weight is not None
    assert hybrid.predicted_bias_m is not None

    frozen = load_frozen_model("TDL-BW", 0)
    rows = slice(
        int(ibiza_dataset["epoch_offsets"][0]),
        int(ibiza_dataset["epoch_offsets"][1]),
    )
    features = torch.from_numpy(ibiza_dataset["features"][rows])
    with torch.inference_mode():
        raw = frozen.model.raw_output(features)
    assert torch.equal(hybrid.predicted_weight, torch.sigmoid(raw[:, 0]).clamp(0, 1))
    assert torch.equal(hybrid.predicted_bias_m, F.relu(raw[:, 1]))


def test_repeated_inference_is_identical_and_does_not_mutate_training_state(
    ibiza_dataset: dict[str, np.ndarray],
) -> None:
    rows = slice(
        int(ibiza_dataset["epoch_offsets"][0]),
        int(ibiza_dataset["epoch_offsets"][1]),
    )
    features = ibiza_dataset["features"][rows]
    for architecture in ARCHITECTURES:
        frozen = load_frozen_model(architecture, 0)
        before = {
            name: value.detach().clone()
            for name, value in frozen.model.state_dict().items()
        }
        first = infer_neural_outputs(frozen.model, architecture, features)
        second = infer_neural_outputs(frozen.model, architecture, features)
        _assert_optional_tensor_equal(
            first.predicted_bias_m, second.predicted_bias_m
        )
        _assert_optional_tensor_equal(
            first.predicted_weight, second.predicted_weight
        )
        assert frozen.model.training is False
        assert all(parameter.grad is None for parameter in frozen.model.parameters())
        assert all(
            parameter.requires_grad is False
            for parameter in frozen.model.parameters()
        )
        assert all(
            torch.equal(before[name], value)
            for name, value in frozen.model.state_dict().items()
        )

        first_epoch = evaluate_epoch(frozen, ibiza_dataset, 0)
        second_epoch = evaluate_epoch(frozen, ibiza_dataset, 0)
        assert torch.equal(first_epoch.receiver_state, second_epoch.receiver_state)


def test_future_ground_truth_cannot_affect_features_outputs_or_position(
    ibiza_dataset: dict[str, np.ndarray],
) -> None:
    assert not any(
        "ground_truth" in name
        for name in inspect.signature(evaluate_epoch).parameters
    )
    absent = dict(ibiza_dataset)
    first_ground_truth = dict(ibiza_dataset)
    first_ground_truth["future_ground_truth_ecef_m"] = np.asarray(
        [1.0, 2.0, 3.0], dtype=np.float64
    )
    changed_ground_truth = dict(ibiza_dataset)
    changed_ground_truth["future_ground_truth_ecef_m"] = np.asarray(
        [1.0e12, -1.0e12, 7.0e11], dtype=np.float64
    )

    for architecture in ARCHITECTURES:
        frozen = load_frozen_model(architecture, 0)
        results = [
            evaluate_epoch(frozen, dataset, 0)
            for dataset in (absent, first_ground_truth, changed_ground_truth)
        ]
        for candidate in results[1:]:
            assert torch.equal(candidate.raw_features, results[0].raw_features)
            _assert_optional_tensor_equal(
                candidate.neural_outputs.predicted_bias_m,
                results[0].neural_outputs.predicted_bias_m,
            )
            _assert_optional_tensor_equal(
                candidate.neural_outputs.predicted_weight,
                results[0].neural_outputs.predicted_weight,
            )
            assert torch.equal(candidate.receiver_state, results[0].receiver_state)


def test_every_smoke_solution_reports_rank_conditioning_and_convergence(
    smoke_results,
) -> None:
    for architecture in smoke_results:
        assert architecture.checkpoint_parameters_unchanged is True
        for epoch in architecture.epochs:
            assert epoch.wls.solution_status == "solved"
            assert epoch.wls.rank_status == "full_rank"
            assert epoch.wls.final_rank == epoch.wls.active_state_count
            assert epoch.wls.conditioning_status == "finite"
            assert np.isfinite(epoch.wls.final_condition_number)
            assert epoch.wls.converged is True
            assert epoch.wls.convergence_status == "converged"
            assert len(epoch.wls.iterations) == 3
            assert all(item.full_rank for item in epoch.wls.iterations)
            assert all(item.condition_finite for item in epoch.wls.iterations)


def test_common_evaluator_reports_parameter_immutability(
    ibiza_dataset: dict[str, np.ndarray],
) -> None:
    frozen = load_frozen_model("TDL-W", 0)
    result = evaluate_architecture(frozen, ibiza_dataset, SMOKE_EPOCH_INDICES)
    assert result.checkpoint_parameters_unchanged is True
    assert result.output_semantics == ("weight",)


def test_required_ibiza_hash_constant_matches_dataset() -> None:
    from validation.ibiza_generalization.checkpoints import sha256_file

    assert sha256_file(DEFAULT_IBIZA_NPZ) == IBIZA_NPZ_SHA256
