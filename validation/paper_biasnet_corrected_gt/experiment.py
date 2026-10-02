"""Controlled BiasNet A/B helpers with one-to-one KLT3 ground truth.

This module deliberately reuses the historical reproduction's architecture,
solver, features, normalization, optimizer settings, and training order.  The
only experimental variable is the array used to select the KLT3 GT target.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pymap3d as p3d
import torch

from validation.paper_biasnet.core import (
    instantiate_released_biasnet,
    solve_paper_bias_position,
)
from validation.paper_biasnet.train_paper_biasnet import (
    DEFAULT_SEED,
    EXPECTED_EPOCHS,
    EXPECTED_MEASUREMENTS,
    RELEASED_LEARNING_RATE,
    set_seed,
)


TRAINING_DATASETS = ("KLT3",)
HELD_OUT_DATASETS = ("KLT1", "KLT2")
MAXIMUM_WLS_ITERATIONS = 10
WLS_TOLERANCE = 1.0e-4


def historical_target_indices(epoch_count: int = EXPECTED_EPOCHS) -> np.ndarray:
    """Indices produced by the released trainer's duplicate append."""

    if epoch_count < 1:
        raise ValueError("epoch_count must be positive")
    return np.arange(epoch_count, dtype=np.int64) // 2


def corrected_target_indices(epoch_count: int = EXPECTED_EPOCHS) -> np.ndarray:
    """One temporally matched target for each retained training epoch."""

    if epoch_count < 1:
        raise ValueError("epoch_count must be positive")
    return np.arange(epoch_count, dtype=np.int64)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_training_dataset(dataset: Mapping[str, np.ndarray]) -> None:
    """Enforce the one-target-per-epoch corrected-training contract."""

    offsets = dataset["epoch_offsets"]
    epoch_times = dataset["epoch_times_gpst_like"]
    targets = dataset["ground_truth_geodetic_deg_m"]
    features = dataset["features"]
    training_epoch_count = int(offsets.size - 1)
    if training_epoch_count != EXPECTED_EPOCHS:
        raise RuntimeError(
            f"expected {EXPECTED_EPOCHS} KLT3 epochs, got {training_epoch_count}"
        )
    if features.shape != (EXPECTED_MEASUREMENTS, 3):
        raise RuntimeError(f"unexpected KLT3 feature shape: {features.shape}")
    if int(offsets[-1]) != EXPECTED_MEASUREMENTS:
        raise RuntimeError(f"unexpected KLT3 measurement count: {offsets[-1]}")
    if epoch_times.shape != (training_epoch_count,):
        raise RuntimeError("GNSS timestamp count differs from training epoch count")
    if targets.shape != (training_epoch_count, 3):
        raise RuntimeError("GT target count differs from training epoch count")
    indices = corrected_target_indices(training_epoch_count)
    if not np.array_equal(indices, np.arange(training_epoch_count)):
        raise RuntimeError("corrected targets are not in one-to-one epoch order")
    if np.unique(indices).size != training_epoch_count:
        raise RuntimeError("corrected targets contain a duplicate index")
    if not np.all(np.diff(epoch_times) > 0.0):
        raise RuntimeError("KLT3 training epochs are not strictly chronological")


@dataclass(frozen=True)
class ControlledABMapping:
    features: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    historical_indices: np.ndarray
    corrected_indices: np.ndarray
    historical_targets: np.ndarray
    corrected_targets: np.ndarray


def controlled_ab_mapping(
    dataset: Mapping[str, np.ndarray],
) -> ControlledABMapping:
    """Build A and B from the same cache; only target selection differs."""

    validate_training_dataset(dataset)
    features = dataset["features"]
    targets = dataset["ground_truth_geodetic_deg_m"]
    epoch_count = targets.shape[0]
    historical_indices = historical_target_indices(epoch_count)
    corrected_indices = corrected_target_indices(epoch_count)
    return ControlledABMapping(
        features=features,
        feature_mean=features.mean(axis=0),
        feature_std=features.std(axis=0),
        historical_indices=historical_indices,
        corrected_indices=corrected_indices,
        historical_targets=targets[historical_indices],
        corrected_targets=targets[corrected_indices],
    )


def optimizer_signature(optimizer: torch.optim.Optimizer) -> dict[str, object]:
    group = optimizer.param_groups[0]
    return {
        "class": type(optimizer).__name__,
        "learning_rate": float(group["lr"]),
        "betas": [float(item) for item in group["betas"]],
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "amsgrad": bool(group["amsgrad"]),
        "initial_state_entries": len(optimizer.state),
    }


def initialize_controlled_ab(
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, object]]:
    """Return B's model after proving A and B start from identical state."""

    set_seed(seed)
    historical_model = instantiate_released_biasnet(
        feature_mean, feature_std, device=device
    )
    historical_optimizer = torch.optim.Adam(
        historical_model.parameters(), lr=RELEASED_LEARNING_RATE
    )
    set_seed(seed)
    corrected_model = instantiate_released_biasnet(
        feature_mean, feature_std, device=device
    )
    corrected_optimizer = torch.optim.Adam(
        corrected_model.parameters(), lr=RELEASED_LEARNING_RATE
    )
    historical_hash = model_state_sha256(historical_model)
    corrected_hash = model_state_sha256(corrected_model)
    parameters_equal = all(
        torch.equal(left, right)
        for left, right in zip(
            historical_model.state_dict().values(),
            corrected_model.state_dict().values(),
            strict=True,
        )
    )
    historical_optimizer_signature = optimizer_signature(historical_optimizer)
    corrected_optimizer_signature = optimizer_signature(corrected_optimizer)
    if not parameters_equal or historical_hash != corrected_hash:
        raise RuntimeError("A/B initial model parameters differ")
    if historical_optimizer_signature != corrected_optimizer_signature:
        raise RuntimeError("A/B optimizer configurations differ")
    return corrected_model, corrected_optimizer, {
        "seed": seed,
        "same_theta_0": True,
        "historical_initial_model_sha256": historical_hash,
        "corrected_initial_model_sha256": corrected_hash,
        "historical_optimizer": historical_optimizer_signature,
        "corrected_optimizer": corrected_optimizer_signature,
        "same_optimizer_configuration": True,
    }


def ab_control_record(
    dataset: Mapping[str, np.ndarray], initialization: dict[str, object]
) -> dict[str, object]:
    mapping = controlled_ab_mapping(dataset)
    target_tensors_differ = not np.array_equal(
        mapping.historical_targets, mapping.corrected_targets
    )
    shared_arrays = {
        name: _array_sha256(dataset[name])
        for name in (
            "features",
            "satellite_positions_ecef_m",
            "satellite_clock_bias_s",
            "corrected_pseudorange_m",
            "system_clock_indices",
            "epoch_offsets",
            "initial_states",
            "epoch_times_gpst_like",
        )
    }
    return {
        **initialization,
        "training_datasets": list(TRAINING_DATASETS),
        "held_out_datasets": list(HELD_OUT_DATASETS),
        "held_out_data_used_for_training": False,
        "same_feature_array": True,
        "same_feature_array_sha256": _array_sha256(mapping.features),
        "same_normalization": True,
        "feature_mean": mapping.feature_mean.tolist(),
        "feature_population_std": mapping.feature_std.tolist(),
        "shared_non_target_array_sha256": shared_arrays,
        "historical_target_index_sha256": _array_sha256(mapping.historical_indices),
        "corrected_target_index_sha256": _array_sha256(mapping.corrected_indices),
        "historical_target_tensor_sha256": _array_sha256(mapping.historical_targets),
        "corrected_target_tensor_sha256": _array_sha256(mapping.corrected_targets),
        "target_tensors_differ": target_tensors_differ,
        "only_gt_target_mapping_differs": bool(
            initialization["same_theta_0"]
            and initialization["same_optimizer_configuration"]
            and target_tensors_differ
        ),
    }


def epoch_position_loss(
    predicted_state: torch.Tensor, ground_truth_geodetic: np.ndarray
) -> torch.Tensor:
    east, north, up = p3d.ecef2enu(
        predicted_state[0],
        predicted_state[1],
        predicted_state[2],
        float(ground_truth_geodetic[0]),
        float(ground_truth_geodetic[1]),
        float(ground_truth_geodetic[2]),
    )
    return torch.linalg.vector_norm(torch.stack((east, north, up)))


def corrected_full_dataset_loss(
    model: torch.nn.Module,
    dataset: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Historical objective and solver with target ``i`` mapped to GT ``i``."""

    validate_training_dataset(dataset)
    features = dataset["features"]
    offsets = dataset["epoch_offsets"]
    ground_truth = dataset["ground_truth_geodetic_deg_m"]
    target_indices = corrected_target_indices(EXPECTED_EPOCHS)
    loss: torch.Tensor | int = 0
    minimum_bias = float("inf")
    maximum_bias = float("-inf")
    maximum_wls_iterations = 0

    for epoch_index, gt_index in enumerate(target_indices):
        if int(gt_index) != epoch_index:
            raise RuntimeError("corrected GT target is not owned by its GNSS epoch")
        start = int(offsets[epoch_index])
        stop = int(offsets[epoch_index + 1])
        feature_tensor = torch.as_tensor(
            features[start:stop], dtype=torch.float32, device=device
        )
        predicted_bias = model(feature_tensor).squeeze(-1)
        if not bool(torch.all(torch.isfinite(predicted_bias)).detach().cpu()):
            raise RuntimeError(f"non-finite BiasNet output at epoch {epoch_index}")
        minimum_bias = min(
            minimum_bias, float(torch.min(predicted_bias).detach().cpu())
        )
        maximum_bias = max(
            maximum_bias, float(torch.max(predicted_bias).detach().cpu())
        )
        solution = solve_paper_bias_position(
            dataset["satellite_positions_ecef_m"][start:stop],
            dataset["satellite_clock_bias_s"][start:stop],
            dataset["corrected_pseudorange_m"][start:stop],
            dataset["system_clock_indices"][start:stop],
            predicted_bias,
            dataset["initial_states"][epoch_index],
            return_trace=True,
        )
        maximum_wls_iterations = max(
            maximum_wls_iterations, len(solution.iterations)
        )
        if not bool(torch.all(torch.isfinite(solution.state)).detach().cpu()):
            raise RuntimeError(f"non-finite WLS state at epoch {epoch_index}")
        # The controlled correction: one retained epoch, one matched GT target.
        epoch_loss = epoch_position_loss(solution.state, ground_truth[epoch_index])
        if not bool(torch.isfinite(epoch_loss).detach().cpu()):
            raise RuntimeError(f"non-finite position loss at epoch {epoch_index}")
        loss = loss + epoch_loss

    assert isinstance(loss, torch.Tensor)
    return loss, {
        "minimum_bias_m": minimum_bias,
        "maximum_bias_m": maximum_bias,
        "maximum_wls_iterations": maximum_wls_iterations,
    }


__all__ = [
    "ControlledABMapping",
    "HELD_OUT_DATASETS",
    "MAXIMUM_WLS_ITERATIONS",
    "TRAINING_DATASETS",
    "WLS_TOLERANCE",
    "ab_control_record",
    "controlled_ab_mapping",
    "corrected_full_dataset_loss",
    "corrected_target_indices",
    "historical_target_indices",
    "initialize_controlled_ab",
    "model_state_sha256",
    "optimizer_signature",
    "validate_training_dataset",
]
