"""Shared machinery for the controlled TDL-BW seed-sensitivity runs.

This module deliberately keeps the seed outside the scientific configuration
hash.  The seed is the treatment variable; every hashed control is invariant.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from validation.paper_biasnet.train_paper_biasnet import load_dataset
from validation.paper_hybrid.core import (
    HybridShareNet,
    instantiate_released_hybrid,
    solve_paper_hybrid_position,
)
from validation.paper_hybrid.train_paper_hybrid import (
    DEFAULT_SEED,
    EXPECTED_EPOCHS,
    EXPECTED_MEASUREMENTS,
    MAXIMUM_WLS_ITERATIONS,
    RELEASED_LEARNING_RATE,
    WLS_TOLERANCE,
    epoch_position_loss,
    set_seed,
)


PREDEFINED_SEEDS = tuple(range(10))
TRAINING_EPOCHS = 100


def canonical_json_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_hash(digest: object, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    _update_hash(digest, "dtype", array.dtype.str.encode("ascii"))
    _update_hash(digest, "shape", json.dumps(array.shape).encode("ascii"))
    _update_hash(digest, "values", array.tobytes(order="C"))
    return digest.hexdigest()


def named_array_sha256(values: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(values, key=lambda item: item[0]):
        _update_hash(digest, name, array_sha256(array).encode("ascii"))
    return digest.hexdigest()


def named_tensor_sha256(values: Iterable[tuple[str, torch.Tensor]]) -> str:
    return named_array_sha256(
        (name, tensor.detach().cpu().contiguous().numpy()) for name, tensor in values
    )


def parameter_hash(model: nn.Module) -> str:
    """Hash trainable initial parameters, excluding normalization buffers."""

    return named_tensor_sha256(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def parameter_tensor_hashes(model: nn.Module) -> dict[str, str]:
    return {
        name: array_sha256(parameter.detach().cpu().contiguous().numpy())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def training_data_identity(
    dataset: Mapping[str, np.ndarray], manifest: Mapping[str, object]
) -> dict[str, object]:
    """Identify every cached training array, including features, rows, and GT."""

    array_hashes = {
        name: array_sha256(values) for name, values in sorted(dataset.items())
    }
    cache = manifest["cache"]
    record = {
        "dataset": "KLT3",
        "epoch_count": int(dataset["ground_truth_geodetic_deg_m"].shape[0]),
        "measurement_count": int(dataset["features"].shape[0]),
        "feature_columns": int(dataset["features"].shape[1]),
        "cache_sha256": str(cache["sha256"]),
        "array_sha256": array_hashes,
        "row_identity_sha256": named_array_sha256(
            (
                ("epoch_offsets", dataset["epoch_offsets"]),
                ("satellite_ids", dataset["satellite_ids"]),
                ("split_epoch_indices", dataset["split_epoch_indices"]),
                ("epoch_times_gpst_like", dataset["epoch_times_gpst_like"]),
            )
        ),
        "features_sha256": array_hashes["features"],
        "ground_truth_sha256": array_hashes["ground_truth_geodetic_deg_m"],
    }
    record["identity_sha256"] = canonical_json_hash(record)
    return record


def normalization_identity(dataset: Mapping[str, np.ndarray]) -> dict[str, object]:
    source_mean = dataset["features"].mean(axis=0)
    source_std = dataset["features"].std(axis=0)
    stored_mean = source_mean.astype(np.float32).astype(np.float64)
    stored_std = source_std.astype(np.float32).astype(np.float64)
    return {
        "sha256": named_array_sha256(
            (("stored_mean", stored_mean), ("stored_std", stored_std))
        ),
        "source_mean_float64": source_mean.tolist(),
        "source_population_std_float64": source_std.tolist(),
        "model_mean_float32_then_double": stored_mean.tolist(),
        "model_std_float32_then_double": stored_std.tolist(),
    }


def architecture_identity(model: HybridShareNet) -> dict[str, object]:
    modules = [
        {
            "name": name,
            "type": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
        }
        for name, module in model.named_modules()
    ]
    parameters = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": parameter.requires_grad,
        }
        for name, parameter in model.named_parameters()
    ]
    record = {
        "class": "validation.paper_hybrid.core.HybridShareNet",
        "modules": modules,
        "parameters": parameters,
        "raw_output_columns": {"0": "weight", "1": "bias"},
        "output_transform": {
            "weight": "clamp(sigmoid(raw[:, 0]), 0, 1)",
            "bias": "relu(raw[:, 1])",
        },
    }
    record["sha256"] = canonical_json_hash(record)
    return record


def optimizer_identity(optimizer: torch.optim.Optimizer) -> dict[str, object]:
    group = optimizer.param_groups[0]
    return {
        "class": "torch.optim.Adam",
        "learning_rate": float(group["lr"]),
        "betas": [float(value) for value in group["betas"]],
        "epsilon": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "amsgrad": bool(group["amsgrad"]),
        "maximize": bool(group["maximize"]),
        "optimizer_updates_per_training_epoch": 1,
    }


def scientific_configuration(
    model: HybridShareNet, optimizer: torch.optim.Optimizer
) -> dict[str, object]:
    """Return all scientific settings except the deliberately varied seed."""

    architecture = architecture_identity(model)
    first_parameter = next(model.parameters())
    return {
        "reference_commit": "dd5eac669676ba0a922102047e58c2dfc9be9267",
        "architecture_sha256": architecture["sha256"],
        "architecture": architecture,
        "execution": {
            "training_device": str(first_parameter.device),
            "model_dtype": str(first_parameter.dtype),
        },
        "optimizer": optimizer_identity(optimizer),
        "training": {
            "epochs": TRAINING_EPOCHS,
            "iteration_order": "chronological",
            "shuffle": False,
            "batch_configuration_read_but_unused": 128,
            "loss": "sum of per-epoch 3D ENU Euclidean position-error norms",
            "ground_truth": "one aligned KLT3 geodetic target per GNSS epoch",
            "features": ["SNR", "elevation", "OLS residual"],
        },
        "solver": {
            "maximum_iterations": MAXIMUM_WLS_ITERATIONS,
            "convergence_tolerance": WLS_TOLERANCE,
            "bias_observation_model": "corrected_pseudorange_m - predicted_bias_m",
            "weight_observation_model": "one diagonal weight in H.T @ W @ H",
        },
        "evaluation": {
            "datasets": ["KLT1", "KLT2"],
            "model_state": "frozen after exactly 100 Adam updates",
            "mode": "eval with torch.no_grad",
            "procedure": "released timestamp/OLS filters and historical WLS predictor",
            "aggregation_2d": "mean_i(sqrt(E_i^2 + N_i^2))",
            "aggregation_3d": "mean_i(sqrt(E_i^2 + N_i^2 + U_i^2))",
        },
        "seed_included_in_configuration_hash": False,
    }


def create_model(
    dataset: Mapping[str, np.ndarray], seed: int, device: torch.device | str
) -> HybridShareNet:
    """Seed all RNGs immediately before default Linear initialization."""

    set_seed(seed)
    return instantiate_released_hybrid(
        dataset["features"].mean(axis=0),
        dataset["features"].std(axis=0),
        device=device,
    )


def create_model_and_optimizer(
    dataset: Mapping[str, np.ndarray], seed: int, device: torch.device | str
) -> tuple[HybridShareNet, torch.optim.Adam]:
    model = create_model(dataset, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RELEASED_LEARNING_RATE)
    return model, optimizer


def instrumented_forward(
    model: HybridShareNet, features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose raw outputs while using the exact released transformations once."""

    raw = model.raw_output(features)
    weight, bias = model.transform_raw(raw)
    return raw, weight, bias


def distribution(values: np.ndarray | torch.Tensor) -> dict[str, object]:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p5": float(np.percentile(array, 5.0)),
        "p95": float(np.percentile(array, 95.0)),
        "all_finite": bool(np.all(np.isfinite(array))),
    }


def output_diagnostics(
    raw_bias_preactivation: np.ndarray | torch.Tensor,
    weight: np.ndarray | torch.Tensor,
    bias: np.ndarray | torch.Tensor,
) -> dict[str, object]:
    raw_array = (
        raw_bias_preactivation.detach().cpu().numpy()
        if isinstance(raw_bias_preactivation, torch.Tensor)
        else np.asarray(raw_bias_preactivation)
    ).reshape(-1)
    weight_array = (
        weight.detach().cpu().numpy()
        if isinstance(weight, torch.Tensor)
        else np.asarray(weight)
    ).reshape(-1)
    bias_array = (
        bias.detach().cpu().numpy()
        if isinstance(bias, torch.Tensor)
        else np.asarray(bias)
    ).reshape(-1)
    if not (raw_array.shape == weight_array.shape == bias_array.shape):
        raise ValueError("raw bias, weight, and bias rows must align")
    raw_stats = distribution(raw_array)
    raw_stats.update(
        {
            "fraction_gt_zero": float(np.mean(raw_array > 0.0)),
            "fraction_lt_zero": float(np.mean(raw_array < 0.0)),
        }
    )
    bias_stats = distribution(bias_array)
    bias_stats.update(
        {
            "fraction_gt_zero": float(np.mean(bias_array > 0.0)),
            "fraction_eq_zero": float(np.mean(bias_array == 0.0)),
        }
    )
    return {
        "bias_preactivation": raw_stats,
        "bias": bias_stats,
        "weight": distribution(weight_array),
    }


def dataset_output_diagnostics(
    model: HybridShareNet,
    dataset: Mapping[str, np.ndarray],
    device: torch.device,
) -> dict[str, object]:
    raw_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    bias_chunks: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        offsets = dataset["epoch_offsets"]
        for epoch_index in range(len(offsets) - 1):
            start = int(offsets[epoch_index])
            stop = int(offsets[epoch_index + 1])
            features = torch.as_tensor(
                dataset["features"][start:stop], dtype=torch.float32, device=device
            )
            raw, weight, bias = instrumented_forward(model, features)
            raw_chunks.append(raw[:, 1].detach().cpu())
            weight_chunks.append(weight.detach().cpu())
            bias_chunks.append(bias.detach().cpu())
    model.train(was_training)
    return output_diagnostics(
        torch.cat(raw_chunks), torch.cat(weight_chunks), torch.cat(bias_chunks)
    )


def initial_output_hash(
    model: HybridShareNet,
    dataset: Mapping[str, np.ndarray],
    device: torch.device,
) -> str:
    features = torch.as_tensor(
        dataset["features"], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        raw, weight, bias = instrumented_forward(model, features)
    return named_tensor_sha256(
        (("raw", raw), ("weight", weight), ("bias", bias))
    )


def full_dataset_loss_with_diagnostics(
    model: HybridShareNet,
    dataset: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    require_gradients: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Run the released objective and observe the already-computed outputs."""

    offsets = dataset["epoch_offsets"]
    if len(offsets) - 1 != EXPECTED_EPOCHS:
        raise RuntimeError("unexpected KLT3 training epoch count")
    raw_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    bias_chunks: list[torch.Tensor] = []
    loss: torch.Tensor | None = None
    maximum_wls_iterations = 0
    context = torch.enable_grad() if require_gradients else torch.no_grad()
    with context:
        for epoch_index in range(EXPECTED_EPOCHS):
            start = int(offsets[epoch_index])
            stop = int(offsets[epoch_index + 1])
            features = torch.as_tensor(
                dataset["features"][start:stop], dtype=torch.float32, device=device
            )
            raw, weight, bias = instrumented_forward(model, features)
            if not bool(torch.all(torch.isfinite(raw)).detach().cpu()):
                raise RuntimeError(f"non-finite raw model output at epoch {epoch_index}")
            raw_chunks.append(raw[:, 1].detach().cpu())
            weight_chunks.append(weight.detach().cpu())
            bias_chunks.append(bias.detach().cpu())
            solution = solve_paper_hybrid_position(
                dataset["satellite_positions_ecef_m"][start:stop],
                dataset["satellite_clock_bias_s"][start:stop],
                dataset["corrected_pseudorange_m"][start:stop],
                dataset["system_clock_indices"][start:stop],
                weight,
                bias,
                dataset["initial_states"][epoch_index],
                return_trace=True,
            )
            maximum_wls_iterations = max(
                maximum_wls_iterations, len(solution.iterations)
            )
            if not bool(torch.all(torch.isfinite(solution.state)).detach().cpu()):
                raise RuntimeError(f"non-finite WLS state at epoch {epoch_index}")
            epoch_loss = epoch_position_loss(
                solution.state,
                dataset["ground_truth_geodetic_deg_m"][epoch_index],
            )
            if not bool(torch.isfinite(epoch_loss).detach().cpu()):
                raise RuntimeError(f"non-finite position loss at epoch {epoch_index}")
            loss = epoch_loss if loss is None else loss + epoch_loss
    if loss is None:
        raise RuntimeError("empty KLT3 training dataset")
    diagnostics = output_diagnostics(
        torch.cat(raw_chunks), torch.cat(weight_chunks), torch.cat(bias_chunks)
    )
    diagnostics["maximum_wls_iterations"] = maximum_wls_iterations
    return loss, diagnostics


def _combined_gradient_norm(parameters: Sequence[torch.Tensor]) -> float:
    squared = 0.0
    for gradient in parameters:
        squared += float(torch.sum(gradient.detach() ** 2).cpu())
    return math.sqrt(squared)


def gradient_diagnostics(model: HybridShareNet) -> dict[str, object]:
    final = model.seq[-1]
    if not isinstance(final, nn.Linear) or final.out_features != 2:
        raise RuntimeError("unexpected HybridShareNet output layer")
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if any(gradient is None for gradient in gradients.values()):
        missing = [name for name, gradient in gradients.items() if gradient is None]
        raise RuntimeError(f"trainable parameters without gradients: {missing}")
    concrete = {name: value for name, value in gradients.items() if value is not None}
    finite = all(
        bool(torch.all(torch.isfinite(value)).detach().cpu())
        for value in concrete.values()
    )
    weight_row = [concrete["seq.7.weight"][0], concrete["seq.7.bias"][0]]
    bias_row = [concrete["seq.7.weight"][1], concrete["seq.7.bias"][1]]
    shared = [
        value for name, value in concrete.items() if not name.startswith("seq.7.")
    ]
    return {
        "bias_output_row_l2_norm": _combined_gradient_norm(bias_row),
        "weight_output_row_l2_norm": _combined_gradient_norm(weight_row),
        "shared_layers_l2_norm": _combined_gradient_norm(shared),
        "global_l2_norm": _combined_gradient_norm(list(concrete.values())),
        "all_trainable_gradients_present": len(concrete) == len(gradients),
        "all_trainable_gradients_finite": finite,
    }


def model_tensors_finite(model: nn.Module) -> bool:
    return all(
        bool(torch.all(torch.isfinite(value)).detach().cpu())
        for value in model.state_dict().values()
    )


def build_initial_snapshot(
    dataset: Mapping[str, np.ndarray],
    seed: int,
    device: torch.device,
) -> tuple[HybridShareNet, torch.optim.Adam, dict[str, object]]:
    model, optimizer = create_model_and_optimizer(dataset, seed, device)
    normalization = normalization_identity(dataset)
    np.testing.assert_array_equal(
        model.seq[0].mean.detach().cpu().numpy(),
        np.asarray(normalization["model_mean_float32_then_double"]),
    )
    np.testing.assert_array_equal(
        model.seq[0].std.detach().cpu().numpy(),
        np.asarray(normalization["model_std_float32_then_double"]),
    )
    configuration = scientific_configuration(model, optimizer)
    snapshot = {
        "seed": seed,
        "initial_parameter_sha256": parameter_hash(model),
        "initial_parameter_tensor_sha256": parameter_tensor_hashes(model),
        "initial_output_sha256": initial_output_hash(model, dataset, device),
        "architecture_sha256": configuration["architecture_sha256"],
        "optimizer": configuration["optimizer"],
        "configuration_sha256": canonical_json_hash(configuration),
        "normalization_sha256": normalization["sha256"],
    }
    return model, optimizer, snapshot


def validate_training_dataset(dataset: Mapping[str, np.ndarray]) -> None:
    if dataset["features"].shape != (EXPECTED_MEASUREMENTS, 3):
        raise RuntimeError("unexpected KLT3 feature matrix")
    if dataset["epoch_offsets"].shape != (EXPECTED_EPOCHS + 1,):
        raise RuntimeError("unexpected KLT3 epoch offsets")


__all__ = [
    "DEFAULT_SEED",
    "PREDEFINED_SEEDS",
    "TRAINING_EPOCHS",
    "architecture_identity",
    "build_initial_snapshot",
    "canonical_json_hash",
    "create_model",
    "create_model_and_optimizer",
    "dataset_output_diagnostics",
    "distribution",
    "file_sha256",
    "full_dataset_loss_with_diagnostics",
    "gradient_diagnostics",
    "initial_output_hash",
    "instrumented_forward",
    "load_dataset",
    "model_tensors_finite",
    "normalization_identity",
    "output_diagnostics",
    "parameter_hash",
    "parameter_tensor_hashes",
    "scientific_configuration",
    "training_data_identity",
    "validate_training_dataset",
]
