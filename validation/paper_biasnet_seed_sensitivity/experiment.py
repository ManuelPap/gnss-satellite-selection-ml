"""Shared machinery for the controlled corrected-GT BiasNet seed sweep.

The initialization seed is deliberately excluded from every scientific-control
hash.  It is the treatment variable; data, model, optimizer, loss, solver, and
evaluation settings are invariant.
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

from validation.paper_biasnet.core import (
    BiasNet,
    instantiate_released_biasnet,
    solve_paper_bias_position,
)
from validation.paper_biasnet.train_paper_biasnet import (
    DEFAULT_SEED,
    EXPECTED_EPOCHS,
    EXPECTED_MEASUREMENTS,
    RELEASED_LEARNING_RATE,
    load_dataset,
    set_seed,
)
from validation.paper_biasnet_corrected_gt.experiment import (
    MAXIMUM_WLS_ITERATIONS,
    WLS_TOLERANCE,
    corrected_target_indices,
    epoch_position_loss,
    validate_training_dataset,
)


PREDEFINED_SEEDS = tuple(range(10))
TRAINING_EPOCHS = 500
SELECTED_EPOCHS = (1, 2, 5, 10, 25, 50, 100, 200, 300, 400, 500)


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
    """Hash trainable parameters only, excluding normalization buffers."""

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
    """Identify all cached KLT3 arrays and the corrected one-to-one GT map."""

    array_hashes = {
        name: array_sha256(values) for name, values in sorted(dataset.items())
    }
    indices = corrected_target_indices(EXPECTED_EPOCHS)
    record: dict[str, object] = {
        "dataset": "KLT3",
        "epoch_count": int(dataset["ground_truth_geodetic_deg_m"].shape[0]),
        "measurement_count": int(dataset["features"].shape[0]),
        "feature_columns": int(dataset["features"].shape[1]),
        "cache_sha256": str(manifest["cache"]["sha256"]),
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
        "corrected_target_indices_sha256": array_sha256(indices),
        "ground_truth_mapping": "epoch i -> matched KLT3 GT row i",
        "unique_target_count": int(np.unique(indices).size),
        "duplicated_historical_gt_defect_present": False,
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


def architecture_identity(model: BiasNet) -> dict[str, object]:
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
    record: dict[str, object] = {
        "released_class": "BiasNetTest",
        "local_class": "validation.paper_biasnet.core.BiasNet",
        "dimensions": [3, 64, 128, 1],
        "hidden_activations": ["ReLU", "ReLU"],
        "output_activation": "linear",
        "output_clipping": False,
        "output_unit": "metre",
        "modules": modules,
        "parameters": parameters,
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
    model: BiasNet, optimizer: torch.optim.Optimizer
) -> dict[str, object]:
    """Return every scientific setting except the deliberately varied seed."""

    architecture = architecture_identity(model)
    first_parameter = next(model.parameters())
    return {
        "reference_commit": "dd5eac669676ba0a922102047e58c2dfc9be9267",
        "architecture_sha256": architecture["sha256"],
        "architecture": architecture,
        "execution": {
            "training_device": str(first_parameter.device),
            "model_dtype": str(first_parameter.dtype),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "seed_application": (
                "Python, NumPy, and Torch immediately before BiasNetTest construction"
            ),
        },
        "optimizer": optimizer_identity(optimizer),
        "training": {
            "epochs": TRAINING_EPOCHS,
            "dataset": "KLT3",
            "dataset_epochs": EXPECTED_EPOCHS,
            "measurement_rows": EXPECTED_MEASUREMENTS,
            "iteration_order": "chronological full dataset",
            "shuffle": False,
            "batch_configuration_read_but_unused": 128,
            "loss": "sum of per-epoch unsquared 3D ENU Euclidean norms",
            "ground_truth": "one aligned KLT3 geodetic target per GNSS epoch",
            "duplicated_historical_gt_defect_present": False,
            "features": ["SNR", "elevation", "OLS residual"],
            "feature_order": [0, 1, 2],
        },
        "solver": {
            "initialization": "per-epoch released OLS state",
            "maximum_iterations": MAXIMUM_WLS_ITERATIONS,
            "convergence_tolerance": WLS_TOLERANCE,
            "observation_model": "corrected_pseudorange_m - predicted_bias_m",
            "weighting": "identity",
        },
        "evaluation": {
            "datasets": ["KLT1", "KLT2"],
            "expected_cardinality": {
                "KLT1": {"epochs": 203, "measurements": 4676},
                "KLT2": {"epochs": 209, "measurements": 4914},
            },
            "model_state": "frozen after exactly 500 Adam updates",
            "mode": "eval with torch.no_grad and no test normalization",
            "procedure": "released timestamp/OLS filters and historical WLS predictor",
            "aggregation_2d": "mean_i(sqrt(E_i^2 + N_i^2))",
            "aggregation_3d": "mean_i(sqrt(E_i^2 + N_i^2 + U_i^2))",
        },
        "seed_included_in_configuration_hash": False,
    }


def create_model(
    dataset: Mapping[str, np.ndarray], seed: int, device: torch.device | str
) -> BiasNet:
    """Apply the seed immediately before released default initialization."""

    set_seed(seed)
    return instantiate_released_biasnet(
        dataset["features"].mean(axis=0),
        dataset["features"].std(axis=0),
        device=device,
    )


def create_model_and_optimizer(
    dataset: Mapping[str, np.ndarray], seed: int, device: torch.device | str
) -> tuple[BiasNet, torch.optim.Adam]:
    model = create_model(dataset, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RELEASED_LEARNING_RATE)
    return model, optimizer


def instrumented_forward(
    model: BiasNet, features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return both post-ReLU tensors and the unchanged linear model output."""

    if len(model.seq) != 6:
        raise RuntimeError("unexpected BiasNetTest module count")
    standardized = model.seq[0](features)
    hidden_1 = model.seq[2](model.seq[1](standardized))
    hidden_2 = model.seq[4](model.seq[3](hidden_1))
    output = model.seq[5](hidden_2)
    return hidden_1, hidden_2, output


def distribution(values: np.ndarray | torch.Tensor) -> dict[str, object]:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if not array.size:
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


def hidden_activation_diagnostics(
    values: np.ndarray | torch.Tensor,
) -> dict[str, object]:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("hidden activations must be a nonempty row-by-neuron matrix")
    stats = distribution(array)
    dead = np.all(array == 0.0, axis=0)
    stats.update(
        {
            "fraction_exactly_zero": float(np.mean(array == 0.0)),
            "fraction_positive": float(np.mean(array > 0.0)),
            "dead_neuron_count": int(dead.sum()),
            "dead_neuron_fraction": float(dead.mean()),
            "neuron_count": int(array.shape[1]),
            "measurement_row_count": int(array.shape[0]),
            "dead_neuron_definition": (
                "post-ReLU activation is exactly zero for every KLT3 row"
            ),
        }
    )
    return stats


def output_diagnostics(
    hidden_1: np.ndarray | torch.Tensor,
    hidden_2: np.ndarray | torch.Tensor,
    bias_output: np.ndarray | torch.Tensor,
) -> dict[str, object]:
    return {
        "hidden_layer_1": hidden_activation_diagnostics(hidden_1),
        "hidden_layer_2": hidden_activation_diagnostics(hidden_2),
        "bias_output": distribution(bias_output),
    }


def dataset_output_diagnostics(
    model: BiasNet,
    dataset: Mapping[str, np.ndarray],
    device: torch.device,
) -> dict[str, object]:
    hidden_1_chunks: list[torch.Tensor] = []
    hidden_2_chunks: list[torch.Tensor] = []
    output_chunks: list[torch.Tensor] = []
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
            hidden_1, hidden_2, output = instrumented_forward(model, features)
            hidden_1_chunks.append(hidden_1.detach().cpu())
            hidden_2_chunks.append(hidden_2.detach().cpu())
            output_chunks.append(output.detach().cpu())
    model.train(was_training)
    return output_diagnostics(
        torch.cat(hidden_1_chunks),
        torch.cat(hidden_2_chunks),
        torch.cat(output_chunks),
    )


def initial_output_hash(
    model: BiasNet,
    dataset: Mapping[str, np.ndarray],
    device: torch.device,
) -> str:
    hidden_1_chunks: list[torch.Tensor] = []
    hidden_2_chunks: list[torch.Tensor] = []
    output_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        offsets = dataset["epoch_offsets"]
        for epoch_index in range(len(offsets) - 1):
            start = int(offsets[epoch_index])
            stop = int(offsets[epoch_index + 1])
            features = torch.as_tensor(
                dataset["features"][start:stop], dtype=torch.float32, device=device
            )
            hidden_1, hidden_2, output = instrumented_forward(model, features)
            hidden_1_chunks.append(hidden_1.detach().cpu())
            hidden_2_chunks.append(hidden_2.detach().cpu())
            output_chunks.append(output.detach().cpu())
    return named_tensor_sha256(
        (
            ("hidden_1", torch.cat(hidden_1_chunks)),
            ("hidden_2", torch.cat(hidden_2_chunks)),
            ("output", torch.cat(output_chunks)),
        )
    )


def full_dataset_loss_with_diagnostics(
    model: BiasNet,
    dataset: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    require_gradients: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Evaluate the corrected objective and diagnose the exact forward tensors."""

    validate_training_dataset(dataset)
    offsets = dataset["epoch_offsets"]
    hidden_1_chunks: list[torch.Tensor] = []
    hidden_2_chunks: list[torch.Tensor] = []
    output_chunks: list[torch.Tensor] = []
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
            hidden_1, hidden_2, output = instrumented_forward(model, features)
            predicted_bias = output.squeeze(-1)
            if not bool(torch.all(torch.isfinite(predicted_bias)).detach().cpu()):
                raise RuntimeError(f"non-finite BiasNet output at epoch {epoch_index}")
            hidden_1_chunks.append(hidden_1.detach().cpu())
            hidden_2_chunks.append(hidden_2.detach().cpu())
            output_chunks.append(predicted_bias.detach().cpu())
            solution = solve_paper_bias_position(
                dataset["satellite_positions_ecef_m"][start:stop],
                dataset["satellite_clock_bias_s"][start:stop],
                dataset["corrected_pseudorange_m"][start:stop],
                dataset["system_clock_indices"][start:stop],
                predicted_bias,
                dataset["initial_states"][epoch_index],
                convergence_tolerance=WLS_TOLERANCE,
                maximum_iterations=MAXIMUM_WLS_ITERATIONS,
                return_trace=True,
            )
            maximum_wls_iterations = max(
                maximum_wls_iterations, len(solution.iterations)
            )
            if not bool(torch.all(torch.isfinite(solution.state)).detach().cpu()):
                raise RuntimeError(f"non-finite WLS state at epoch {epoch_index}")
            epoch_loss = epoch_position_loss(
                solution.state, dataset["ground_truth_geodetic_deg_m"][epoch_index]
            )
            if not bool(torch.isfinite(epoch_loss).detach().cpu()):
                raise RuntimeError(f"non-finite position loss at epoch {epoch_index}")
            loss = epoch_loss if loss is None else loss + epoch_loss
    if loss is None:
        raise RuntimeError("empty KLT3 training dataset")
    diagnostics = output_diagnostics(
        torch.cat(hidden_1_chunks),
        torch.cat(hidden_2_chunks),
        torch.cat(output_chunks),
    )
    diagnostics["maximum_wls_iterations"] = maximum_wls_iterations
    return loss, diagnostics


def _combined_gradient_norm(gradients: Sequence[torch.Tensor]) -> float:
    squared = sum(float(torch.sum(value.detach() ** 2).cpu()) for value in gradients)
    return math.sqrt(squared)


def gradient_diagnostics(
    model: BiasNet, *, detailed: bool = False
) -> dict[str, object]:
    expected_groups = {
        "first_hidden_layer_l2_norm": ("seq.1.weight", "seq.1.bias"),
        "second_hidden_layer_l2_norm": ("seq.3.weight", "seq.3.bias"),
        "final_output_layer_l2_norm": ("seq.5.weight", "seq.5.bias"),
    }
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = [name for name, gradient in gradients.items() if gradient is None]
    concrete = {name: value for name, value in gradients.items() if value is not None}
    finite = all(
        bool(torch.all(torch.isfinite(value)).detach().cpu())
        for value in concrete.values()
    )
    result: dict[str, object] = {
        name: _combined_gradient_norm([concrete[item] for item in parameter_names])
        if all(item in concrete for item in parameter_names)
        else None
        for name, parameter_names in expected_groups.items()
    }
    result.update(
        {
            "global_l2_norm": _combined_gradient_norm(list(concrete.values())),
            "all_trainable_gradients_present": not missing,
            "all_trainable_gradients_finite": finite and not missing,
            "missing_trainable_gradients": missing,
        }
    )
    if detailed:
        result["per_parameter"] = {
            name: {
                "l2_norm": float(torch.linalg.vector_norm(gradient).detach().cpu())
                if gradient is not None
                else None,
                "finite": bool(
                    gradient is not None
                    and torch.all(torch.isfinite(gradient)).detach().cpu()
                ),
            }
            for name, gradient in gradients.items()
        }
    return result


def parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def parameter_change_diagnostics(
    before: Mapping[str, torch.Tensor],
    model: nn.Module,
    *,
    detailed: bool = False,
) -> dict[str, object]:
    changes = {
        name: float(torch.max(torch.abs(parameter.detach() - before[name])).cpu())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    result: dict[str, object] = {
        "all_expected_trainable_tensors_changed": all(
            value > 0.0 for value in changes.values()
        ),
        "changed_tensor_count": sum(value > 0.0 for value in changes.values()),
        "expected_trainable_tensor_count": len(changes),
        "minimum_parameter_max_abs_change": min(changes.values()),
        "maximum_parameter_max_abs_change": max(changes.values()),
    }
    if detailed:
        result["per_parameter_max_abs_change"] = changes
    return result


def model_tensors_finite(model: nn.Module) -> bool:
    return all(
        bool(torch.all(torch.isfinite(value)).detach().cpu())
        for value in model.state_dict().values()
    )


def build_initial_snapshot(
    dataset: Mapping[str, np.ndarray],
    seed: int,
    device: torch.device,
) -> tuple[BiasNet, torch.optim.Adam, dict[str, object]]:
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


__all__ = [
    "DEFAULT_SEED",
    "PREDEFINED_SEEDS",
    "SELECTED_EPOCHS",
    "TRAINING_EPOCHS",
    "architecture_identity",
    "array_sha256",
    "build_initial_snapshot",
    "canonical_json_hash",
    "create_model",
    "create_model_and_optimizer",
    "dataset_output_diagnostics",
    "distribution",
    "file_sha256",
    "full_dataset_loss_with_diagnostics",
    "gradient_diagnostics",
    "hidden_activation_diagnostics",
    "initial_output_hash",
    "instrumented_forward",
    "load_dataset",
    "model_tensors_finite",
    "normalization_identity",
    "output_diagnostics",
    "parameter_change_diagnostics",
    "parameter_hash",
    "parameter_snapshot",
    "parameter_tensor_hashes",
    "scientific_configuration",
    "training_data_identity",
    "validate_training_dataset",
]
