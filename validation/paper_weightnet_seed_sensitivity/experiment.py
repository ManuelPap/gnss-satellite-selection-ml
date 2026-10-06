"""Shared machinery for the controlled paper-era WeightNet seed sweep.

The seed is deliberately excluded from every scientific-control hash: it is
the treatment variable.  Data, architecture, optimizer, objective, WLS, and
held-out evaluation settings are fixed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pymap3d as p3d
import torch
from torch import nn

from validation.paper_weightnet.core import (
    WeightNet,
    instantiate_released_weightnet,
    solve_paper_weighted_position,
)


REFERENCE_COMMIT = "dd5eac669676ba0a922102047e58c2dfc9be9267"
PREDEFINED_SEEDS = tuple(range(10))
HISTORICAL_SEED = 20_260_929
TRAINING_EPOCHS = 500
EXPECTED_EPOCHS = 405
EXPECTED_MEASUREMENTS = 8857
RELEASED_LEARNING_RATE = 0.01
WLS_TOLERANCE = 1.0e-4
MAXIMUM_WLS_ITERATIONS = 10
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
    encoded = label.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
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
    """Hash trainable parameters only, excluding frozen normalization tensors."""

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


def set_seed(seed: int) -> None:
    """Match the frozen trainer and seed immediately before construction."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(
    path: Path, manifest_path: Path, device: torch.device
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load and validate the frozen KLT3 cache without modifying it."""

    path = path.resolve()
    manifest_path = manifest_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"feature cache does not exist: {path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"feature manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    actual_hash = file_sha256(path)
    if actual_hash != manifest["cache"]["sha256"]:
        raise RuntimeError(
            f"feature cache hash mismatch: expected {manifest['cache']['sha256']}, "
            f"got {actual_hash}"
        )
    with np.load(path, allow_pickle=False) as cache:
        dataset = {name: cache[name].copy() for name in cache.files}
    validate_training_dataset(dataset)
    for name, values in dataset.items():
        if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
            raise RuntimeError(f"cached array {name} contains a non-finite value")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return dataset, manifest


def validate_training_dataset(dataset: Mapping[str, np.ndarray]) -> None:
    if dataset["features"].shape != (EXPECTED_MEASUREMENTS, 3):
        raise RuntimeError("unexpected KLT3 feature matrix")
    offsets = dataset["epoch_offsets"]
    if offsets.shape != (EXPECTED_EPOCHS + 1,):
        raise RuntimeError("unexpected KLT3 epoch offsets")
    if int(offsets[-1]) != EXPECTED_MEASUREMENTS:
        raise RuntimeError("unexpected KLT3 measurement count")
    if not np.all(np.diff(offsets) >= 4):
        raise RuntimeError("a KLT3 epoch has fewer than four measurements")
    if dataset["ground_truth_geodetic_deg_m"].shape != (EXPECTED_EPOCHS, 3):
        raise RuntimeError("unexpected KLT3 ground-truth mapping")


def training_data_identity(
    dataset: Mapping[str, np.ndarray], manifest: Mapping[str, object]
) -> dict[str, object]:
    """Identify every cached KLT3 array, including rows, features, and GT."""

    array_hashes = {
        name: array_sha256(values) for name, values in sorted(dataset.items())
    }
    record: dict[str, object] = {
        "dataset": "KLT3",
        "epoch_count": int(dataset["ground_truth_geodetic_deg_m"].shape[0]),
        "measurement_count": int(dataset["features"].shape[0]),
        "feature_columns": int(dataset["features"].shape[1]),
        "feature_order": ["C/N0", "elevation", "OLS residual"],
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
        "ground_truth_mapping": "one cached matched GT row per successful epoch",
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
        "fitted_on": "KLT3 only",
        "held_out_statistics_used": False,
    }


def architecture_identity(model: WeightNet) -> dict[str, object]:
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
        "historical_class": "model.WeightNet",
        "local_class": "validation.paper_weightnet.core.WeightNet",
        "reference_commit": REFERENCE_COMMIT,
        "dimensions": [3, 64, 128, 64, 1],
        "standardization": "frozen KLT3 population mean/std",
        "activations": ["Sigmoid", "Sigmoid", "Sigmoid", "Sigmoid"],
        "output_transform": "clamp(final_sigmoid * 10, min=0, max=10)",
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
    model: WeightNet, optimizer: torch.optim.Optimizer
) -> dict[str, object]:
    """Return all scientific settings except the deliberately varied seed."""

    architecture = architecture_identity(model)
    first_parameter = next(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "reference_commit": REFERENCE_COMMIT,
        "architecture_sha256": architecture["sha256"],
        "architecture": architecture,
        "execution": {
            "training_device": str(first_parameter.device),
            "model_dtype": str(first_parameter.dtype),
            "torch_default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "seed_application": (
                "Python, NumPy, Torch, and available CUDA RNGs immediately before "
                "WeightNet construction"
            ),
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "pymap3d": importlib.metadata.version("pymap3d"),
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda_version": torch.version.cuda,
            },
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
            "features": ["C/N0", "elevation", "OLS residual"],
            "feature_order": [0, 1, 2],
            "normalization": "KLT3 population mean/std only",
        },
        "solver": {
            "initialization": "per-epoch equal-weight released OLS state",
            "maximum_iterations": MAXIMUM_WLS_ITERATIONS,
            "convergence_tolerance": WLS_TOLERANCE,
            "observation_model": "historical released Torch WLS conventions",
            "weighting": "diag(WeightNet(features))",
        },
        "evaluation": {
            "datasets": ["KLT1", "KLT2"],
            "expected_cardinality": {
                "KLT1": {"epochs": 203, "measurements": 4676},
                "KLT2": {"epochs": 209, "measurements": 4914},
            },
            "model_state": "frozen after exactly 500 Adam updates",
            "mode": "eval with torch.no_grad and no test normalization fitting",
            "procedure": "released timestamp/OLS filters and historical WLS predictor",
            "aggregation_2d": "mean_i(sqrt(E_i^2 + N_i^2))",
            "aggregation_3d": "mean_i(sqrt(E_i^2 + N_i^2 + U_i^2))",
        },
        "seed_included_in_configuration_hash": False,
    }


def create_model(
    dataset: Mapping[str, np.ndarray], seed: int, device: torch.device | str
) -> WeightNet:
    """Apply the seed before the released default Linear initialization."""

    set_seed(seed)
    return instantiate_released_weightnet(
        dataset["features"].mean(axis=0),
        dataset["features"].std(axis=0),
        device=device,
    )


def create_model_and_optimizer(
    dataset: Mapping[str, np.ndarray], seed: int, device: torch.device | str
) -> tuple[WeightNet, torch.optim.Adam]:
    model = create_model(dataset, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RELEASED_LEARNING_RATE)
    return model, optimizer


def instrumented_forward(
    model: WeightNet, features: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Expose sigmoid inputs/outputs while preserving the exact forward graph."""

    if len(model.seq) != 9:
        raise RuntimeError("unexpected WeightNet module count")
    standardized = model.seq[0](features)
    pre_1 = model.seq[1](standardized)
    activation_1 = model.seq[2](pre_1)
    pre_2 = model.seq[3](activation_1)
    activation_2 = model.seq[4](pre_2)
    pre_3 = model.seq[5](activation_2)
    activation_3 = model.seq[6](pre_3)
    final_preactivation = model.seq[7](activation_3)
    final_sigmoid = model.seq[8](final_preactivation)
    weights = torch.clamp(final_sigmoid * 10.0, min=0.0, max=10.0)
    return weights, {
        "layer_1_preactivation": pre_1,
        "layer_1_activation": activation_1,
        "layer_2_preactivation": pre_2,
        "layer_2_activation": activation_2,
        "layer_3_preactivation": pre_3,
        "layer_3_activation": activation_3,
        "final_preactivation": final_preactivation,
        "final_sigmoid": final_sigmoid,
        "final_weight": weights,
    }


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


def sigmoid_layer_diagnostics(
    preactivation: np.ndarray | torch.Tensor,
    activation: np.ndarray | torch.Tensor,
) -> dict[str, object]:
    activation_array = (
        activation.detach().cpu().numpy()
        if isinstance(activation, torch.Tensor)
        else np.asarray(activation)
    )
    activation_array = np.asarray(activation_array, dtype=np.float64).reshape(-1)
    return {
        "preactivation": distribution(preactivation),
        "activation": {
            **distribution(activation_array),
            "fraction_lt_0_01": float(np.mean(activation_array < 0.01)),
            "fraction_gt_0_99": float(np.mean(activation_array > 0.99)),
            "threshold_purpose": "neural sigmoid saturation diagnostic",
        },
    }


def output_diagnostics(tensors: Mapping[str, np.ndarray | torch.Tensor]) -> dict[str, object]:
    weights = tensors["final_weight"]
    weight_array = (
        weights.detach().cpu().numpy()
        if isinstance(weights, torch.Tensor)
        else np.asarray(weights)
    )
    weight_array = np.asarray(weight_array, dtype=np.float64).reshape(-1)
    weight_stats = distribution(weight_array)
    weight_stats.update(
        {
            "fraction_lt_1e_5": float(np.mean(weight_array < 1.0e-5)),
            "fraction_lt_0_01": float(np.mean(weight_array < 0.01)),
            "fraction_gt_0_5": float(np.mean(weight_array > 0.5)),
            "fraction_gt_0_99": float(np.mean(weight_array > 0.99)),
            "gnss_threshold_purpose": "descriptive learned-weight thresholds",
            "fraction_lt_0_1": float(np.mean(weight_array < 0.1)),
            "fraction_gt_9_9": float(np.mean(weight_array > 9.9)),
            "sigmoid_saturation_equivalence": (
                "weight < 0.1 iff final sigmoid < 0.01; weight > 9.9 iff "
                "final sigmoid > 0.99 for finite released outputs"
            ),
        }
    )
    return {
        "hidden_sigmoid_1": sigmoid_layer_diagnostics(
            tensors["layer_1_preactivation"], tensors["layer_1_activation"]
        ),
        "hidden_sigmoid_2": sigmoid_layer_diagnostics(
            tensors["layer_2_preactivation"], tensors["layer_2_activation"]
        ),
        "hidden_sigmoid_3": sigmoid_layer_diagnostics(
            tensors["layer_3_preactivation"], tensors["layer_3_activation"]
        ),
        "final_sigmoid": sigmoid_layer_diagnostics(
            tensors["final_preactivation"], tensors["final_sigmoid"]
        ),
        "final_weight": weight_stats,
    }


def _concatenate_tensor_records(
    records: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not records:
        raise RuntimeError("no activation records were collected")
    return {
        name: torch.cat([record[name].detach().cpu() for record in records])
        for name in records[0]
    }


def dataset_output_diagnostics(
    model: WeightNet,
    dataset: Mapping[str, np.ndarray],
    device: torch.device,
) -> dict[str, object]:
    records: list[dict[str, torch.Tensor]] = []
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
            _weights, tensors = instrumented_forward(model, features)
            records.append(tensors)
    model.train(was_training)
    return output_diagnostics(_concatenate_tensor_records(records))


def initial_output_hash(
    model: WeightNet,
    dataset: Mapping[str, np.ndarray],
    device: torch.device,
) -> str:
    features = torch.as_tensor(dataset["features"], dtype=torch.float32, device=device)
    with torch.no_grad():
        _weights, tensors = instrumented_forward(model, features)
    return named_tensor_sha256(tensors.items())


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


def full_dataset_loss_with_diagnostics(
    model: WeightNet,
    dataset: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    require_gradients: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Run the frozen objective and observe tensors already in its forward graph."""

    validate_training_dataset(dataset)
    offsets = dataset["epoch_offsets"]
    activation_records: list[dict[str, torch.Tensor]] = []
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
            weights_matrix, tensors = instrumented_forward(model, features)
            weights = weights_matrix.squeeze(-1)
            if not bool(torch.all(torch.isfinite(weights)).detach().cpu()):
                raise RuntimeError(f"non-finite WeightNet output at epoch {epoch_index}")
            activation_records.append(tensors)
            solution = solve_paper_weighted_position(
                dataset["satellite_positions_ecef_m"][start:stop],
                dataset["satellite_clock_bias_s"][start:stop],
                dataset["corrected_pseudorange_m"][start:stop],
                dataset["system_clock_indices"][start:stop],
                weights,
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
        _concatenate_tensor_records(activation_records)
    )
    diagnostics["maximum_wls_iterations"] = maximum_wls_iterations
    return loss, diagnostics


def _combined_gradient_norm(gradients: Sequence[torch.Tensor]) -> float:
    squared = sum(float(torch.sum(value.detach() ** 2).cpu()) for value in gradients)
    return math.sqrt(squared)


def gradient_diagnostics(
    model: WeightNet, *, detailed: bool = False
) -> dict[str, object]:
    expected_groups = {
        "first_layer_l2_norm": ("seq.1.weight", "seq.1.bias"),
        "second_layer_l2_norm": ("seq.3.weight", "seq.3.bias"),
        "third_layer_l2_norm": ("seq.5.weight", "seq.5.bias"),
        "final_output_layer_l2_norm": ("seq.7.weight", "seq.7.bias"),
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
    dataset: Mapping[str, np.ndarray], seed: int, device: torch.device
) -> tuple[WeightNet, torch.optim.Adam, dict[str, object]]:
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
    "EXPECTED_EPOCHS",
    "EXPECTED_MEASUREMENTS",
    "HISTORICAL_SEED",
    "MAXIMUM_WLS_ITERATIONS",
    "PREDEFINED_SEEDS",
    "REFERENCE_COMMIT",
    "RELEASED_LEARNING_RATE",
    "SELECTED_EPOCHS",
    "TRAINING_EPOCHS",
    "WLS_TOLERANCE",
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
    "set_seed",
    "sigmoid_layer_diagnostics",
    "training_data_identity",
    "validate_training_dataset",
]
