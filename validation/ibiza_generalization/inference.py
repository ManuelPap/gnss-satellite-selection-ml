"""Common frozen-network inference over the observation-only Ibiza dataset."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from validation.ibiza_generalization.adapters import raw_features_for_architecture
from validation.ibiza_generalization.checkpoints import (
    ARCHITECTURES,
    FrozenModel,
    REPOSITORY_ROOT,
    load_frozen_model,
    sha256_file,
)
from validation.ibiza_generalization.preprocess import validate_dataset_arrays
from validation.paper_biasnet.core import solve_paper_bias_position
from validation.paper_hybrid.core import solve_paper_hybrid_position
from validation.paper_weightnet.core import solve_paper_weighted_position


IBIZA_NPZ_SHA256 = (
    "edb0189e9eadf3266d75984e3041a90306dd44b3ebecc0101eb188f933dc88c5"
)
DEFAULT_IBIZA_NPZ = (
    REPOSITORY_ROOT.parent
    / "external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz"
)
SMOKE_EPOCH_INDICES = (0, 1, 2)


@dataclass(frozen=True)
class RowIdentity:
    source_row_index: tuple[int, ...]
    source_observation_index: tuple[int, ...]
    rtklib_satellite_number: tuple[int, ...]


@dataclass(frozen=True)
class NeuralOutputs:
    semantics: tuple[str, ...]
    predicted_bias_m: torch.Tensor | None
    predicted_weight: torch.Tensor | None


@dataclass(frozen=True)
class WLSIterationDiagnostic:
    iteration: int
    rank: int
    active_state_count: int
    full_rank: bool
    condition_number: float
    condition_finite: bool
    delta_state_norm: float


@dataclass(frozen=True)
class WLSDiagnostics:
    iterations: tuple[WLSIterationDiagnostic, ...]
    final_rank: int
    active_state_count: int
    rank_status: str
    final_condition_number: float
    conditioning_status: str
    converged: bool
    convergence_status: str
    solution_status: str


@dataclass(frozen=True)
class EpochInference:
    architecture: str
    seed: int
    accepted_epoch_index: int
    split_epoch_index: int
    epoch_time_gpst_like_s: float
    row_start: int
    row_stop: int
    row_identity: RowIdentity
    raw_features: torch.Tensor
    neural_outputs: NeuralOutputs
    receiver_state: torch.Tensor
    wls: WLSDiagnostics


@dataclass(frozen=True)
class ArchitectureInference:
    architecture: str
    seed: int
    checkpoint_sha256: str
    output_semantics: tuple[str, ...]
    checkpoint_parameters_unchanged: bool
    epochs: tuple[EpochInference, ...]


def load_ibiza_dataset(
    path: Path = DEFAULT_IBIZA_NPZ,
    *,
    expected_sha256: str = IBIZA_NPZ_SHA256,
) -> dict[str, np.ndarray]:
    """Hash-check and load the ground-truth-free, raw-feature Ibiza NPZ."""

    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Ibiza NPZ SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_dataset_arrays(arrays)
    return arrays


def _state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _state_is_identical(
    before: Mapping[str, torch.Tensor], model: nn.Module
) -> bool:
    after = model.state_dict()
    return tuple(before) == tuple(after) and all(
        torch.equal(before[name], after[name]) for name in before
    )


def infer_neural_outputs(
    model: nn.Module,
    architecture: str,
    raw_features: np.ndarray | torch.Tensor,
) -> NeuralOutputs:
    """Run one architecture in eval/inference mode without normalizing inputs."""

    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}")
    try:
        device = next(model.parameters()).device
    except StopIteration as error:  # These released networks always have parameters.
        raise RuntimeError("frozen model has no parameters") from error
    tensor = torch.as_tensor(raw_features, dtype=torch.float64, device=device)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError("raw features must have shape (rows, 3)")
    model.eval()
    with torch.inference_mode():
        prediction = model(tensor)
        if architecture == "TDL-B":
            return NeuralOutputs(
                semantics=("bias_m",),
                predicted_bias_m=prediction.reshape(-1).clone(),
                predicted_weight=None,
            )
        if architecture == "TDL-W":
            return NeuralOutputs(
                semantics=("weight",),
                predicted_bias_m=None,
                predicted_weight=prediction.reshape(-1).clone(),
            )
        weight, bias = prediction
        return NeuralOutputs(
            semantics=("weight", "bias_m"),
            predicted_bias_m=bias.reshape(-1).clone(),
            predicted_weight=weight.reshape(-1).clone(),
        )


def _epoch_slice(
    dataset: Mapping[str, np.ndarray], accepted_epoch_index: int
) -> slice:
    offsets = dataset["epoch_offsets"]
    epoch_count = offsets.size - 1
    if accepted_epoch_index < 0 or accepted_epoch_index >= epoch_count:
        raise IndexError(
            f"accepted epoch index {accepted_epoch_index} outside [0, {epoch_count})"
        )
    start = int(offsets[accepted_epoch_index])
    stop = int(offsets[accepted_epoch_index + 1])
    rows = slice(start, stop)
    if not np.all(dataset["epoch_index"][rows] == accepted_epoch_index):
        raise RuntimeError("epoch offsets and per-row epoch indices are misaligned")
    return rows


def _row_identity(
    dataset: Mapping[str, np.ndarray], rows: slice
) -> RowIdentity:
    return RowIdentity(
        source_row_index=tuple(
            int(value) for value in dataset["source_row_index"][rows]
        ),
        source_observation_index=tuple(
            int(value) for value in dataset["source_observation_index"][rows]
        ),
        rtklib_satellite_number=tuple(
            int(value) for value in dataset["rtklib_satellite_number"][rows]
        ),
    )


def _wls_diagnostics(solution: object) -> WLSDiagnostics:
    traces = tuple(solution.iterations)
    if not traces:
        raise RuntimeError("paper WLS returned no iteration diagnostics")
    diagnostics: list[WLSIterationDiagnostic] = []
    for number, trace in enumerate(traces, start=1):
        normal = trace.normal_matrix_HTWH.detach()
        active_count = len(trace.active_state_indices)
        rank = int(torch.linalg.matrix_rank(normal).cpu())
        condition = float(torch.linalg.cond(normal).cpu())
        diagnostics.append(
            WLSIterationDiagnostic(
                iteration=number,
                rank=rank,
                active_state_count=active_count,
                full_rank=rank == active_count,
                condition_number=condition,
                condition_finite=math.isfinite(condition),
                delta_state_norm=float(
                    torch.linalg.vector_norm(trace.delta_state).detach().cpu()
                ),
            )
        )
    final = diagnostics[-1]
    rank_status = "full_rank" if final.full_rank else "rank_deficient"
    conditioning_status = "finite" if final.condition_finite else "non_finite"
    converged = bool(solution.converged)
    convergence_status = "converged" if converged else "maximum_iterations"
    solution_status = (
        "solved"
        if final.full_rank and final.condition_finite and converged
        else "diagnostic_failure"
    )
    return WLSDiagnostics(
        iterations=tuple(diagnostics),
        final_rank=final.rank,
        active_state_count=final.active_state_count,
        rank_status=rank_status,
        final_condition_number=final.condition_number,
        conditioning_status=conditioning_status,
        converged=converged,
        convergence_status=convergence_status,
        solution_status=solution_status,
    )


def evaluate_epoch(
    frozen: FrozenModel,
    dataset: Mapping[str, np.ndarray],
    accepted_epoch_index: int,
) -> EpochInference:
    """Evaluate one accepted epoch without any ground-truth data route."""

    architecture = frozen.record.architecture
    rows = _epoch_slice(dataset, accepted_epoch_index)
    common_raw = raw_features_for_architecture(dataset, architecture)
    raw_features = torch.as_tensor(
        common_raw[rows], dtype=torch.float64, device="cpu"
    ).clone()
    output = infer_neural_outputs(frozen.model, architecture, raw_features)

    common_arguments = (
        dataset["satellite_position_ecef_m"][rows],
        dataset["satellite_clock_bias_s"][rows],
        dataset["corrected_pseudorange_m"][rows],
        dataset["system_clock_index"][rows],
    )
    initial_state = dataset["epoch_ols_initial_state"][accepted_epoch_index]
    with torch.inference_mode():
        if architecture == "TDL-B":
            paper_solution = solve_paper_bias_position(
                *common_arguments,
                output.predicted_bias_m,
                initial_state,
                return_trace=True,
            )
            wls_solution = paper_solution.wls
        elif architecture == "TDL-W":
            wls_solution = solve_paper_weighted_position(
                *common_arguments,
                output.predicted_weight,
                initial_state,
                return_trace=True,
            )
        else:
            paper_solution = solve_paper_hybrid_position(
                *common_arguments,
                output.predicted_weight,
                output.predicted_bias_m,
                initial_state,
                return_trace=True,
            )
            wls_solution = paper_solution.wls

    return EpochInference(
        architecture=architecture,
        seed=frozen.record.seed,
        accepted_epoch_index=accepted_epoch_index,
        split_epoch_index=int(dataset["epoch_split_index"][accepted_epoch_index]),
        epoch_time_gpst_like_s=float(
            dataset["epoch_time_gpst_like_s"][accepted_epoch_index]
        ),
        row_start=int(rows.start),
        row_stop=int(rows.stop),
        row_identity=_row_identity(dataset, rows),
        raw_features=raw_features,
        neural_outputs=output,
        receiver_state=wls_solution.state.detach().clone(),
        wls=_wls_diagnostics(wls_solution),
    )


def evaluate_architecture(
    frozen: FrozenModel,
    dataset: Mapping[str, np.ndarray],
    accepted_epoch_indices: Sequence[int],
) -> ArchitectureInference:
    """Evaluate an immutable model over an explicitly selected epoch subset."""

    before = _state_snapshot(frozen.model)
    epochs = tuple(
        evaluate_epoch(frozen, dataset, accepted_epoch_index)
        for accepted_epoch_index in accepted_epoch_indices
    )
    unchanged = _state_is_identical(before, frozen.model)
    if not unchanged:
        raise RuntimeError("frozen checkpoint parameters changed during inference")
    semantics = epochs[0].neural_outputs.semantics if epochs else ()
    return ArchitectureInference(
        architecture=frozen.record.architecture,
        seed=frozen.record.seed,
        checkpoint_sha256=frozen.record.sha256,
        output_semantics=semantics,
        checkpoint_parameters_unchanged=unchanged,
        epochs=epochs,
    )


def run_seed_zero_smoke(
    *,
    dataset_path: Path = DEFAULT_IBIZA_NPZ,
    accepted_epoch_indices: Sequence[int] = SMOKE_EPOCH_INDICES,
) -> tuple[ArchitectureInference, ...]:
    """Run only seed 0 of TDL-B/W/BW on one common deterministic subset."""

    indices = tuple(int(value) for value in accepted_epoch_indices)
    if not indices:
        raise ValueError("the smoke subset must contain at least one epoch")
    dataset = load_ibiza_dataset(dataset_path)
    results = tuple(
        evaluate_architecture(
            load_frozen_model(architecture, 0), dataset, indices
        )
        for architecture in ARCHITECTURES
    )
    reference = results[0].epochs
    for architecture_result in results[1:]:
        for expected, actual in zip(
            reference, architecture_result.epochs, strict=True
        ):
            if expected.row_identity != actual.row_identity or not torch.equal(
                expected.raw_features, actual.raw_features
            ):
                raise RuntimeError(
                    "architectures did not consume identical Ibiza rows/features"
                )
    return results


__all__ = [
    "ArchitectureInference",
    "DEFAULT_IBIZA_NPZ",
    "EpochInference",
    "IBIZA_NPZ_SHA256",
    "NeuralOutputs",
    "RowIdentity",
    "SMOKE_EPOCH_INDICES",
    "WLSDiagnostics",
    "WLSIterationDiagnostic",
    "evaluate_architecture",
    "evaluate_epoch",
    "infer_neural_outputs",
    "load_ibiza_dataset",
    "run_seed_zero_smoke",
]
