"""Independent differentiable weighted least-squares GNSS positioning.

This module implements the four-state, one-constellation synthetic model

    state = [x, y, z, beta]

where position is Earth-centred, Earth-fixed (ECEF) in metres and ``beta`` is
receiver clock bias expressed as a range in metres.  The observation model is

    h_i(state) = ||satellite_i - position||_2 + beta

and the residual sign convention is ``v = observed - predicted``.

``precision_i`` means inverse measurement variance.  It appears once in the
normal equations as Lambda = diag(precision); it is not a square-root weight.
Satellite clocks, atmospheric delays and Earth-rotation corrections are
intentionally absent from this first mathematical validation milestone.

The iterative formulation follows Hu et al. and the paper-era TDL-GNSS
implementation.  This project uses ``torch.linalg.solve`` instead of forming
an explicit matrix inverse.  No pseudoinverse or fallback solution is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import torch


ArrayT = TypeVar("ArrayT")


class WLSGeometryError(RuntimeError):
    """Raised when the weighted normal system is rank deficient or unacceptable."""

    def __init__(
        self,
        message: str,
        *,
        rank: int | None = None,
        condition_number: float | None = None,
        singular_values: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.rank = rank
        self.condition_number = condition_number
        self.singular_values = singular_values


@dataclass(frozen=True)
class IterationDiagnostics(Generic[ArrayT]):
    """All algebraic quantities from one Gauss--Newton WLS iteration."""

    predicted_pseudorange: ArrayT
    residual: ArrayT
    jacobian: ArrayT
    precision: ArrayT
    precision_matrix: ArrayT
    normal_matrix: ArrayT
    rhs: ArrayT
    delta_state: ArrayT
    updated_state: ArrayT
    rank: int
    singular_values: ArrayT
    condition_number: float


@dataclass(frozen=True)
class WLSSolution(Generic[ArrayT]):
    """Final WLS state and the diagnostics for every executed iteration."""

    state: ArrayT
    iterations: tuple[IterationDiagnostics[ArrayT], ...]
    converged: bool


def _torch_input(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _torch_device(*values: Any) -> torch.device:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _validate_shapes(
    state_shape: tuple[int, ...],
    satellite_shape: tuple[int, ...],
    pseudorange_shape: tuple[int, ...],
    precision_shape: tuple[int, ...],
) -> int:
    if state_shape != (4,):
        raise ValueError(f"state must have shape (4,), got {state_shape}")
    if len(satellite_shape) != 2 or satellite_shape[1] != 3:
        raise ValueError(
            "satellite_positions must have shape (n_satellites, 3), "
            f"got {satellite_shape}"
        )
    n_satellites = satellite_shape[0]
    if n_satellites < 4:
        raise WLSGeometryError(
            f"at least four observations are required, got {n_satellites}",
            rank=min(n_satellites, 4),
        )
    if pseudorange_shape != (n_satellites,):
        raise ValueError(
            f"pseudoranges must have shape ({n_satellites},), got {pseudorange_shape}"
        )
    if precision_shape != (n_satellites,):
        raise ValueError(
            f"precisions must have shape ({n_satellites},), got {precision_shape}"
        )
    return n_satellites


def predict_pseudoranges_torch(
    state: torch.Tensor | np.ndarray,
    satellite_positions: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Return geometric pseudoranges plus receiver clock bias in metres."""

    device = _torch_device(state, satellite_positions)
    state_t = _torch_input(state, device=device).reshape(-1)
    satellites_t = _torch_input(satellite_positions, device=device)
    if state_t.shape != (4,):
        raise ValueError(f"state must have shape (4,), got {tuple(state_t.shape)}")
    if satellites_t.ndim != 2 or satellites_t.shape[1] != 3:
        raise ValueError("satellite_positions must have shape (n_satellites, 3)")
    geometric_range = torch.linalg.vector_norm(
        satellites_t - state_t[:3], dim=1
    )
    return geometric_range + state_t[3]


def predict_pseudoranges_numpy(
    state: np.ndarray,
    satellite_positions: np.ndarray,
) -> np.ndarray:
    """NumPy counterpart of :func:`predict_pseudoranges_torch`."""

    state_n = np.asarray(state, dtype=np.float64).reshape(-1)
    satellites_n = np.asarray(satellite_positions, dtype=np.float64)
    if state_n.shape != (4,):
        raise ValueError(f"state must have shape (4,), got {state_n.shape}")
    if satellites_n.ndim != 2 or satellites_n.shape[1] != 3:
        raise ValueError("satellite_positions must have shape (n_satellites, 3)")
    return np.linalg.norm(satellites_n - state_n[:3], axis=1) + state_n[3]


def _torch_linearize(
    state: torch.Tensor, satellite_positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    difference = state[:3] - satellite_positions
    geometric_range = torch.linalg.vector_norm(difference, dim=1)
    if bool(torch.any(geometric_range <= 0).detach().cpu()):
        raise WLSGeometryError("receiver and satellite positions must be distinct")
    predicted = geometric_range + state[3]
    jacobian_position = difference / geometric_range[:, None]
    jacobian = torch.cat(
        (jacobian_position, torch.ones_like(geometric_range[:, None])), dim=1
    )
    return predicted, jacobian


def _numpy_linearize(
    state: np.ndarray, satellite_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    difference = state[:3] - satellite_positions
    geometric_range = np.linalg.norm(difference, axis=1)
    if np.any(geometric_range <= 0):
        raise WLSGeometryError("receiver and satellite positions must be distinct")
    predicted = geometric_range + state[3]
    jacobian = np.column_stack(
        (difference / geometric_range[:, None], np.ones(geometric_range.shape[0]))
    )
    return predicted, jacobian


def _torch_normal_diagnostics(
    normal_matrix: torch.Tensor,
) -> tuple[int, torch.Tensor, float]:
    singular_values = torch.linalg.svdvals(normal_matrix)
    largest = singular_values[0]
    tolerance = (
        largest
        * normal_matrix.shape[0]
        * torch.finfo(normal_matrix.dtype).eps
    )
    rank = int(torch.count_nonzero(singular_values > tolerance).detach().cpu())
    smallest_value = float(singular_values[-1].detach().cpu())
    largest_value = float(largest.detach().cpu())
    condition_number = (
        largest_value / smallest_value if smallest_value > 0.0 else float("inf")
    )
    return rank, singular_values, condition_number


def _numpy_normal_diagnostics(
    normal_matrix: np.ndarray,
) -> tuple[int, np.ndarray, float]:
    singular_values = np.linalg.svd(normal_matrix, compute_uv=False)
    tolerance = (
        singular_values[0]
        * normal_matrix.shape[0]
        * np.finfo(normal_matrix.dtype).eps
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    return rank, singular_values, condition_number


def _raise_if_unacceptable(
    *,
    rank: int,
    condition_number: float,
    singular_values: Any,
    max_condition_number: float,
) -> None:
    if rank < 4:
        raise WLSGeometryError(
            f"rank-deficient normal matrix: rank {rank} < 4",
            rank=rank,
            condition_number=condition_number,
            singular_values=singular_values,
        )
    if not np.isfinite(condition_number) or condition_number > max_condition_number:
        raise WLSGeometryError(
            "numerically unacceptable normal matrix: "
            f"condition number {condition_number:.6e} exceeds "
            f"{max_condition_number:.6e}",
            rank=rank,
            condition_number=condition_number,
            singular_values=singular_values,
        )


def solve_wls_torch(
    satellite_positions: torch.Tensor | np.ndarray,
    pseudoranges: torch.Tensor | np.ndarray,
    precisions: torch.Tensor | np.ndarray,
    initial_state: torch.Tensor | np.ndarray,
    *,
    iterations: int = 6,
    convergence_tolerance: float | None = None,
    max_condition_number: float = 1.0e12,
) -> WLSSolution[torch.Tensor]:
    """Solve the synthetic GNSS WLS problem with differentiable Torch operations.

    ``convergence_tolerance=None`` executes exactly ``iterations`` iterations.
    Fixed iteration counts are required for finite-difference gradient tests so
    that every perturbation follows the same computational graph.
    """

    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if max_condition_number <= 1.0:
        raise ValueError("max_condition_number must be greater than one")

    device = _torch_device(precisions, initial_state, satellite_positions, pseudoranges)
    satellites_t = _torch_input(satellite_positions, device=device)
    pseudoranges_t = _torch_input(pseudoranges, device=device).reshape(-1)
    precisions_t = _torch_input(precisions, device=device).reshape(-1)
    state = _torch_input(initial_state, device=device).reshape(-1)

    _validate_shapes(
        tuple(state.shape),
        tuple(satellites_t.shape),
        tuple(pseudoranges_t.shape),
        tuple(precisions_t.shape),
    )
    for name, value in (
        ("state", state),
        ("satellite_positions", satellites_t),
        ("pseudoranges", pseudoranges_t),
        ("precisions", precisions_t),
    ):
        if not bool(torch.all(torch.isfinite(value)).detach().cpu()):
            raise ValueError(f"{name} contains a non-finite value")
    if bool(torch.any(precisions_t <= 0).detach().cpu()):
        raise ValueError("every measurement precision must be strictly positive")

    precision_matrix = torch.diag(precisions_t)
    diagnostics: list[IterationDiagnostics[torch.Tensor]] = []
    converged = False

    for _ in range(iterations):
        predicted, jacobian = _torch_linearize(state, satellites_t)
        residual = pseudoranges_t - predicted
        normal_matrix = jacobian.T @ precision_matrix @ jacobian
        rhs = jacobian.T @ precision_matrix @ residual
        rank, singular_values, condition_number = _torch_normal_diagnostics(
            normal_matrix
        )
        _raise_if_unacceptable(
            rank=rank,
            condition_number=condition_number,
            singular_values=singular_values.detach().cpu().numpy(),
            max_condition_number=max_condition_number,
        )
        try:
            delta_state = torch.linalg.solve(normal_matrix, rhs)
        except RuntimeError as error:
            raise WLSGeometryError(
                f"torch.linalg.solve failed: {error}",
                rank=rank,
                condition_number=condition_number,
                singular_values=singular_values.detach().cpu().numpy(),
            ) from error
        updated_state = state + delta_state
        diagnostics.append(
            IterationDiagnostics(
                predicted_pseudorange=predicted,
                residual=residual,
                jacobian=jacobian,
                precision=precisions_t,
                precision_matrix=precision_matrix,
                normal_matrix=normal_matrix,
                rhs=rhs,
                delta_state=delta_state,
                updated_state=updated_state,
                rank=rank,
                singular_values=singular_values,
                condition_number=condition_number,
            )
        )
        state = updated_state
        if convergence_tolerance is not None:
            delta_norm = float(torch.linalg.vector_norm(delta_state).detach().cpu())
            if delta_norm <= convergence_tolerance:
                converged = True
                break

    return WLSSolution(
        state=state,
        iterations=tuple(diagnostics),
        converged=converged,
    )


def solve_wls_numpy(
    satellite_positions: np.ndarray,
    pseudoranges: np.ndarray,
    precisions: np.ndarray,
    initial_state: np.ndarray,
    *,
    iterations: int = 6,
    convergence_tolerance: float | None = None,
    max_condition_number: float = 1.0e12,
) -> WLSSolution[np.ndarray]:
    """Independent NumPy float64 validation oracle for the same equations."""

    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if max_condition_number <= 1.0:
        raise ValueError("max_condition_number must be greater than one")

    satellites_n = np.asarray(satellite_positions, dtype=np.float64)
    pseudoranges_n = np.asarray(pseudoranges, dtype=np.float64).reshape(-1)
    precisions_n = np.asarray(precisions, dtype=np.float64).reshape(-1)
    state = np.asarray(initial_state, dtype=np.float64).reshape(-1).copy()
    _validate_shapes(
        state.shape,
        satellites_n.shape,
        pseudoranges_n.shape,
        precisions_n.shape,
    )
    for name, value in (
        ("state", state),
        ("satellite_positions", satellites_n),
        ("pseudoranges", pseudoranges_n),
        ("precisions", precisions_n),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains a non-finite value")
    if np.any(precisions_n <= 0):
        raise ValueError("every measurement precision must be strictly positive")

    precision_matrix = np.diag(precisions_n)
    diagnostics: list[IterationDiagnostics[np.ndarray]] = []
    converged = False

    for _ in range(iterations):
        predicted, jacobian = _numpy_linearize(state, satellites_n)
        residual = pseudoranges_n - predicted
        normal_matrix = jacobian.T @ precision_matrix @ jacobian
        rhs = jacobian.T @ precision_matrix @ residual
        rank, singular_values, condition_number = _numpy_normal_diagnostics(
            normal_matrix
        )
        _raise_if_unacceptable(
            rank=rank,
            condition_number=condition_number,
            singular_values=singular_values.copy(),
            max_condition_number=max_condition_number,
        )
        try:
            delta_state = np.linalg.solve(normal_matrix, rhs)
        except np.linalg.LinAlgError as error:
            raise WLSGeometryError(
                f"numpy.linalg.solve failed: {error}",
                rank=rank,
                condition_number=condition_number,
                singular_values=singular_values.copy(),
            ) from error
        updated_state = state + delta_state
        diagnostics.append(
            IterationDiagnostics(
                predicted_pseudorange=predicted,
                residual=residual,
                jacobian=jacobian,
                precision=precisions_n,
                precision_matrix=precision_matrix,
                normal_matrix=normal_matrix,
                rhs=rhs,
                delta_state=delta_state,
                updated_state=updated_state,
                rank=rank,
                singular_values=singular_values,
                condition_number=condition_number,
            )
        )
        state = updated_state
        if convergence_tolerance is not None:
            if np.linalg.norm(delta_state) <= convergence_tolerance:
                converged = True
                break

    return WLSSolution(
        state=state,
        iterations=tuple(diagnostics),
        converged=converged,
    )

