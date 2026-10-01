"""Independent reproduction of the paper-era WeightNet training core.

The equations and layer dimensions in this module are derived from TDL-GNSS
commit ``dd5eac669676ba0a922102047e58c2dfc9be9267``.  No upstream source is
vendored.  The intentionally unusual details (seven-state layout, explicit
inverse, sigmoid hidden layers, and weights scaled to 0--10) are compatibility
behaviour, not recommendations for new models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


SPEED_OF_LIGHT_M_S = 299_792_458.0
EARTH_ROTATION_RATE_RAD_S = 7.2921151467e-5
SYSTEM_TO_CLOCK_INDEX = {"G": 3, "C": 4, "E": 5, "R": 6}
FEATURE_NAMES = ("C/N0", "elevation", "OLS residual")
FEATURE_UNITS = ("dB-Hz-like SNR[0]/1000", "radian", "metre")


class StandardizeLayer(nn.Module):
    """Frozen affine standardization, matching the archived parameter layout."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std


class WeightNet(nn.Module):
    """Exact released ``dd5eac6`` WeightNet computational architecture."""

    def __init__(
        self,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if input_mean is None:
            input_mean = torch.tensor([0, 0, 0], dtype=torch.float64)
        if input_std is None:
            input_std = torch.tensor([1, 1, 1], dtype=torch.float64)
        self.seq = nn.Sequential(
            StandardizeLayer(input_mean, input_std),
            nn.Linear(3, 64),
            nn.Sigmoid(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.seq(value) * 10.0
        return torch.clamp(value, min=0.0, max=10.0)


def instantiate_released_weightnet(
    input_mean: np.ndarray | Sequence[float],
    input_std: np.ndarray | Sequence[float],
    *,
    device: torch.device | str = "cpu",
) -> WeightNet:
    """Instantiate with the archived float32-then-double conversion order."""

    mean = torch.tensor(input_mean, dtype=torch.float32)
    std = torch.tensor(input_std, dtype=torch.float32)
    model = WeightNet(mean, std)
    model.double()
    return model.to(device)


def construct_features(
    snr_scaled: np.ndarray | Sequence[float],
    elevation_rad: np.ndarray | Sequence[float],
    ols_residual_m: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Construct ``[SNR[0]/1000, elevation radians, OLS residual metres]``."""

    columns = [
        np.asarray(snr_scaled, dtype=np.float64).reshape(-1),
        np.asarray(elevation_rad, dtype=np.float64).reshape(-1),
        np.asarray(ols_residual_m, dtype=np.float64).reshape(-1),
    ]
    lengths = {column.size for column in columns}
    if len(lengths) != 1:
        raise ValueError("feature columns must have the same number of rows")
    features = np.column_stack(columns)
    if features.shape[0] == 0:
        raise ValueError("at least one feature row is required")
    if not np.all(np.isfinite(features)):
        raise ValueError("features contain a non-finite value")
    return features


@dataclass(frozen=True)
class PaperWLSIteration:
    predicted_observation_m: torch.Tensor
    jacobian_H: torch.Tensor
    residual_v_m: torch.Tensor
    weight_vector: torch.Tensor
    weight_matrix_W: torch.Tensor
    normal_matrix_HTWH: torch.Tensor
    rhs_HTWv: torch.Tensor
    delta_state: torch.Tensor
    updated_state: torch.Tensor
    active_state_indices: tuple[int, ...]


@dataclass(frozen=True)
class PaperWLSSolution:
    state: torch.Tensor
    iterations: tuple[PaperWLSIteration, ...]
    converged: bool


def _as_double_tensor(
    value: torch.Tensor | np.ndarray | Sequence[float],
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float64, device=device)
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def solve_paper_weighted_position(
    satellite_positions_ecef_m: torch.Tensor | np.ndarray,
    satellite_clock_bias_s: torch.Tensor | np.ndarray,
    corrected_pseudorange_m: torch.Tensor | np.ndarray,
    system_clock_indices: torch.Tensor | np.ndarray,
    weights: torch.Tensor | np.ndarray,
    initial_state: torch.Tensor | np.ndarray,
    *,
    convergence_tolerance: float = 1.0e-4,
    maximum_iterations: int = 10,
    return_trace: bool = False,
) -> PaperWLSSolution:
    """Run the released Torch observation model and explicit-inverse WLS.

    Atmospheric terms are zero because the archived Torch implementation sent
    an allocated but unpopulated line-of-sight vector to ``satazel``.  That
    behaviour was established by the preceding frozen real-KLT validation.
    """

    if isinstance(weights, torch.Tensor):
        device = weights.device
    elif isinstance(initial_state, torch.Tensor):
        device = initial_state.device
    else:
        device = torch.device("cpu")
    positions = _as_double_tensor(satellite_positions_ecef_m, device=device)
    clock_bias = _as_double_tensor(satellite_clock_bias_s, device=device).reshape(-1)
    pseudorange = _as_double_tensor(corrected_pseudorange_m, device=device).reshape(-1)
    clock_indices = torch.as_tensor(
        system_clock_indices, dtype=torch.int64, device=device
    ).reshape(-1)
    weight_vector = _as_double_tensor(weights, device=device).reshape(-1)
    state = _as_double_tensor(initial_state, device=device).reshape(-1)

    count = positions.shape[0]
    expected_vector = (count,)
    if positions.shape != (count, 3) or count < 4:
        raise ValueError("satellite positions must have shape (n>=4, 3)")
    for name, value in (
        ("satellite_clock_bias_s", clock_bias),
        ("corrected_pseudorange_m", pseudorange),
        ("system_clock_indices", clock_indices),
        ("weights", weight_vector),
    ):
        if tuple(value.shape) != expected_vector:
            raise ValueError(f"{name} must have shape {expected_vector}")
    if tuple(state.shape) != (7,):
        raise ValueError("initial_state must have shape (7,)")
    if not bool(torch.all((clock_indices >= 3) & (clock_indices <= 6))):
        raise ValueError("system clock indices must be in [3, 6]")
    for name, value in (
        ("positions", positions),
        ("clock_bias", clock_bias),
        ("pseudorange", pseudorange),
        ("weights", weight_vector),
        ("state", state),
    ):
        if not bool(torch.all(torch.isfinite(value)).detach().cpu()):
            raise ValueError(f"{name} contains a non-finite value")
    if not bool(torch.all(weight_vector > 0.0).detach().cpu()):
        raise ValueError("all weights must be strictly positive")

    active_indices = (0, 1, 2, *sorted(set(clock_indices.detach().cpu().tolist())))
    active = torch.tensor(active_indices, dtype=torch.int64, device=device)
    weight_matrix = torch.diag(weight_vector)
    delta = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64, device=device)
    diagnostics: list[PaperWLSIteration] = []
    iteration = 0

    while float(torch.linalg.vector_norm(delta).detach().cpu()) > convergence_tolerance:
        if iteration >= maximum_iterations:
            break
        difference = positions - state[:3]
        geometric_range = torch.linalg.vector_norm(difference, dim=1)
        sagnac = (
            EARTH_ROTATION_RATE_RAD_S
            * (positions[:, 0] * state[1] - positions[:, 1] * state[0])
            / SPEED_OF_LIGHT_M_S
        )
        corrected_range = geometric_range + sagnac
        jacobian_position = -difference / corrected_range[:, None]
        clock_columns = torch.zeros(
            (count, len(active_indices) - 3), dtype=torch.float64, device=device
        )
        for column, state_index in enumerate(active_indices[3:]):
            clock_columns[:, column] = (clock_indices == state_index).to(torch.float64)
        jacobian = torch.cat((jacobian_position, clock_columns), dim=1)
        predicted = (
            corrected_range
            + state[clock_indices]
            - SPEED_OF_LIGHT_M_S * clock_bias
        )
        residual = pseudorange - predicted

        weighted_transpose = jacobian.T @ weight_matrix
        normal = weighted_transpose @ jacobian
        rhs = weighted_transpose @ residual[:, None]
        inverse_times_transpose = torch.inverse(normal) @ jacobian.T
        gain = inverse_times_transpose @ weight_matrix
        delta = gain @ residual[:, None]

        updated = state.clone()
        updated[active] = updated[active] + delta.reshape(-1)
        if return_trace:
            diagnostics.append(
                PaperWLSIteration(
                    predicted_observation_m=predicted,
                    jacobian_H=jacobian,
                    residual_v_m=residual,
                    weight_vector=weight_vector,
                    weight_matrix_W=weight_matrix,
                    normal_matrix_HTWH=normal,
                    rhs_HTWv=rhs.reshape(-1),
                    delta_state=delta.reshape(-1),
                    updated_state=updated,
                    active_state_indices=active_indices,
                )
            )
        state = updated
        iteration += 1

    return PaperWLSSolution(
        state=state,
        iterations=tuple(diagnostics),
        converged=iteration < maximum_iterations,
    )


def parameter_gradient_norm(model: nn.Module) -> float:
    """Return the L2 norm over all present trainable parameter gradients."""

    total = 0.0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            gradient = parameter.grad.detach()
            total += float(torch.sum(gradient * gradient).cpu())
    return total**0.5


__all__ = [
    "EARTH_ROTATION_RATE_RAD_S",
    "FEATURE_NAMES",
    "FEATURE_UNITS",
    "PaperWLSIteration",
    "PaperWLSSolution",
    "SPEED_OF_LIGHT_M_S",
    "SYSTEM_TO_CLOCK_INDEX",
    "StandardizeLayer",
    "WeightNet",
    "construct_features",
    "instantiate_released_weightnet",
    "parameter_gradient_norm",
    "solve_paper_weighted_position",
]
