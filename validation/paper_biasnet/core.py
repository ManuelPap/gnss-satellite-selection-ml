"""Controlled reproduction of the released paper-era standalone BiasNet.

The executable target is TDL-GNSS commit ``dd5eac6``'s ``BiasNetTest``
class, not the unused BatchNorm-equipped ``BiasNet`` class.  Positive network
outputs are metre-valued corrections subtracted from the RTKLIB-corrected
pseudorange before the released equal-weight differentiable solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch
from torch import nn

try:
    from validation.paper_weightnet.core import (
        FEATURE_NAMES,
        FEATURE_UNITS,
        PaperWLSSolution,
        StandardizeLayer,
        construct_features,
        parameter_gradient_norm,
        solve_paper_weighted_position,
    )
except ModuleNotFoundError:  # Direct execution from validation/paper_biasnet.
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from validation.paper_weightnet.core import (
        FEATURE_NAMES,
        FEATURE_UNITS,
        PaperWLSSolution,
        StandardizeLayer,
        construct_features,
        parameter_gradient_norm,
        solve_paper_weighted_position,
    )


BIAS_UNIT = "metre"
BIAS_SIGN_CONVENTION = "corrected_pseudorange_m = pseudorange_m - predicted_bias_m"


class BiasNet(nn.Module):
    """Exact released ``BiasNetTest`` computational architecture.

    The historical class name is misleading: this is the class instantiated
    by both ``bias_network_train.py`` and ``bias_network_predict.py``.
    """

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
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.seq(value)


def instantiate_released_biasnet(
    input_mean: np.ndarray | Sequence[float],
    input_std: np.ndarray | Sequence[float],
    *,
    device: torch.device | str = "cpu",
) -> BiasNet:
    """Preserve the source's float32 construction followed by ``double()``."""

    mean = torch.tensor(input_mean, dtype=torch.float32)
    std = torch.tensor(input_std, dtype=torch.float32)
    model = BiasNet(mean, std)
    model.double()
    return model.to(device)


def subtract_predicted_bias(
    pseudorange_m: torch.Tensor | np.ndarray | Sequence[float],
    predicted_bias_m: torch.Tensor | np.ndarray | Sequence[float],
) -> torch.Tensor:
    """Apply the released sign convention in metres without clipping."""

    if isinstance(predicted_bias_m, torch.Tensor):
        device = predicted_bias_m.device
    elif isinstance(pseudorange_m, torch.Tensor):
        device = pseudorange_m.device
    else:
        device = torch.device("cpu")
    pseudorange = torch.as_tensor(
        pseudorange_m, dtype=torch.float64, device=device
    ).reshape(-1)
    bias = torch.as_tensor(
        predicted_bias_m, dtype=torch.float64, device=device
    ).reshape(-1)
    if pseudorange.shape != bias.shape:
        raise ValueError("pseudorange and predicted bias rows must align")
    if not bool(torch.all(torch.isfinite(pseudorange)).detach().cpu()):
        raise ValueError("pseudorange contains a non-finite value")
    if not bool(torch.all(torch.isfinite(bias)).detach().cpu()):
        raise ValueError("predicted bias contains a non-finite value")
    return pseudorange - bias


@dataclass(frozen=True)
class PaperBiasSolution:
    """Bias correction plus the released equal-weight positioning result."""

    raw_corrected_pseudorange_m: torch.Tensor
    predicted_bias_m: torch.Tensor
    bias_corrected_pseudorange_m: torch.Tensor
    wls: PaperWLSSolution

    @property
    def state(self) -> torch.Tensor:
        return self.wls.state

    @property
    def iterations(self) -> tuple[object, ...]:
        return self.wls.iterations

    @property
    def converged(self) -> bool:
        return self.wls.converged


def solve_paper_bias_position(
    satellite_positions_ecef_m: torch.Tensor | np.ndarray,
    satellite_clock_bias_s: torch.Tensor | np.ndarray,
    corrected_pseudorange_m: torch.Tensor | np.ndarray,
    system_clock_indices: torch.Tensor | np.ndarray,
    predicted_bias_m: torch.Tensor | np.ndarray,
    initial_state: torch.Tensor | np.ndarray,
    *,
    convergence_tolerance: float = 1.0e-4,
    maximum_iterations: int = 10,
    return_trace: bool = False,
) -> PaperBiasSolution:
    """Run ``P - b`` followed by the released identity-weight GNSS solve."""

    bias_corrected = subtract_predicted_bias(
        corrected_pseudorange_m, predicted_bias_m
    )
    count = bias_corrected.numel()
    weights = torch.ones(
        count, dtype=torch.float64, device=bias_corrected.device
    )
    wls = solve_paper_weighted_position(
        satellite_positions_ecef_m,
        satellite_clock_bias_s,
        bias_corrected,
        system_clock_indices,
        weights,
        initial_state,
        convergence_tolerance=convergence_tolerance,
        maximum_iterations=maximum_iterations,
        return_trace=return_trace,
    )
    raw = torch.as_tensor(
        corrected_pseudorange_m,
        dtype=torch.float64,
        device=bias_corrected.device,
    ).reshape(-1)
    bias = torch.as_tensor(
        predicted_bias_m, dtype=torch.float64, device=bias_corrected.device
    ).reshape(-1)
    return PaperBiasSolution(
        raw_corrected_pseudorange_m=raw,
        predicted_bias_m=bias,
        bias_corrected_pseudorange_m=bias_corrected,
        wls=wls,
    )


__all__ = [
    "BIAS_SIGN_CONVENTION",
    "BIAS_UNIT",
    "BiasNet",
    "FEATURE_NAMES",
    "FEATURE_UNITS",
    "PaperBiasSolution",
    "construct_features",
    "instantiate_released_biasnet",
    "parameter_gradient_norm",
    "solve_paper_bias_position",
    "subtract_predicted_bias",
]
