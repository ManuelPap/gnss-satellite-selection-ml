"""Released paper-era shared TDL-BW network and controlled hybrid solver.

The target is TDL-GNSS commit ``dd5eac6``.  The model returns a tuple in the
historical order ``(weight, bias)`` even though the paper writes ``(b, W)``.
Both outputs come from one shared network.  Positive metre-valued bias is
subtracted from the RTKLIB-corrected pseudorange and the dimensionless weight
is used once on the diagonal of the historical WLS normal equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from validation.paper_biasnet.core import subtract_predicted_bias
    from validation.paper_weightnet.core import (
        FEATURE_NAMES,
        FEATURE_UNITS,
        PaperWLSSolution,
        StandardizeLayer,
        construct_features,
        parameter_gradient_norm,
        solve_paper_weighted_position,
    )
except ModuleNotFoundError:  # Direct execution from this directory.
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from validation.paper_biasnet.core import subtract_predicted_bias
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
WEIGHT_UNIT = "dimensionless relative WLS coefficient"
OUTPUT_ORDER = ("weight", "bias")
RAW_COLUMN_TO_OUTPUT = {0: "weight", 1: "bias"}
BIAS_SIGN_CONVENTION = "corrected_pseudorange_m = pseudorange_m - predicted_bias_m"


class HybridShareNet(nn.Module):
    """Exact executable ``dd5eac6`` shared hybrid architecture."""

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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def raw_output(self, value: torch.Tensor) -> torch.Tensor:
        """Return the two pre-transformation columns for auditing."""

        return self.seq(value)

    @staticmethod
    def transform_raw(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError("raw hybrid output must have shape (rows, 2)")
        # Source order is semantically established by the trainer assigning
        # predict[0] to W and predict[1] to b before the WLS call.
        weight = torch.clamp(torch.sigmoid(raw[:, 0]), min=0.0, max=1.0)
        bias = F.relu(raw[:, 1])
        return weight, bias

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform_raw(self.raw_output(value))


def instantiate_released_hybrid(
    input_mean: np.ndarray | Sequence[float],
    input_std: np.ndarray | Sequence[float],
    *,
    device: torch.device | str = "cpu",
) -> HybridShareNet:
    """Preserve source construction as float32 followed by ``double()``."""

    mean = torch.tensor(input_mean, dtype=torch.float32)
    std = torch.tensor(input_std, dtype=torch.float32)
    model = HybridShareNet(mean, std)
    model.double()
    return model.to(device)


@dataclass(frozen=True)
class PaperHybridSolution:
    """Both neural mechanisms plus the historical WLS result."""

    raw_corrected_pseudorange_m: torch.Tensor
    predicted_bias_m: torch.Tensor
    bias_corrected_pseudorange_m: torch.Tensor
    predicted_weight: torch.Tensor
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


def solve_paper_hybrid_position(
    satellite_positions_ecef_m: torch.Tensor | np.ndarray,
    satellite_clock_bias_s: torch.Tensor | np.ndarray,
    corrected_pseudorange_m: torch.Tensor | np.ndarray,
    system_clock_indices: torch.Tensor | np.ndarray,
    predicted_weight: torch.Tensor | np.ndarray,
    predicted_bias_m: torch.Tensor | np.ndarray,
    initial_state: torch.Tensor | np.ndarray,
    *,
    convergence_tolerance: float = 1.0e-4,
    maximum_iterations: int = 10,
    return_trace: bool = False,
) -> PaperHybridSolution:
    """Apply ``P-b`` and ``W=diag(w)`` in the released solver."""

    bias_corrected = subtract_predicted_bias(
        corrected_pseudorange_m, predicted_bias_m
    )
    if isinstance(predicted_weight, torch.Tensor):
        device = predicted_weight.device
    else:
        device = bias_corrected.device
    weight = torch.as_tensor(
        predicted_weight, dtype=torch.float64, device=device
    ).reshape(-1)
    if weight.shape != bias_corrected.shape:
        raise ValueError("weight, bias, and pseudorange rows must align")
    if not bool(torch.all(torch.isfinite(weight)).detach().cpu()):
        raise ValueError("predicted weight contains a non-finite value")
    if not bool(torch.all(weight > 0.0).detach().cpu()):
        raise ValueError("released sigmoid hybrid weights must be strictly positive")
    wls = solve_paper_weighted_position(
        satellite_positions_ecef_m,
        satellite_clock_bias_s,
        bias_corrected,
        system_clock_indices,
        weight,
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
    return PaperHybridSolution(
        raw_corrected_pseudorange_m=raw,
        predicted_bias_m=bias,
        bias_corrected_pseudorange_m=bias_corrected,
        predicted_weight=weight,
        wls=wls,
    )


__all__ = [
    "BIAS_SIGN_CONVENTION",
    "BIAS_UNIT",
    "FEATURE_NAMES",
    "FEATURE_UNITS",
    "HybridShareNet",
    "OUTPUT_ORDER",
    "PaperHybridSolution",
    "RAW_COLUMN_TO_OUTPUT",
    "WEIGHT_UNIT",
    "construct_features",
    "instantiate_released_hybrid",
    "parameter_gradient_norm",
    "solve_paper_hybrid_position",
]
