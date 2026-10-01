"""Controlled NumPy reproduction of the paper-era GPS observation model.

This module is deliberately separate from :mod:`differentiable_wls`.  It
captures one historical, reference-compatible path and has no TDL-GNSS,
``pyrtklib``, or Torch runtime dependency.

Physical GNSS terms
-------------------

For satellite ECEF position ``s`` and receiver ECEF position ``r`` the model
uses the geometric range and first-order Earth-rotation correction

``rho = ||s - r||``

``S = OMEGA_E / c * (s_x * r_y - s_y * r_x)``.

The satellite clock bias in seconds is converted to a range correction as
``-c * delta_t_sat``.  The GPS receiver clock state is already expressed in
metres.

Paper-era implementation conventions
-------------------------------------

The state has seven entries
``[x, y, z, b_GPS, b_BDS, b_Galileo, b_GLONASS]``.  This GPS-only path uses
active columns ``[0, 1, 2, 3]``, predicts

``rho + S + b_GPS - c * delta_t_sat``

and defines the residual as corrected pseudorange minus prediction.  The
historical Jacobian is preserved exactly: its position numerator is
``r - s``, while its denominator is ``rho + S``.  It does not differentiate
the Sagnac expression.  Its GPS receiver-clock column is one.  The WLS helper
also preserves the historical explicit matrix inverse and adds the resulting
four-state update to the active slots of the seven-state vector.

Preserved legacy defects
------------------------

The archived Torch path passed an allocated but unpopulated line-of-sight
vector to its elevation/atmosphere calls.  Both ionosphere and troposphere
delays consequently became exactly zero.  This reference-compatible function
sets both arrays to zero intentionally.  It must not be used as a corrected
physical atmosphere model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


SPEED_OF_LIGHT_M_S = 299_792_458.0
"""Vacuum speed of light used by the archived RTKLIB path."""

EARTH_ROTATION_RATE_RAD_S = 7.2921151467e-5
"""Earth rotation rate used by the archived RTKLIB path."""

GPS_ACTIVE_STATE_INDICES = np.asarray([0, 1, 2, 3], dtype=np.int64)


@dataclass(frozen=True)
class PaperGPSObservation:
    """Decomposed paper-era observations in explicit satellite-row order."""

    satellite_ids: tuple[str, ...]
    satellite_rows: np.ndarray
    geometric_range_m: np.ndarray
    sagnac_m: np.ndarray
    satellite_clock_correction_m: np.ndarray
    receiver_clock_bias_m: np.ndarray
    ionosphere_delay_m: np.ndarray
    troposphere_delay_m: np.ndarray
    predicted_observation_m: np.ndarray
    residual_v_m: np.ndarray
    jacobian_H: np.ndarray


@dataclass(frozen=True)
class PaperWLSIteration:
    """Observation model and normal-equation products for one iteration."""

    observation: PaperGPSObservation
    weight_matrix_W: np.ndarray
    normal_matrix_HTWH: np.ndarray
    rhs_HTWv: np.ndarray
    delta_state: np.ndarray
    updated_state: np.ndarray


def _float_vector(
    name: str,
    value: np.ndarray | Sequence[float],
    length: int,
    *,
    allow_column: bool = False,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    allowed_shapes = {(length,)}
    if allow_column:
        allowed_shapes.add((length, 1))
    if result.shape not in allowed_shapes:
        expected = f"({length},)"
        if allow_column:
            expected += f" or ({length}, 1)"
        raise ValueError(f"{name} must have shape {expected}, got {result.shape}")
    result = result.reshape(length)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _satellite_ids(value: Sequence[str], length: int) -> tuple[str, ...]:
    result = tuple(str(item) for item in value)
    if len(result) != length:
        raise ValueError(
            f"satellite_ids must contain {length} entries, got {len(result)}"
        )
    if len(set(result)) != length:
        raise ValueError("satellite_ids must be unique")
    if any(not satellite_id.startswith("G") for satellite_id in result):
        raise ValueError("paper_observation_model is GPS-only")
    return result


def _satellite_rows(
    value: np.ndarray | Sequence[int] | None, length: int
) -> np.ndarray:
    if value is None:
        return np.arange(length, dtype=np.int64)
    raw = np.asarray(value)
    if raw.shape != (length,):
        raise ValueError(
            f"satellite_rows must have shape ({length},), got {raw.shape}"
        )
    if raw.dtype.kind not in "iu":
        raise ValueError("satellite_rows must contain integers")
    result = raw.astype(np.int64, copy=True)
    if np.any(result < 0):
        raise ValueError("satellite_rows must be non-negative")
    if np.unique(result).size != length:
        raise ValueError("satellite_rows must be unique")
    return result


def assert_exact_satellite_row_alignment(
    actual_satellite_ids: Sequence[str],
    actual_satellite_rows: np.ndarray | Sequence[int],
    expected_satellite_ids: Sequence[str],
    expected_satellite_rows: np.ndarray | Sequence[int],
) -> None:
    """Raise if satellite identity or row order differs from the reference.

    Equality is intentionally positional.  Sorting either side would conceal
    the legacy residual-row ordering hazard that this comparison audits.
    """

    actual_ids = tuple(str(item) for item in actual_satellite_ids)
    expected_ids = tuple(str(item) for item in expected_satellite_ids)
    actual_rows = np.asarray(actual_satellite_rows)
    expected_rows = np.asarray(expected_satellite_rows)
    if actual_ids != expected_ids or not np.array_equal(actual_rows, expected_rows):
        raise ValueError(
            "satellite-row alignment differs from the paper reference: "
            f"actual ids/rows={actual_ids}/{actual_rows.tolist()}, "
            f"expected ids/rows={expected_ids}/{expected_rows.tolist()}"
        )


def paper_observation_model(
    satellite_positions_ecef_m: np.ndarray | Sequence[Sequence[float]],
    satellite_clock_bias_s: np.ndarray | Sequence[float],
    corrected_pseudorange_m: np.ndarray | Sequence[float],
    state: np.ndarray | Sequence[float],
    satellite_ids: Sequence[str],
    *,
    satellite_rows: np.ndarray | Sequence[int] | None = None,
) -> PaperGPSObservation:
    """Evaluate the controlled, GPS-only paper-era observation model.

    All outputs are NumPy ``float64`` arrays.  No satellite filtering or
    sorting is performed: output row ``i`` refers to exactly input row ``i``.
    ``satellite_rows`` records that identity explicitly for external alignment
    checks.
    """

    positions = np.asarray(satellite_positions_ecef_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "satellite_positions_ecef_m must have shape (n_satellites, 3), "
            f"got {positions.shape}"
        )
    satellite_count = positions.shape[0]
    if satellite_count == 0:
        raise ValueError("at least one satellite observation is required")
    if not np.all(np.isfinite(positions)):
        raise ValueError("satellite_positions_ecef_m contains a non-finite value")

    clock_bias = _float_vector(
        "satellite_clock_bias_s", satellite_clock_bias_s, satellite_count
    )
    pseudorange = _float_vector(
        "corrected_pseudorange_m",
        corrected_pseudorange_m,
        satellite_count,
        allow_column=True,
    )
    state_n = _float_vector("state", state, 7)
    ids = _satellite_ids(satellite_ids, satellite_count)
    rows = _satellite_rows(satellite_rows, satellite_count)

    receiver_xyz = state_n[:3]
    satellite_minus_receiver = positions - receiver_xyz
    geometric_range = np.linalg.norm(satellite_minus_receiver, axis=1)
    if np.any(geometric_range <= 0.0):
        raise ValueError("receiver and satellite positions must be distinct")

    # Physical first-order Earth-rotation range correction, with the archived
    # sign and coordinate convention.
    sagnac = (
        EARTH_ROTATION_RATE_RAD_S
        / SPEED_OF_LIGHT_M_S
        * (
            positions[:, 0] * receiver_xyz[1]
            - positions[:, 1] * receiver_xyz[0]
        )
    )
    satellite_clock_correction = -SPEED_OF_LIGHT_M_S * clock_bias
    receiver_clock_bias = np.full(
        satellite_count, state_n[3], dtype=np.float64
    )

    # Preserved legacy defect: the archived elevation path yielded zero for
    # both atmosphere delays.  Do not insert a physical correction here.
    ionosphere_delay = np.zeros(satellite_count, dtype=np.float64)
    troposphere_delay = np.zeros(satellite_count, dtype=np.float64)

    corrected_range = geometric_range + sagnac
    if np.any(corrected_range <= 0.0):
        raise ValueError("paper-era corrected range must be strictly positive")
    predicted = (
        corrected_range
        + receiver_clock_bias
        + satellite_clock_correction
        + ionosphere_delay
        + troposphere_delay
    )
    residual = pseudorange - predicted

    # Historical convention, not the analytic Jacobian of the entire model:
    # geometric numerator, rho+S denominator, and no Sagnac derivatives.
    jacobian_position = -satellite_minus_receiver / corrected_range[:, None]
    jacobian = np.column_stack(
        (jacobian_position, np.ones(satellite_count, dtype=np.float64))
    )

    return PaperGPSObservation(
        satellite_ids=ids,
        satellite_rows=rows,
        geometric_range_m=geometric_range,
        sagnac_m=sagnac,
        satellite_clock_correction_m=satellite_clock_correction,
        receiver_clock_bias_m=receiver_clock_bias,
        ionosphere_delay_m=ionosphere_delay,
        troposphere_delay_m=troposphere_delay,
        predicted_observation_m=predicted,
        residual_v_m=residual,
        jacobian_H=jacobian,
    )


def paper_wls_iteration(
    satellite_positions_ecef_m: np.ndarray | Sequence[Sequence[float]],
    satellite_clock_bias_s: np.ndarray | Sequence[float],
    corrected_pseudorange_m: np.ndarray | Sequence[float],
    weights: np.ndarray | Sequence[float],
    state: np.ndarray | Sequence[float],
    satellite_ids: Sequence[str],
    *,
    satellite_rows: np.ndarray | Sequence[int] | None = None,
) -> PaperWLSIteration:
    """Execute one reference-compatible paper-era GPS WLS iteration.

    ``weights`` are the diagonal entries used directly in ``W``.  The update
    deliberately uses an explicit inverse to mirror the archived solver; this
    is a compatibility routine, not the recommended formulation for new code.
    """

    observation = paper_observation_model(
        satellite_positions_ecef_m,
        satellite_clock_bias_s,
        corrected_pseudorange_m,
        state,
        satellite_ids,
        satellite_rows=satellite_rows,
    )
    satellite_count = len(observation.satellite_ids)
    if satellite_count < 4:
        raise ValueError(
            f"at least four GPS observations are required, got {satellite_count}"
        )
    weight_diagonal = _float_vector("weights", weights, satellite_count)
    if np.any(weight_diagonal <= 0.0):
        raise ValueError("weights must be strictly positive")

    weight_matrix = np.diag(weight_diagonal)
    weighted_transpose = observation.jacobian_H.T @ weight_matrix
    normal_matrix = weighted_transpose @ observation.jacobian_H
    if np.linalg.matrix_rank(normal_matrix) < 4:
        raise ValueError("paper-era GPS normal matrix is rank deficient")
    rhs = weighted_transpose @ observation.residual_v_m[:, None]

    # Preserve the archived multiplication order around the explicit inverse.
    inverse_times_transpose = np.linalg.inv(normal_matrix) @ observation.jacobian_H.T
    gain = inverse_times_transpose @ weight_matrix
    delta_state = (gain @ observation.residual_v_m[:, None]).reshape(4)

    updated_state = _float_vector("state", state, 7).copy()
    updated_state[GPS_ACTIVE_STATE_INDICES] += delta_state
    return PaperWLSIteration(
        observation=observation,
        weight_matrix_W=weight_matrix,
        normal_matrix_HTWH=normal_matrix,
        rhs_HTWv=rhs.reshape(4),
        delta_state=delta_state,
        updated_state=updated_state,
    )


__all__ = [
    "EARTH_ROTATION_RATE_RAD_S",
    "GPS_ACTIVE_STATE_INDICES",
    "PaperGPSObservation",
    "PaperWLSIteration",
    "SPEED_OF_LIGHT_M_S",
    "assert_exact_satellite_row_alignment",
    "paper_observation_model",
    "paper_wls_iteration",
]
