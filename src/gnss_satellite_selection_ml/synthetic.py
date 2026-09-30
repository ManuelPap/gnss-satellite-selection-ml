"""Deterministic synthetic data for mathematical WLS validation.

The generated satellite ECEF positions lie on a nominal 26,560 km orbital
radius and are observed above the receiver's local horizon.  This is enough to
exercise realistic GNSS distance scales and geometry, but it is deliberately
not a complete GNSS simulator: satellite clocks, atmosphere, relativistic
effects, signal propagation and receiver physics are all omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .differentiable_wls import (
    predict_pseudoranges_numpy,
    solve_wls_numpy,
)


DEFAULT_RECEIVER_ECEF_M = np.array(
    [1_115_851.0, -4_842_952.0, 3_985_350.0], dtype=np.float64
)
GNSS_ORBIT_RADIUS_M = 26_560_000.0


@dataclass(frozen=True)
class SyntheticEpoch:
    """One deterministic synthetic single-constellation GNSS epoch."""

    true_state: np.ndarray
    satellite_positions: np.ndarray
    azimuth_rad: np.ndarray
    elevation_rad: np.ndarray
    exact_pseudoranges: np.ndarray
    pseudoranges: np.ndarray
    measurement_error: np.ndarray
    precisions: np.ndarray
    quality_indicator: np.ndarray
    corrupted_mask: np.ndarray
    constellation_ids: tuple[str, ...]


@dataclass(frozen=True)
class LearningEpoch:
    """Synthetic epoch plus paper-inspired features for precision learning."""

    satellite_positions: np.ndarray
    pseudoranges: np.ndarray
    true_state: np.ndarray
    initial_state: np.ndarray
    features: np.ndarray
    corrupted_mask: np.ndarray


def _local_basis(receiver_ecef_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = receiver_ecef_m / np.linalg.norm(receiver_ecef_m)
    earth_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    east = np.cross(earth_axis, up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    north /= np.linalg.norm(north)
    return east, north, up


def _elevation_from_receiver(
    receiver_ecef_m: np.ndarray,
    satellite_positions: np.ndarray,
) -> np.ndarray:
    receiver_norm = np.linalg.norm(receiver_ecef_m)
    if receiver_norm <= 0.0:
        raise ValueError("receiver ECEF position must be nonzero")
    line_of_sight = satellite_positions - receiver_ecef_m
    line_of_sight /= np.linalg.norm(line_of_sight, axis=1, keepdims=True)
    sine_elevation = line_of_sight @ (receiver_ecef_m / receiver_norm)
    return np.arcsin(np.clip(sine_elevation, -1.0, 1.0))


def _satellites_from_azimuth_elevation(
    receiver_ecef_m: np.ndarray,
    azimuth_rad: np.ndarray,
    elevation_rad: np.ndarray,
    *,
    orbit_radius_m: float = GNSS_ORBIT_RADIUS_M,
) -> np.ndarray:
    east, north, up = _local_basis(receiver_ecef_m)
    line_of_sight = (
        np.cos(elevation_rad)[:, None] * np.sin(azimuth_rad)[:, None] * east
        + np.cos(elevation_rad)[:, None] * np.cos(azimuth_rad)[:, None] * north
        + np.sin(elevation_rad)[:, None] * up
    )
    receiver_dot_los = line_of_sight @ receiver_ecef_m
    receiver_radius_squared = float(receiver_ecef_m @ receiver_ecef_m)
    distance = -receiver_dot_los + np.sqrt(
        receiver_dot_los**2 + orbit_radius_m**2 - receiver_radius_squared
    )
    satellites = receiver_ecef_m + distance[:, None] * line_of_sight
    return satellites.astype(np.float64)


def make_synthetic_epoch(
    *,
    seed: int = 7,
    receiver_ecef_m: np.ndarray | None = None,
    receiver_clock_bias_m: float = 75.0,
    noise_std_m: float = 0.7,
    azimuth_offset_rad: float = 0.0,
    angle_jitter_std_rad: float = 0.0,
    corrupted_indices: tuple[int, ...] = (),
    corruption_m: float = 20.0,
) -> SyntheticEpoch:
    """Construct one eight-satellite, well-distributed synthetic epoch."""

    rng = np.random.default_rng(seed)
    receiver = (
        DEFAULT_RECEIVER_ECEF_M.copy()
        if receiver_ecef_m is None
        else np.asarray(receiver_ecef_m, dtype=np.float64).reshape(3).copy()
    )
    base_azimuth_deg = np.array(
        [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
        dtype=np.float64,
    )
    base_elevation_deg = np.array(
        [22.0, 38.0, 55.0, 72.0, 30.0, 62.0, 45.0, 27.0],
        dtype=np.float64,
    )
    azimuth = np.deg2rad(base_azimuth_deg) + azimuth_offset_rad
    elevation = np.deg2rad(base_elevation_deg)
    if angle_jitter_std_rad > 0.0:
        azimuth += rng.normal(0.0, angle_jitter_std_rad, size=8)
        elevation += rng.normal(0.0, angle_jitter_std_rad, size=8)
        elevation = np.clip(elevation, np.deg2rad(12.0), np.deg2rad(82.0))

    satellites = _satellites_from_azimuth_elevation(
        receiver, azimuth, elevation
    )
    true_state = np.append(receiver, np.float64(receiver_clock_bias_m))
    exact_pseudoranges = predict_pseudoranges_numpy(true_state, satellites)
    measurement_error = rng.normal(0.0, noise_std_m, size=8).astype(np.float64)
    corrupted_mask = np.zeros(8, dtype=bool)
    for order, index in enumerate(corrupted_indices):
        if index < 0 or index >= 8:
            raise ValueError(f"corrupted satellite index out of range: {index}")
        corrupted_mask[index] = True
        sign = 1.0 if order % 2 == 0 else -1.0
        measurement_error[index] += sign * corruption_m

    quality = 0.82 + 0.10 * np.sin(elevation)
    quality += rng.normal(0.0, 0.025, size=8)
    quality[corrupted_mask] = 0.12 + rng.normal(
        0.0, 0.02, size=int(corrupted_mask.sum())
    )
    quality = np.clip(quality, 0.01, 0.99).astype(np.float64)

    return SyntheticEpoch(
        true_state=true_state,
        satellite_positions=satellites,
        azimuth_rad=azimuth.astype(np.float64),
        elevation_rad=elevation.astype(np.float64),
        exact_pseudoranges=exact_pseudoranges,
        pseudoranges=exact_pseudoranges + measurement_error,
        measurement_error=measurement_error,
        precisions=np.ones(8, dtype=np.float64),
        quality_indicator=quality,
        corrupted_mask=corrupted_mask,
        constellation_ids=("G",) * 8,
    )


def make_precision_learning_dataset(
    *,
    num_epochs: int,
    seed: int,
    corrupted_per_epoch: int = 2,
) -> tuple[LearningEpoch, ...]:
    """Build deterministic paper-inspired synthetic precision-learning data.

    ``quality_indicator`` is an abstract synthetic variable correlated with the
    injected measurement degradation.  It is not a physically modelled C/N0.
    OLS residuals are generated from an equal-precision solve, matching the
    conceptual feature pipeline in Hu et al.
    """

    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    if corrupted_per_epoch < 1 or corrupted_per_epoch >= 8:
        raise ValueError("corrupted_per_epoch must be between one and seven")

    rng = np.random.default_rng(seed)
    result: list[LearningEpoch] = []
    # The OLS feature solve must not receive ground truth. The ECEF origin is a
    # fixed convergence seed, not a position label or an epoch-specific prior.
    truth_independent_initial_state = np.zeros(4, dtype=np.float64)

    for _ in range(num_epochs):
        epoch_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        corrupted = tuple(
            int(value)
            for value in rng.choice(8, size=corrupted_per_epoch, replace=False)
        )
        epoch = make_synthetic_epoch(
            seed=epoch_seed,
            receiver_clock_bias_m=float(rng.uniform(30.0, 120.0)),
            noise_std_m=0.45,
            azimuth_offset_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
            angle_jitter_std_rad=np.deg2rad(2.0),
            corrupted_indices=corrupted,
            corruption_m=float(rng.uniform(14.0, 24.0)),
        )
        coarse_state = truth_independent_initial_state
        ols = solve_wls_numpy(
            epoch.satellite_positions,
            epoch.pseudoranges,
            np.ones(8, dtype=np.float64),
            coarse_state,
            iterations=8,
        )
        ols_residual = epoch.pseudoranges - predict_pseudoranges_numpy(
            ols.state, epoch.satellite_positions
        )
        ols_elevation = _elevation_from_receiver(
            ols.state[:3], epoch.satellite_positions
        )
        features = np.column_stack(
            (epoch.quality_indicator, ols_elevation, ols_residual)
        ).astype(np.float64)
        result.append(
            LearningEpoch(
                satellite_positions=epoch.satellite_positions,
                pseudoranges=epoch.pseudoranges,
                true_state=epoch.true_state,
                initial_state=ols.state,
                features=features,
                corrupted_mask=epoch.corrupted_mask,
            )
        )

    return tuple(result)

