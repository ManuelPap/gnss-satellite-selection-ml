import numpy as np
import pytest

from gnss_satellite_selection_ml.differentiable_wls import (
    WLSGeometryError,
    predict_pseudoranges_numpy,
    solve_wls_numpy,
    solve_wls_torch,
)
from gnss_satellite_selection_ml.synthetic import make_synthetic_epoch


def test_good_geometry_reports_full_rank_and_finite_condition() -> None:
    epoch = make_synthetic_epoch(seed=51)
    initial = np.zeros(4, dtype=np.float64)
    solution = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        epoch.precisions,
        initial,
        iterations=1,
    )
    diagnostic = solution.iterations[0]
    assert diagnostic.rank == 4
    assert np.isfinite(diagnostic.condition_number)
    assert diagnostic.condition_number < 1.0e4


def test_duplicate_geometry_is_rejected_as_rank_deficient() -> None:
    epoch = make_synthetic_epoch(seed=53, noise_std_m=0.0)
    satellites = np.repeat(epoch.satellite_positions[:1], 8, axis=0)
    pseudoranges = predict_pseudoranges_numpy(epoch.true_state, satellites)

    with pytest.raises(WLSGeometryError, match="rank-deficient") as caught:
        solve_wls_torch(
            satellites,
            pseudoranges,
            np.ones(8),
            epoch.true_state,
            iterations=1,
        )
    assert caught.value.rank < 4


def test_nearly_singular_geometry_is_rejected_by_condition_number() -> None:
    epoch = make_synthetic_epoch(seed=59, noise_std_m=0.0)
    receiver = epoch.true_state[:3]
    base_direction = epoch.satellite_positions[0] - receiver
    base_direction /= np.linalg.norm(base_direction)
    tangent_a = np.cross(base_direction, np.array([0.0, 0.0, 1.0]))
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(base_direction, tangent_a)
    perturbations = np.linspace(-1.0, 1.0, 8)
    directions = (
        base_direction
        + 1.0e-2 * perturbations[:, None] * tangent_a
        + 5.0e-3 * perturbations[::-1, None] ** 2 * tangent_b
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    satellites = receiver + 22_000_000.0 * directions
    pseudoranges = predict_pseudoranges_numpy(epoch.true_state, satellites)

    with pytest.raises(WLSGeometryError, match="numerically unacceptable") as caught:
        solve_wls_torch(
            satellites,
            pseudoranges,
            np.ones(8),
            np.zeros(4, dtype=np.float64),
            iterations=1,
            max_condition_number=1.0e10,
        )
    assert caught.value.condition_number > 1.0e10


def test_exactly_determined_geometry_solves_without_pseudoinverse() -> None:
    epoch = make_synthetic_epoch(seed=61, noise_std_m=0.0)
    indices = np.array([0, 2, 5, 7])
    satellites = epoch.satellite_positions[indices]
    pseudoranges = epoch.pseudoranges[indices]
    initial = np.zeros(4, dtype=np.float64)

    solution = solve_wls_torch(
        satellites,
        pseudoranges,
        np.ones(4),
        initial,
        iterations=6,
    )
    assert solution.iterations[-1].rank == 4
    np.testing.assert_allclose(
        solution.state.detach().numpy(), epoch.true_state, atol=1.0e-7
    )


def test_common_very_small_precision_scale_remains_valid() -> None:
    epoch = make_synthetic_epoch(seed=67, noise_std_m=0.5)
    initial = np.zeros(4, dtype=np.float64)
    ordinary = solve_wls_numpy(
        epoch.satellite_positions,
        epoch.pseudoranges,
        np.ones(8),
        initial,
        iterations=5,
    )
    tiny = solve_wls_numpy(
        epoch.satellite_positions,
        epoch.pseudoranges,
        np.full(8, 1.0e-14),
        initial,
        iterations=5,
    )
    np.testing.assert_allclose(ordinary.state, tiny.state, rtol=1.0e-10, atol=1.0e-8)


def test_nonpositive_precision_is_rejected() -> None:
    epoch = make_synthetic_epoch(seed=71)
    invalid = np.ones(8)
    invalid[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        solve_wls_torch(
            epoch.satellite_positions,
            epoch.pseudoranges,
            invalid,
            epoch.true_state,
            iterations=1,
        )

