import numpy as np
import torch

from gnss_satellite_selection_ml.differentiable_wls import (
    solve_wls_numpy,
    solve_wls_torch,
)
from gnss_satellite_selection_ml.synthetic import make_synthetic_epoch


def _initial_state() -> np.ndarray:
    return np.zeros(4, dtype=np.float64)


def test_torch_matches_independent_numpy_iteration_by_iteration() -> None:
    epoch = make_synthetic_epoch(seed=11, noise_std_m=0.8)
    initial_state = _initial_state()
    precisions = np.array([0.7, 1.1, 0.9, 1.4, 0.6, 1.3, 0.8, 1.2])

    numpy_solution = solve_wls_numpy(
        epoch.satellite_positions,
        epoch.pseudoranges,
        precisions,
        initial_state,
        iterations=6,
    )
    torch_solution = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        precisions,
        initial_state,
        iterations=6,
    )

    assert torch_solution.state.dtype == torch.float64
    assert len(torch_solution.iterations) == len(numpy_solution.iterations) == 6

    # Near convergence, a few pseudorange ULPs propagate into near-zero rhs and
    # update values, so each quantity gets a scale-appropriate absolute floor.
    field_tolerances = {
        "predicted_pseudorange": 1.5e-8,
        "residual": 1.5e-8,
        "jacobian": 1.0e-14,
        "precision": 0.0,
        "precision_matrix": 0.0,
        "normal_matrix": 1.0e-14,
        "rhs": 6.0e-8,
        "delta_state": 1.5e-8,
        "updated_state": 1.0e-8,
        "singular_values": 1.0e-14,
    }
    for torch_iteration, numpy_iteration in zip(
        torch_solution.iterations, numpy_solution.iterations, strict=True
    ):
        for field_name, absolute_tolerance in field_tolerances.items():
            torch_value = getattr(torch_iteration, field_name).detach().numpy()
            numpy_value = getattr(numpy_iteration, field_name)
            np.testing.assert_allclose(
                torch_value,
                numpy_value,
                rtol=2.0e-14,
                atol=absolute_tolerance,
                err_msg=field_name,
            )
        assert torch_iteration.rank == numpy_iteration.rank == 4
        np.testing.assert_allclose(
            torch_iteration.condition_number,
            numpy_iteration.condition_number,
            rtol=2.0e-14,
        )

    np.testing.assert_allclose(
        torch_solution.state.detach().numpy(),
        numpy_solution.state,
        rtol=2.0e-14,
        atol=5.0e-9,
    )


def test_synthetic_satellites_have_gnss_scale_and_known_state() -> None:
    epoch = make_synthetic_epoch(seed=4, noise_std_m=0.0)
    satellite_radii = np.linalg.norm(epoch.satellite_positions, axis=1)
    geometric_ranges = np.linalg.norm(
        epoch.satellite_positions - epoch.true_state[:3], axis=1
    )

    np.testing.assert_allclose(satellite_radii, 26_560_000.0, atol=1.0e-8)
    assert np.all(geometric_ranges > 19_000_000.0)
    assert np.all(geometric_ranges < 27_000_000.0)
    np.testing.assert_allclose(
        epoch.pseudoranges,
        geometric_ranges + epoch.true_state[3],
        atol=1.0e-10,
    )

