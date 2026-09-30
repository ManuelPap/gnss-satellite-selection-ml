import numpy as np
import torch

from gnss_satellite_selection_ml.differentiable_wls import solve_wls_torch
from gnss_satellite_selection_ml.synthetic import make_synthetic_epoch


def _initial_state() -> np.ndarray:
    return np.zeros(4, dtype=np.float64)


def test_global_precision_scale_invariance_and_gradient_null_direction() -> None:
    epoch = make_synthetic_epoch(seed=41, noise_std_m=0.9)
    initial_state = _initial_state()
    values = np.array([0.6, 1.3, 0.8, 1.1, 1.5, 0.7, 1.2, 0.9])

    base = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        values,
        initial_state,
        iterations=5,
    )
    scaled = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        37.0 * values,
        initial_state,
        iterations=5,
    )
    torch.testing.assert_close(base.state, scaled.state, rtol=1.0e-11, atol=1.0e-8)

    precisions = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    solution = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        precisions,
        initial_state,
        iterations=5,
    )
    truth = torch.as_tensor(epoch.true_state[:3], dtype=torch.float64)
    loss = torch.sum((solution.state[:3] - truth) ** 2)
    loss.backward()
    gradient = precisions.grad
    projection = torch.dot(gradient, precisions.detach())
    relative_projection = torch.abs(projection) / (
        torch.linalg.vector_norm(gradient)
        * torch.linalg.vector_norm(precisions.detach())
    )
    assert float(relative_projection) < 1.0e-9


def test_satellite_row_permutation_invariance() -> None:
    epoch = make_synthetic_epoch(seed=43, noise_std_m=0.8)
    initial_state = _initial_state()
    precisions = np.array([1.0, 0.6, 1.3, 0.8, 1.1, 0.7, 1.4, 0.9])
    permutation = np.array([5, 1, 7, 3, 0, 6, 2, 4])

    original = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        precisions,
        initial_state,
        iterations=5,
    )
    permuted = solve_wls_torch(
        epoch.satellite_positions[permutation],
        epoch.pseudoranges[permutation],
        precisions[permutation],
        initial_state,
        iterations=5,
    )
    torch.testing.assert_close(
        original.state, permuted.state, rtol=1.0e-11, atol=1.0e-8
    )


def test_reducing_bad_measurement_precision_reduces_position_influence() -> None:
    bad_index = 3
    epoch = make_synthetic_epoch(
        seed=47,
        noise_std_m=0.0,
        corrupted_indices=(bad_index,),
        corruption_m=20.0,
    )
    initial_state = _initial_state()
    equal_precision = np.ones(8)
    reduced_precision = equal_precision.copy()
    reduced_precision[bad_index] = 1.0e-3

    equal_solution = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        equal_precision,
        initial_state,
        iterations=6,
    )
    reduced_solution = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        reduced_precision,
        initial_state,
        iterations=6,
    )
    truth = torch.as_tensor(epoch.true_state[:3], dtype=torch.float64)
    equal_error = torch.linalg.vector_norm(equal_solution.state[:3] - truth)
    reduced_error = torch.linalg.vector_norm(reduced_solution.state[:3] - truth)

    assert float(reduced_error) < 0.02 * float(equal_error)

