import numpy as np
import torch

from gnss_satellite_selection_ml.differentiable_wls import (
    predict_pseudoranges_numpy,
    solve_wls_numpy,
    solve_wls_torch,
)
from gnss_satellite_selection_ml.synthetic import (
    make_precision_learning_dataset,
    make_synthetic_epoch,
)


def _loss_for_precisions(
    epoch,
    initial_state: np.ndarray,
    precisions: torch.Tensor,
) -> torch.Tensor:
    solution = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        precisions,
        initial_state,
        iterations=5,
    )
    truth = torch.as_tensor(epoch.true_state[:3], dtype=torch.float64)
    return torch.sum((solution.state[:3] - truth) ** 2)


def _finite_difference_gradient(
    epoch,
    initial_state: np.ndarray,
    precisions: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    gradient = np.empty_like(precisions)
    for index in range(precisions.size):
        plus = precisions.copy()
        minus = precisions.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        assert minus[index] > 0.0
        plus_loss = float(
            _loss_for_precisions(
                epoch, initial_state, torch.as_tensor(plus, dtype=torch.float64)
            )
        )
        minus_loss = float(
            _loss_for_precisions(
                epoch, initial_state, torch.as_tensor(minus, dtype=torch.float64)
            )
        )
        gradient[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
    return gradient


def test_autograd_precision_gradient_matches_central_finite_differences() -> None:
    epoch = make_synthetic_epoch(
        seed=23,
        noise_std_m=0.6,
        corrupted_indices=(2,),
        corruption_m=8.0,
    )
    initial_state = np.zeros(4, dtype=np.float64)
    precision_values = np.array([0.8, 1.2, 0.7, 1.4, 0.9, 1.1, 0.6, 1.3])
    precisions = torch.tensor(
        precision_values, dtype=torch.float64, requires_grad=True
    )

    loss = _loss_for_precisions(epoch, initial_state, precisions)
    loss.backward()
    autograd_gradient = precisions.grad.detach().numpy()

    assert np.all(np.isfinite(autograd_gradient))
    assert np.linalg.norm(autograd_gradient) > 0.0

    for epsilon in (1.0e-2, 1.0e-3, 1.0e-4):
        finite_difference = _finite_difference_gradient(
            epoch, initial_state, precision_values, epsilon
        )
        np.testing.assert_allclose(
            autograd_gradient,
            finite_difference,
            rtol=2.0e-4,
            atol=2.0e-5,
            err_msg=f"epsilon={epsilon}",
        )


def test_ground_truth_changes_only_the_loss() -> None:
    epoch = make_synthetic_epoch(seed=31)
    initial_state = np.zeros(4, dtype=np.float64)
    solution = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        epoch.precisions,
        initial_state,
        iterations=5,
    )
    state_before = solution.state.detach().clone()

    truth_a = torch.as_tensor(epoch.true_state[:3], dtype=torch.float64)
    truth_b = truth_a + torch.tensor([100.0, -50.0, 25.0], dtype=torch.float64)
    loss_a = torch.sum((solution.state[:3] - truth_a) ** 2)
    loss_b = torch.sum((solution.state[:3] - truth_b) ** 2)

    torch.testing.assert_close(solution.state.detach(), state_before)
    assert not torch.isclose(loss_a, loss_b)


def test_learning_features_are_reconstructed_without_ground_truth() -> None:
    learning_epoch = make_precision_learning_dataset(num_epochs=1, seed=101)[0]
    ols = solve_wls_numpy(
        learning_epoch.satellite_positions,
        learning_epoch.pseudoranges,
        np.ones(8, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        iterations=8,
    )
    expected_residual = (
        learning_epoch.pseudoranges
        - predict_pseudoranges_numpy(
            ols.state, learning_epoch.satellite_positions
        )
    )
    receiver = ols.state[:3]
    line_of_sight = learning_epoch.satellite_positions - receiver
    line_of_sight /= np.linalg.norm(line_of_sight, axis=1, keepdims=True)
    expected_elevation = np.arcsin(
        np.clip(line_of_sight @ (receiver / np.linalg.norm(receiver)), -1.0, 1.0)
    )

    np.testing.assert_allclose(learning_epoch.initial_state, ols.state)
    np.testing.assert_allclose(
        learning_epoch.features[:, 1], expected_elevation, atol=1.0e-15
    )
    np.testing.assert_allclose(
        learning_epoch.features[:, 2], expected_residual, atol=1.0e-15
    )

