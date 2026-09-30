"""Deterministic end-to-end validation of synthetic differentiable GNSS WLS.

This script first reruns compact numerical checks for the Phase A/B gate, then
trains a deliberately tiny network to predict bounded positive measurement
precisions from three synthetic features.  It is a mathematical experiment,
not a real-data GNSS performance claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from gnss_satellite_selection_ml.differentiable_wls import (
    WLSGeometryError,
    predict_pseudoranges_numpy,
    solve_wls_numpy,
    solve_wls_torch,
)
from gnss_satellite_selection_ml.synthetic import (
    LearningEpoch,
    make_precision_learning_dataset,
    make_synthetic_epoch,
)


WLS_ITERATIONS = 3
MIN_PRECISION = 0.02
MAX_PRECISION = 1.00


class TinyPrecisionNetwork(torch.nn.Module):
    """Map three standardized features to a bounded positive precision."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        unit_interval = torch.sigmoid(self.layers(features).squeeze(-1))
        return MIN_PRECISION + (MAX_PRECISION - MIN_PRECISION) * unit_interval


@dataclass(frozen=True)
class Evaluation:
    mean_position_error_m: float
    mean_squared_position_error_m2: float
    mean_clean_precision: float
    mean_corrupted_precision: float


def _standardization(
    dataset: tuple[LearningEpoch, ...],
) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([epoch.features for epoch in dataset], axis=0)
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def _precision(
    model: TinyPrecisionNetwork,
    epoch: LearningEpoch,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> torch.Tensor:
    standardized = (epoch.features - feature_mean) / feature_scale
    features = torch.as_tensor(standardized, dtype=torch.float64)
    return model(features)


def _position_loss(
    model: TinyPrecisionNetwork,
    epoch: LearningEpoch,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    precision = _precision(model, epoch, feature_mean, feature_scale)
    solution = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        precision,
        epoch.initial_state,
        iterations=WLS_ITERATIONS,
    )
    truth = torch.as_tensor(epoch.true_state[:3], dtype=torch.float64)
    squared_error = torch.sum((solution.state[:3] - truth) ** 2)
    return squared_error, precision


def _dataset_loss(
    model: TinyPrecisionNetwork,
    dataset: tuple[LearningEpoch, ...],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> torch.Tensor:
    losses = [
        _position_loss(model, epoch, feature_mean, feature_scale)[0]
        for epoch in dataset
    ]
    return torch.stack(losses).mean()


def _evaluate(
    model: TinyPrecisionNetwork,
    dataset: tuple[LearningEpoch, ...],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> tuple[Evaluation, float]:
    learned_errors: list[float] = []
    learned_squared_errors: list[float] = []
    equal_errors: list[float] = []
    clean_precisions: list[float] = []
    corrupted_precisions: list[float] = []

    with torch.no_grad():
        for epoch in dataset:
            squared_error, precision = _position_loss(
                model, epoch, feature_mean, feature_scale
            )
            learned_squared_errors.append(float(squared_error))
            learned_errors.append(float(torch.sqrt(squared_error)))

            equal_solution = solve_wls_torch(
                epoch.satellite_positions,
                epoch.pseudoranges,
                np.ones(8, dtype=np.float64),
                epoch.initial_state,
                iterations=WLS_ITERATIONS,
            )
            truth = torch.as_tensor(epoch.true_state[:3], dtype=torch.float64)
            equal_errors.append(
                float(torch.linalg.vector_norm(equal_solution.state[:3] - truth))
            )

            corrupted = torch.as_tensor(epoch.corrupted_mask)
            clean_precisions.extend(precision[~corrupted].tolist())
            corrupted_precisions.extend(precision[corrupted].tolist())

    evaluation = Evaluation(
        mean_position_error_m=float(np.mean(learned_errors)),
        mean_squared_position_error_m2=float(np.mean(learned_squared_errors)),
        mean_clean_precision=float(np.mean(clean_precisions)),
        mean_corrupted_precision=float(np.mean(corrupted_precisions)),
    )
    return evaluation, float(np.mean(equal_errors))


def _forward_oracle_report() -> dict[str, tuple[float, float]]:
    epoch = make_synthetic_epoch(seed=11, noise_std_m=0.8)
    initial = np.zeros(4, dtype=np.float64)
    precision = np.array([0.7, 1.1, 0.9, 1.4, 0.6, 1.3, 0.8, 1.2])
    numpy_solution = solve_wls_numpy(
        epoch.satellite_positions,
        epoch.pseudoranges,
        precision,
        initial,
        iterations=6,
    )
    torch_solution = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        precision,
        initial,
        iterations=6,
    )

    fields = (
        "predicted_pseudorange",
        "residual",
        "jacobian",
        "precision",
        "precision_matrix",
        "normal_matrix",
        "rhs",
        "delta_state",
        "updated_state",
        "singular_values",
    )
    report: dict[str, tuple[float, float]] = {}
    for field in fields:
        torch_values = np.concatenate(
            [
                getattr(iteration, field).detach().numpy().reshape(-1)
                for iteration in torch_solution.iterations
            ]
        )
        numpy_values = np.concatenate(
            [
                np.asarray(getattr(iteration, field)).reshape(-1)
                for iteration in numpy_solution.iterations
            ]
        )
        max_absolute = float(np.max(np.abs(torch_values - numpy_values)))
        reference_scale = float(np.max(np.abs(numpy_values)))
        max_relative = (
            max_absolute / reference_scale if reference_scale > 0.0 else 0.0
        )
        report[field] = (max_absolute, max_relative)

    final_difference = np.abs(
        torch_solution.state.detach().numpy() - numpy_solution.state
    )
    report["final_state"] = (
        float(np.max(final_difference)),
        float(np.max(final_difference) / np.max(np.abs(numpy_solution.state))),
    )
    return report


def _gradient_report() -> tuple[float, list[tuple[float, float, float]]]:
    epoch = make_synthetic_epoch(
        seed=23,
        noise_std_m=0.6,
        corrupted_indices=(2,),
        corruption_m=8.0,
    )
    initial = np.zeros(4, dtype=np.float64)
    values = np.array([0.8, 1.2, 0.7, 1.4, 0.9, 1.1, 0.6, 1.3])

    def loss_for(precision: torch.Tensor) -> torch.Tensor:
        solution = solve_wls_torch(
            epoch.satellite_positions,
            epoch.pseudoranges,
            precision,
            initial,
            iterations=5,
        )
        truth = torch.as_tensor(epoch.true_state[:3], dtype=torch.float64)
        return torch.sum((solution.state[:3] - truth) ** 2)

    precision = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    loss_for(precision).backward()
    autograd = precision.grad.detach().numpy()
    comparisons: list[tuple[float, float, float]] = []

    for epsilon in (1.0e-2, 1.0e-3, 1.0e-4):
        finite_difference = np.empty_like(values)
        for index in range(values.size):
            plus = values.copy()
            minus = values.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_loss = float(
                loss_for(torch.as_tensor(plus, dtype=torch.float64)).detach()
            )
            minus_loss = float(
                loss_for(torch.as_tensor(minus, dtype=torch.float64)).detach()
            )
            finite_difference[index] = (
                plus_loss - minus_loss
            ) / (2.0 * epsilon)
        maximum_absolute = float(np.max(np.abs(autograd - finite_difference)))
        scale = float(np.max(np.abs(finite_difference)))
        relative_infinity_norm = maximum_absolute / scale
        comparisons.append((epsilon, maximum_absolute, relative_infinity_norm))

    return float(np.linalg.norm(autograd)), comparisons


def _invariant_report() -> dict[str, float]:
    epoch = make_synthetic_epoch(seed=41, noise_std_m=0.9)
    initial = np.zeros(4, dtype=np.float64)
    values = np.array([0.6, 1.3, 0.8, 1.1, 1.5, 0.7, 1.2, 0.9])
    base = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        values,
        initial,
        iterations=5,
    )
    scaled = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        37.0 * values,
        initial,
        iterations=5,
    )
    scale_state_delta = float(torch.max(torch.abs(base.state - scaled.state)))

    differentiable = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    solution = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        differentiable,
        initial,
        iterations=5,
    )
    truth = torch.as_tensor(epoch.true_state[:3], dtype=torch.float64)
    torch.sum((solution.state[:3] - truth) ** 2).backward()
    gradient = differentiable.grad
    relative_projection = float(
        torch.abs(torch.dot(gradient, differentiable.detach()))
        / (
            torch.linalg.vector_norm(gradient)
            * torch.linalg.vector_norm(differentiable.detach())
        )
    )

    permutation = np.array([5, 1, 7, 3, 0, 6, 2, 4])
    permuted = solve_wls_torch(
        epoch.satellite_positions[permutation],
        epoch.pseudoranges[permutation],
        values[permutation],
        initial,
        iterations=5,
    )
    permutation_delta = float(torch.max(torch.abs(base.state - permuted.state)))

    bad_index = 3
    corrupted = make_synthetic_epoch(
        seed=47,
        noise_std_m=0.0,
        corrupted_indices=(bad_index,),
        corruption_m=20.0,
    )
    corrupted_initial = np.zeros(4, dtype=np.float64)
    equal = solve_wls_torch(
        corrupted.satellite_positions,
        corrupted.pseudoranges,
        np.ones(8),
        corrupted_initial,
        iterations=6,
    )
    downweighted_values = np.ones(8)
    downweighted_values[bad_index] = 1.0e-3
    downweighted = solve_wls_torch(
        corrupted.satellite_positions,
        corrupted.pseudoranges,
        downweighted_values,
        corrupted_initial,
        iterations=6,
    )
    corrupted_truth = torch.as_tensor(corrupted.true_state[:3])
    equal_error = float(
        torch.linalg.vector_norm(equal.state[:3] - corrupted_truth)
    )
    downweighted_error = float(
        torch.linalg.vector_norm(downweighted.state[:3] - corrupted_truth)
    )

    state_before_losses = base.state.detach().clone()
    truth_a = torch.as_tensor(epoch.true_state[:3])
    truth_b = truth_a + torch.tensor([100.0, -50.0, 25.0])
    _ = torch.sum((base.state[:3] - truth_a) ** 2)
    _ = torch.sum((base.state[:3] - truth_b) ** 2)
    leakage_state_delta = float(
        torch.max(torch.abs(base.state.detach() - state_before_losses))
    )

    return {
        "global_scale_state_delta_m": scale_state_delta,
        "gradient_scale_projection": relative_projection,
        "permutation_state_delta_m": permutation_delta,
        "equal_bad_measurement_error_m": equal_error,
        "downweighted_bad_measurement_error_m": downweighted_error,
        "truth_leakage_state_delta_m": leakage_state_delta,
    }


def _conditioning_report() -> dict[str, float]:
    epoch = make_synthetic_epoch(seed=51, noise_std_m=0.0)
    initial = np.zeros(4, dtype=np.float64)
    good = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        np.ones(8),
        initial,
        iterations=1,
    ).iterations[0]

    duplicates = np.repeat(epoch.satellite_positions[:1], 8, axis=0)
    duplicate_ranges = predict_pseudoranges_numpy(epoch.true_state, duplicates)
    try:
        solve_wls_torch(
            duplicates,
            duplicate_ranges,
            np.ones(8),
            epoch.true_state,
            iterations=1,
        )
    except WLSGeometryError as error:
        duplicate_rank = float(error.rank if error.rank is not None else np.nan)
    else:
        raise RuntimeError("duplicate geometry was not rejected")

    receiver = epoch.true_state[:3]
    base_direction = epoch.satellite_positions[0] - receiver
    base_direction /= np.linalg.norm(base_direction)
    tangent_a = np.cross(base_direction, np.array([0.0, 0.0, 1.0]))
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(base_direction, tangent_a)
    perturbation = np.linspace(-1.0, 1.0, 8)
    direction = (
        base_direction
        + 1.0e-2 * perturbation[:, None] * tangent_a
        + 5.0e-3 * perturbation[::-1, None] ** 2 * tangent_b
    )
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    near_satellites = receiver + 22_000_000.0 * direction
    near_ranges = predict_pseudoranges_numpy(epoch.true_state, near_satellites)
    try:
        solve_wls_torch(
            near_satellites,
            near_ranges,
            np.ones(8),
            np.zeros(4, dtype=np.float64),
            iterations=1,
            max_condition_number=1.0e10,
        )
    except WLSGeometryError as error:
        near_condition = float(error.condition_number)
        near_rank = float(error.rank)
    else:
        raise RuntimeError("nearly singular geometry was not rejected")

    exact_indices = np.array([0, 2, 5, 7])
    exact = solve_wls_torch(
        epoch.satellite_positions[exact_indices],
        epoch.pseudoranges[exact_indices],
        np.ones(4),
        initial,
        iterations=6,
    )
    exact_error = float(
        torch.linalg.vector_norm(
            exact.state[:3] - torch.as_tensor(epoch.true_state[:3])
        )
    )

    ordinary = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        np.ones(8),
        initial,
        iterations=5,
    )
    tiny = solve_wls_torch(
        epoch.satellite_positions,
        epoch.pseudoranges,
        np.full(8, 1.0e-14),
        initial,
        iterations=5,
    )
    tiny_scale_delta = float(torch.max(torch.abs(ordinary.state - tiny.state)))

    return {
        "good_rank": float(good.rank),
        "good_condition": good.condition_number,
        "duplicate_rank": duplicate_rank,
        "near_rank": near_rank,
        "near_condition": near_condition,
        "exact_rank": float(exact.iterations[-1].rank),
        "exact_position_error_m": exact_error,
        "tiny_scale_state_delta_m": tiny_scale_delta,
    }


def _learn(
    *,
    seed: int,
    epochs: int,
    train_size: int,
    test_size: int,
) -> tuple[
    TinyPrecisionNetwork,
    np.ndarray,
    np.ndarray,
    tuple[LearningEpoch, ...],
    float,
    float,
    list[float],
]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    train = make_precision_learning_dataset(num_epochs=train_size, seed=seed)
    test = make_precision_learning_dataset(num_epochs=test_size, seed=seed + 1)
    feature_mean, feature_scale = _standardization(train)

    model = TinyPrecisionNetwork().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial_loss = float(
        _dataset_loss(model, train, feature_mean, feature_scale).detach()
    )
    gradient_norms: list[float] = []

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = _dataset_loss(model, train, feature_mean, feature_scale)
        loss.backward()
        squared_gradient_norm = sum(
            float(torch.sum(parameter.grad.detach() ** 2))
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        gradient_norm = squared_gradient_norm**0.5
        if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError("training produced a non-finite or zero gradient")
        gradient_norms.append(gradient_norm)
        optimizer.step()

    final_loss = float(
        _dataset_loss(model, train, feature_mean, feature_scale).detach()
    )
    return (
        model,
        feature_mean,
        feature_scale,
        test,
        initial_loss,
        final_loss,
        gradient_norms,
    )


def _learned_scale_delta(
    model: TinyPrecisionNetwork,
    epoch: LearningEpoch,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> float:
    with torch.no_grad():
        precision = _precision(model, epoch, feature_mean, feature_scale)
        ordinary = solve_wls_torch(
            epoch.satellite_positions,
            epoch.pseudoranges,
            precision,
            epoch.initial_state,
            iterations=WLS_ITERATIONS,
        )
        scaled = solve_wls_torch(
            epoch.satellite_positions,
            epoch.pseudoranges,
            23.0 * precision,
            epoch.initial_state,
            iterations=WLS_ITERATIONS,
        )
    return float(torch.max(torch.abs(ordinary.state - scaled.state)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20_260_929)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--train-size", type=int, default=48)
    parser.add_argument("--test-size", type=int, default=24)
    arguments = parser.parse_args()

    print("Phase A/B compact numerical report")
    print("Torch versus independent NumPy (relative = max-abs / field max-abs):")
    for field, (absolute, relative) in _forward_oracle_report().items():
        print(f"  {field:24s} abs={absolute:.6e} rel={relative:.6e}")

    gradient_norm, gradient_rows = _gradient_report()
    print(f"Autograd precision-gradient norm: {gradient_norm:.6e}")
    for epsilon, absolute, relative in gradient_rows:
        print(
            f"  central FD eps={epsilon:.0e}: "
            f"max_abs={absolute:.6e} rel_inf={relative:.6e}"
        )

    print("Invariants:")
    for name, value in _invariant_report().items():
        print(f"  {name:38s} {value:.6e}")

    print("Conditioning:")
    for name, value in _conditioning_report().items():
        print(f"  {name:38s} {value:.6e}")

    print("Phase C synthetic precision learning")
    (
        model,
        feature_mean,
        feature_scale,
        test,
        initial_loss,
        final_loss,
        gradient_norms,
    ) = _learn(
        seed=arguments.seed,
        epochs=arguments.epochs,
        train_size=arguments.train_size,
        test_size=arguments.test_size,
    )
    learned, equal_error = _evaluate(
        model, test, feature_mean, feature_scale
    )
    scale_delta = _learned_scale_delta(
        model, test[0], feature_mean, feature_scale
    )
    improvement = 100.0 * (
        1.0 - learned.mean_position_error_m / equal_error
    )

    print(f"  seed:                                  {arguments.seed}")
    print(f"  epochs:                                {arguments.epochs}")
    print(f"  initial train mean squared error m^2:  {initial_loss:.6f}")
    print(f"  final train mean squared error m^2:    {final_loss:.6f}")
    print(f"  first/final gradient norm:             {gradient_norms[0]:.6e} / {gradient_norms[-1]:.6e}")
    print(f"  equal-precision test mean error m:     {equal_error:.6f}")
    print(f"  learned test mean error m:             {learned.mean_position_error_m:.6f}")
    print(f"  test mean squared error m^2:           {learned.mean_squared_position_error_m2:.6f}")
    print(f"  position-error improvement:            {improvement:.2f}%")
    print(f"  mean clean precision:                  {learned.mean_clean_precision:.6f}")
    print(f"  mean corrupted precision:              {learned.mean_corrupted_precision:.6f}")
    print(f"  common-scale state delta m:            {scale_delta:.6e}")

    checks = {
        "training loss declined": final_loss < initial_loss,
        "all training gradients finite/nonzero": all(
            np.isfinite(value) and value > 0.0 for value in gradient_norms
        ),
        "corrupted precision below clean": (
            learned.mean_corrupted_precision < learned.mean_clean_precision
        ),
        "learned error below equal precision": (
            learned.mean_position_error_m < equal_error
        ),
        "common precision scale is immaterial": scale_delta < 1.0e-7,
    }
    failed = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if failed:
        raise RuntimeError("acceptance checks failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
