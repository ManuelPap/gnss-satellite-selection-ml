import json
from pathlib import Path

import numpy as np

from validation.paper_weightnet.core import solve_paper_weighted_position


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPOSITORY_ROOT / "validation/paper_weightnet"
PAPER_TRACE = REPOSITORY_ROOT / "validation/real_klt/paper_epoch_trace.npz"
EXPECTED_SATELLITES = ("G01", "G03", "G07", "G14", "G21", "G22", "G28", "G30")


def test_training_wls_core_matches_archived_real_klt1_trace() -> None:
    with np.load(PAPER_TRACE, allow_pickle=False) as trace:
        weights = trace["weight_diagonal"]
        solution = solve_paper_weighted_position(
            trace["satellite_positions_ecef_m"],
            trace["satellite_clock_bias_s"],
            trace["corrected_pseudorange_m"],
            np.full(8, 3, dtype=np.int64),
            weights,
            trace["iteration_0_state_before"],
            return_trace=True,
        )

        assert len(solution.iterations) == 2
        for index, iteration in enumerate(solution.iterations):
            prefix = f"iteration_{index}_"
            np.testing.assert_allclose(
                iteration.predicted_observation_m.detach().numpy(),
                trace[prefix + "predicted_observation_m"].reshape(-1),
                rtol=0.0,
                atol=1.0e-9,
            )
            np.testing.assert_allclose(
                iteration.jacobian_H.detach().numpy(),
                trace[prefix + "H"],
                rtol=0.0,
                atol=1.0e-15,
            )
            np.testing.assert_allclose(
                iteration.normal_matrix_HTWH.detach().numpy(),
                trace[prefix + "normal_matrix"],
                rtol=0.0,
                atol=5.0e-15,
            )
            np.testing.assert_allclose(
                iteration.rhs_HTWv.detach().numpy(),
                trace[prefix + "rhs"].reshape(-1),
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                iteration.updated_state.detach().numpy(),
                trace[prefix + "state_after"],
                rtol=0.0,
                atol=1.0e-12,
            )


def test_nn_weights_preserve_rows_and_controlled_wls_matches_paper_path() -> None:
    comparison = json.loads(
        (ARTIFACT_DIR / "klt1_nn_weight_comparison.json").read_text()
    )
    weights = comparison["weights"]

    assert comparison["status"] == "passed"
    assert tuple(weights) == EXPECTED_SATELLITES
    assert comparison["satellite_row_alignment_exact"] is True
    assert comparison["iterations"] == 2
    assert all(np.isfinite(value) and 0.0 <= value <= 10.0 for value in weights.values())
    discrepancies = comparison["maximum_absolute_discrepancies"]
    assert discrepancies["predicted_observation"] <= 1.0e-9
    assert discrepancies["H"] <= 1.0e-15
    assert discrepancies["residual_v"] <= 1.0e-9
    assert discrepancies["weight_vector"] == 0.0
    assert discrepancies["W"] == 0.0
    assert discrepancies["normal_HTWH"] <= 5.0e-15
    assert discrepancies["rhs_HTWv"] <= 1.0e-12
    assert discrepancies["delta_state"] <= 1.0e-10
    assert discrepancies["updated_state"] <= 1.0e-12
    assert comparison["final_state_maximum_absolute_discrepancy"] <= 1.0e-12


def test_real_klt3_weight_gradient_matches_finite_difference() -> None:
    sanity = json.loads((ARTIFACT_DIR / "gradient_sanity.json").read_text())

    assert sanity["status"] == "passed"
    assert sanity["wls_state_all_finite"] is True
    assert sanity["all_weight_gradients_finite"] is True
    assert sanity["at_least_one_weight_gradient_nonzero"] is True
    assert sanity["weight_gradient_l2_norm"] > 0.0
    assert sanity["maximum_symmetric_relative_error"] <= 1.0e-4
    assert len(sanity["finite_difference"]) == 3

