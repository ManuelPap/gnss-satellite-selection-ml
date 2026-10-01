import json
from pathlib import Path

import numpy as np
import pytest

from gnss_satellite_selection_ml.paper_observation_model import (
    EARTH_ROTATION_RATE_RAD_S,
    SPEED_OF_LIGHT_M_S,
    assert_exact_satellite_row_alignment,
    paper_observation_model,
    paper_wls_iteration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = REPOSITORY_ROOT / "validation/real_klt/paper_epoch_trace.npz"
MANIFEST_PATH = REPOSITORY_ROOT / "validation/real_klt/paper_epoch_manifest.json"

# Absolute tolerances are explicit and intentionally tighter than metre-scale
# model errors.  Relative tolerance is always zero for this frozen trace.
TRACE_TOLERANCES = {
    "geometric_range_m": 1.0e-9,
    "sagnac_m": 1.0e-12,
    "satellite_clock_correction_m": 1.0e-9,
    "predicted_observation_m": 1.0e-9,
    "residual_v_m": 1.0e-9,
    "jacobian_H": 1.0e-15,
    "normal_matrix_HTWH": 5.0e-15,
    "rhs_HTWv": 1.0e-12,
    "delta_state": 1.0e-12,
    "updated_state": 1.0e-12,
}


def _trace_inputs(trace: np.lib.npyio.NpzFile) -> dict[str, object]:
    return {
        "satellite_positions_ecef_m": trace["satellite_positions_ecef_m"],
        "satellite_clock_bias_s": trace["satellite_clock_bias_s"],
        "corrected_pseudorange_m": trace["corrected_pseudorange_m"],
        "weights": trace["weight_diagonal"],
        "satellite_ids": trace["satellite_ids"].tolist(),
    }


def test_observation_decomposition_preserves_zero_atmosphere() -> None:
    with np.load(TRACE_PATH, allow_pickle=False) as trace:
        inputs = _trace_inputs(trace)
        observation = paper_observation_model(
            inputs["satellite_positions_ecef_m"],
            inputs["satellite_clock_bias_s"],
            inputs["corrected_pseudorange_m"],
            trace["iteration_0_state_before"],
            inputs["satellite_ids"],
            satellite_rows=trace["iteration_0_h_compact_rows"],
        )

        assert np.array_equal(observation.ionosphere_delay_m, np.zeros(8))
        assert np.array_equal(observation.troposphere_delay_m, np.zeros(8))
        reconstructed = (
            observation.geometric_range_m
            + observation.sagnac_m
            + observation.satellite_clock_correction_m
            + observation.receiver_clock_bias_m
            + observation.ionosphere_delay_m
            + observation.troposphere_delay_m
        )
        np.testing.assert_allclose(
            reconstructed,
            observation.predicted_observation_m,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            observation.residual_v_m,
            trace["corrected_pseudorange_m"].reshape(-1) - reconstructed,
            rtol=0.0,
            atol=0.0,
        )


def test_satellite_clock_correction_has_negative_clock_bias_sign() -> None:
    positions = np.array(
        [[20_000_000.0, 1_000_000.0, 2_000_000.0],
         [21_000_000.0, 2_000_000.0, 3_000_000.0]]
    )
    clock_bias = np.array([2.0e-6, -3.0e-6])
    observation = paper_observation_model(
        positions,
        clock_bias,
        np.full(2, 25_000_000.0),
        np.zeros(7),
        ["G01", "G02"],
    )

    np.testing.assert_array_equal(
        observation.satellite_clock_correction_m,
        -SPEED_OF_LIGHT_M_S * clock_bias,
    )
    assert observation.satellite_clock_correction_m[0] < 0.0
    assert observation.satellite_clock_correction_m[1] > 0.0


def test_sagnac_uses_archived_sign_and_coordinate_convention() -> None:
    positions = np.array(
        [[2.0, 3.0, 10.0], [3.0, 2.0, 11.0]], dtype=np.float64
    )
    state = np.array([5.0, 7.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    observation = paper_observation_model(
        positions,
        np.zeros(2),
        np.full(2, 100.0),
        state,
        ["G01", "G02"],
    )
    expected = (
        EARTH_ROTATION_RATE_RAD_S
        / SPEED_OF_LIGHT_M_S
        * (positions[:, 0] * state[1] - positions[:, 1] * state[0])
    )

    np.testing.assert_array_equal(observation.sagnac_m, expected)
    assert observation.sagnac_m[0] < 0.0
    assert observation.sagnac_m[1] > 0.0


def test_gps_clock_jacobian_column_is_one() -> None:
    with np.load(TRACE_PATH, allow_pickle=False) as trace:
        inputs = _trace_inputs(trace)
        observation = paper_observation_model(
            inputs["satellite_positions_ecef_m"],
            inputs["satellite_clock_bias_s"],
            inputs["corrected_pseudorange_m"],
            trace["iteration_0_state_before"],
            inputs["satellite_ids"],
        )

    assert observation.jacobian_H.shape == (8, 4)
    np.testing.assert_array_equal(observation.jacobian_H[:, 3], np.ones(8))


def test_exact_satellite_row_alignment_matches_manifest_and_rejects_permutation() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    with np.load(TRACE_PATH, allow_pickle=False) as trace:
        ids = trace["satellite_ids"].tolist()
        rows = trace["h_compact_rows"]
        residual_rows = trace["residual_compact_rows"]

        assert_exact_satellite_row_alignment(
            ids, rows, manifest["satellite_ids"], manifest["h_compact_rows"]
        )
        assert_exact_satellite_row_alignment(ids, residual_rows, ids, rows)

        permutation = np.array([1, 0, 2, 3, 4, 5, 6, 7])
        with pytest.raises(ValueError, match="alignment differs"):
            assert_exact_satellite_row_alignment(
                np.asarray(ids)[permutation].tolist(),
                rows,
                manifest["satellite_ids"],
                manifest["h_compact_rows"],
            )


def test_both_paper_wls_iterations_match_archived_trace() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["iterations"] == 2

    with np.load(TRACE_PATH, allow_pickle=False) as trace:
        inputs = _trace_inputs(trace)
        for iteration_index in range(manifest["iterations"]):
            prefix = f"iteration_{iteration_index}_"
            iteration = paper_wls_iteration(
                **inputs,
                state=trace[prefix + "state_before"],
                satellite_rows=trace[prefix + "h_compact_rows"],
            )
            observation = iteration.observation
            comparisons = {
                "geometric_range_m": (
                    observation.geometric_range_m,
                    trace[prefix + "geometric_range_m"],
                ),
                "sagnac_m": (observation.sagnac_m, trace[prefix + "sagnac_m"]),
                "satellite_clock_correction_m": (
                    observation.satellite_clock_correction_m,
                    trace["satellite_clock_correction_m"],
                ),
                "predicted_observation_m": (
                    observation.predicted_observation_m,
                    trace[prefix + "predicted_observation_m"].reshape(-1),
                ),
                "residual_v_m": (
                    observation.residual_v_m,
                    trace[prefix + "effective_residual_m"].reshape(-1),
                ),
                "jacobian_H": (observation.jacobian_H, trace[prefix + "H"]),
                "normal_matrix_HTWH": (
                    iteration.normal_matrix_HTWH,
                    trace[prefix + "normal_matrix"],
                ),
                "rhs_HTWv": (
                    iteration.rhs_HTWv,
                    trace[prefix + "rhs"].reshape(-1),
                ),
                "delta_state": (
                    iteration.delta_state,
                    trace[prefix + "delta_state"].reshape(-1),
                ),
                "updated_state": (
                    iteration.updated_state,
                    trace[prefix + "state_after"],
                ),
            }
            for name, (actual, expected) in comparisons.items():
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=0.0,
                    atol=TRACE_TOLERANCES[name],
                    err_msg=f"iteration {iteration_index}: {name}",
                )

            assert_exact_satellite_row_alignment(
                observation.satellite_ids,
                observation.satellite_rows,
                trace["satellite_ids"].tolist(),
                trace[prefix + "residual_compact_rows"],
            )
