#!/usr/bin/env python3
"""Compare the independent paper observation model with the KLT1 trace.

The implementation under test uses only NumPy and exported numerical inputs.
This validator neither imports TDL-GNSS nor calls ``H_matrix_prl_torch()`` or
``wls_solve_torch()``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gnss_satellite_selection_ml.paper_observation_model import (  # noqa: E402
    assert_exact_satellite_row_alignment,
    paper_wls_iteration,
)


# Every numerical check uses rtol=0 and the explicit absolute tolerance below.
TOLERANCES = {
    "geometric range": 1.0e-9,
    "Sagnac": 1.0e-12,
    "satellite-clock correction": 1.0e-9,
    "predicted observation": 1.0e-9,
    "residual v": 1.0e-9,
    "Jacobian H": 1.0e-15,
    "H^T W H": 5.0e-15,
    "H^T W v": 1.0e-12,
    "delta state": 1.0e-12,
    "updated state": 1.0e-12,
    "ionosphere delay (preserved defect)": 0.0,
    "troposphere delay (preserved defect)": 0.0,
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare the NumPy paper-era GPS model with the KLT1 trace."
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=here / "paper_epoch_trace.npz",
        help="Archived numerical trace (default: next to this script).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=here / "paper_epoch_manifest.json",
        help="Archived trace manifest (default: next to this script).",
    )
    return parser.parse_args()


def max_absolute_discrepancy(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(
        expected, dtype=np.float64
    )
    if difference.size == 0:
        return 0.0
    return float(np.max(np.abs(difference)))


def main() -> int:
    args = parse_args()
    if not args.trace.is_file():
        raise FileNotFoundError(f"trace does not exist: {args.trace}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {args.manifest}")

    manifest = json.loads(args.manifest.read_text())
    expected_ids = tuple(manifest["satellite_ids"])
    expected_rows = np.asarray(manifest["h_compact_rows"], dtype=np.int64)
    failures: list[str] = []

    print("Paper-era GPS observation-model comparison")
    print("All comparisons use rtol=0 and these absolute tolerances:")
    for name, tolerance in TOLERANCES.items():
        print(f"  {name:<42} {tolerance:.3e}")

    with np.load(args.trace, allow_pickle=False) as trace:
        satellite_ids = tuple(trace["satellite_ids"].tolist())
        try:
            assert_exact_satellite_row_alignment(
                satellite_ids,
                trace["h_compact_rows"],
                expected_ids,
                expected_rows,
            )
            assert_exact_satellite_row_alignment(
                satellite_ids,
                trace["residual_compact_rows"],
                expected_ids,
                expected_rows,
            )
        except ValueError as error:
            failures.append(str(error))

        common_inputs = {
            "satellite_positions_ecef_m": trace[
                "satellite_positions_ecef_m"
            ],
            "satellite_clock_bias_s": trace["satellite_clock_bias_s"],
            "corrected_pseudorange_m": trace["corrected_pseudorange_m"],
            "weights": trace["weight_diagonal"],
            "satellite_ids": satellite_ids,
        }
        previous_updated_state: np.ndarray | None = None

        for index in range(int(manifest["iterations"])):
            prefix = f"iteration_{index}_"
            h_rows = trace[prefix + "h_compact_rows"]
            residual_rows = trace[prefix + "residual_compact_rows"]
            try:
                assert_exact_satellite_row_alignment(
                    satellite_ids, h_rows, expected_ids, expected_rows
                )
                assert_exact_satellite_row_alignment(
                    satellite_ids, residual_rows, satellite_ids, h_rows
                )
            except ValueError as error:
                failures.append(f"iteration {index}: {error}")

            state_before = trace[prefix + "state_before"]
            if previous_updated_state is not None:
                chain_error = max_absolute_discrepancy(
                    state_before, previous_updated_state
                )
                if chain_error > TOLERANCES["updated state"]:
                    failures.append(
                        f"iteration {index}: state chain max_abs={chain_error:.17e}"
                    )

            iteration = paper_wls_iteration(
                **common_inputs,
                state=state_before,
                satellite_rows=h_rows,
            )
            observation = iteration.observation
            comparisons = {
                "geometric range": (
                    observation.geometric_range_m,
                    trace[prefix + "geometric_range_m"],
                ),
                "Sagnac": (observation.sagnac_m, trace[prefix + "sagnac_m"]),
                "satellite-clock correction": (
                    observation.satellite_clock_correction_m,
                    trace["satellite_clock_correction_m"],
                ),
                "predicted observation": (
                    observation.predicted_observation_m,
                    trace[prefix + "predicted_observation_m"].reshape(-1),
                ),
                "residual v": (
                    observation.residual_v_m,
                    trace[prefix + "effective_residual_m"].reshape(-1),
                ),
                "Jacobian H": (observation.jacobian_H, trace[prefix + "H"]),
                "H^T W H": (
                    iteration.normal_matrix_HTWH,
                    trace[prefix + "normal_matrix"],
                ),
                "H^T W v": (
                    iteration.rhs_HTWv,
                    trace[prefix + "rhs"].reshape(-1),
                ),
                "delta state": (
                    iteration.delta_state,
                    trace[prefix + "delta_state"].reshape(-1),
                ),
                "updated state": (
                    iteration.updated_state,
                    trace[prefix + "state_after"],
                ),
                "ionosphere delay (preserved defect)": (
                    observation.ionosphere_delay_m,
                    trace[prefix + "ionosphere_delay_m"],
                ),
                "troposphere delay (preserved defect)": (
                    observation.troposphere_delay_m,
                    trace[prefix + "troposphere_delay_m"],
                ),
            }

            print(f"iteration {index}:")
            for name, (actual, expected) in comparisons.items():
                discrepancy = max_absolute_discrepancy(actual, expected)
                tolerance = TOLERANCES[name]
                passed = discrepancy <= tolerance
                print(
                    f"  {'PASS' if passed else 'FAIL'} {name:<38} "
                    f"max_abs={discrepancy:.17e} tol={tolerance:.3e}"
                )
                if not passed:
                    failures.append(
                        f"iteration {index}: {name} max_abs={discrepancy:.17e} "
                        f"> {tolerance:.17e}"
                    )
            previous_updated_state = iteration.updated_state

    if failures:
        print("paper observation-model comparison FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("paper observation-model comparison PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
