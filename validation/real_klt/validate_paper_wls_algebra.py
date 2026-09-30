#!/usr/bin/env python3
"""Independently validate the exported paper-era WLS normal equations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


DEFAULT_NORMAL_TOL = 5e-15
DEFAULT_RHS_TOL = 1e-12
DEFAULT_DELTA_TOL = 1e-12
DEFAULT_STATE_TOL = 1e-12
DEFAULT_MAX_CONDITION = 1e12


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Validate every exported paper-era WLS iteration with independent "
            "NumPy float64 normal-equation calculations."
        )
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=here / "paper_epoch_trace.npz",
        help="Input NPZ trace (default: next to this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "algebra_validation.json",
        help="Output JSON report (default: next to this script).",
    )
    parser.add_argument("--normal-tol", type=float, default=DEFAULT_NORMAL_TOL)
    parser.add_argument("--rhs-tol", type=float, default=DEFAULT_RHS_TOL)
    parser.add_argument("--delta-tol", type=float, default=DEFAULT_DELTA_TOL)
    parser.add_argument("--state-tol", type=float, default=DEFAULT_STATE_TOL)
    parser.add_argument(
        "--max-condition", type=float, default=DEFAULT_MAX_CONDITION
    )
    return parser.parse_args()


def discrepancy(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    max_abs = float(np.max(np.abs(difference)))
    scale = float(max(1.0, np.max(np.abs(reference))))
    return {"max_abs": max_abs, "relative_max_scale": max_abs / scale}


def iteration_indices(keys: list[str]) -> list[int]:
    pattern = re.compile(r"iteration_(\d+)_H$")
    indices = sorted(
        int(match.group(1))
        for key in keys
        if (match := pattern.fullmatch(key)) is not None
    )
    if not indices:
        raise ValueError("trace contains no iteration_<n>_H arrays")
    if indices != list(range(len(indices))):
        raise ValueError(f"iteration indices are not contiguous: {indices}")
    return indices


def main() -> int:
    args = parse_args()
    if not args.trace.is_file():
        raise FileNotFoundError(f"trace does not exist: {args.trace}")

    reports: list[dict[str, object]] = []
    all_passed = True

    with np.load(args.trace, allow_pickle=False) as trace:
        indices = iteration_indices(trace.files)
        for index in indices:
            prefix = f"iteration_{index}_"
            h = np.asarray(trace[prefix + "H"], dtype=np.float64)
            residual = np.asarray(
                trace[prefix + "effective_residual_m"], dtype=np.float64
            )
            raw_residual = np.asarray(
                trace[prefix + "raw_residual_m"], dtype=np.float64
            )
            weight = np.asarray(trace[prefix + "W"], dtype=np.float64)

            # Independent algebra: do not import or call paper-era Torch code here.
            normal = h.T @ weight @ h
            rhs = h.T @ weight @ residual
            rank = int(np.linalg.matrix_rank(normal))
            condition = float(np.linalg.cond(normal))
            solve_error = None
            try:
                delta = np.linalg.solve(normal, rhs)
            except np.linalg.LinAlgError as exc:
                delta = np.full_like(rhs, np.nan)
                solve_error = str(exc)

            normal_diff = discrepancy(
                normal, trace[prefix + "normal_matrix"]
            )
            rhs_diff = discrepancy(rhs, trace[prefix + "rhs"])
            delta_diff = discrepancy(delta, trace[prefix + "delta_state"])
            active = np.asarray(
                trace[prefix + "active_state_indices"], dtype=np.int64
            )
            state_before = np.asarray(
                trace[prefix + "state_before"], dtype=np.float64
            )
            independently_updated = state_before.copy()
            if solve_error is None:
                independently_updated[active] += delta.reshape(-1)
            state_diff = discrepancy(
                independently_updated, trace[prefix + "state_after"]
            )

            h_rows = np.asarray(
                trace[prefix + "h_compact_rows"], dtype=np.int64
            )
            residual_rows = np.asarray(
                trace[prefix + "residual_compact_rows"], dtype=np.int64
            )
            row_alignment = bool(np.array_equal(h_rows, residual_rows))
            residual_equal = bool(np.array_equal(raw_residual, residual))
            full_rank = rank == h.shape[1]

            passed = bool(
                solve_error is None
                and full_rank
                and np.isfinite(condition)
                and condition <= args.max_condition
                and row_alignment
                and residual_equal
                and normal_diff["max_abs"] <= args.normal_tol
                and rhs_diff["max_abs"] <= args.rhs_tol
                and delta_diff["max_abs"] <= args.delta_tol
                and state_diff["max_abs"] <= args.state_tol
            )
            all_passed = all_passed and passed
            reports.append(
                {
                    "iteration": index,
                    "passed": passed,
                    "rows": int(h.shape[0]),
                    "state_dimension": int(h.shape[1]),
                    "rank": rank,
                    "full_rank": full_rank,
                    "condition_number_2": condition,
                    "solve_error": solve_error,
                    "h_residual_row_alignment_equal": row_alignment,
                    "raw_effective_residual_equal": residual_equal,
                    "normal_matrix": normal_diff,
                    "rhs": rhs_diff,
                    "delta_solve_vs_reference_inverse": delta_diff,
                    "updated_state": state_diff,
                    "weight_diagonal": np.diag(weight).tolist(),
                }
            )

    maxima = {
        "normal_matrix_max_abs": max(
            item["normal_matrix"]["max_abs"] for item in reports
        ),
        "rhs_max_abs": max(item["rhs"]["max_abs"] for item in reports),
        "delta_max_abs": max(
            item["delta_solve_vs_reference_inverse"]["max_abs"]
            for item in reports
        ),
        "updated_state_max_abs": max(
            item["updated_state"]["max_abs"] for item in reports
        ),
    }
    report = {
        "passed": all_passed,
        "trace": str(args.trace),
        "formula": "N=H.T@W@H; g=H.T@W@v; delta=numpy.linalg.solve(N,g)",
        "independent_implementation": (
            "NumPy float64 only; consumes exported H, effective residual v, and W"
        ),
        "tolerances": {
            "normal_matrix_max_abs": args.normal_tol,
            "rhs_max_abs": args.rhs_tol,
            "delta_max_abs": args.delta_tol,
            "updated_state_max_abs": args.state_tol,
            "condition_number_2_max": args.max_condition,
        },
        "max_discrepancies": maxima,
        "iterations": reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    if not all_passed:
        print("algebra validation FAILED", file=sys.stderr)
        return 1
    print("algebra validation PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
