#!/usr/bin/env python3
"""Independently aggregate a hybrid held-out CSV without evaluator imports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def aggregate_csv(path: Path) -> dict[str, float | int]:
    two_d: list[float] = []
    three_d: list[float] = []
    ols_two_d: list[float] = []
    ols_three_d: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            two_d.append(float(row["error_2d_m"]))
            three_d.append(float(row["error_3d_m"]))
            ols_two_d.append(float(row["ols_error_2d_m"]))
            ols_three_d.append(float(row["ols_error_3d_m"]))
    if not two_d:
        raise RuntimeError("CSV contains no epoch rows")

    def metrics(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean_m": float(array.mean()),
            "median_m": float(np.median(array)),
            "rms_m": float(np.sqrt(np.mean(array**2))),
            "p68_m": float(np.percentile(array, 68.0)),
            "p95_m": float(np.percentile(array, 95.0)),
        }

    return {
        "rows": len(two_d),
        "mean_2d_error_m": float(np.mean(two_d)),
        "mean_3d_error_m": float(np.mean(three_d)),
        "ols_mean_2d_error_m": float(np.mean(ols_two_d)),
        "ols_mean_3d_error_m": float(np.mean(ols_three_d)),
        "error_2d_diagnostics": metrics(two_d),
        "error_3d_diagnostics": metrics(three_d),
    }


def compare_summary(
    metrics: dict[str, object], summary_path: Path, tolerance: float
) -> None:
    summary = json.loads(summary_path.read_text())
    expected = {
        "rows": summary["per_epoch_csv"]["rows"],
        "mean_2d_error_m": summary["tdl_bw"]["mean_2d_error_m"],
        "mean_3d_error_m": summary["tdl_bw"]["mean_3d_error_m"],
        "ols_mean_2d_error_m": summary["equal_weight_ols_sanity_baseline"]["mean_2d_error_m"],
        "ols_mean_3d_error_m": summary["equal_weight_ols_sanity_baseline"]["mean_3d_error_m"],
    }
    if int(metrics["rows"]) != int(expected["rows"]):
        raise RuntimeError("CSV and evaluator row counts differ")
    for name in expected:
        if name == "rows":
            continue
        difference = abs(float(metrics[name]) - float(expected[name]))
        if difference > tolerance:
            raise RuntimeError(f"{name} differs by {difference}, tolerance {tolerance}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()
    result = aggregate_csv(args.csv.resolve())
    compare_summary(result, args.summary.resolve(), args.tolerance)
    result["agreed_with_evaluator_summary"] = True
    result["tolerance"] = args.tolerance
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
