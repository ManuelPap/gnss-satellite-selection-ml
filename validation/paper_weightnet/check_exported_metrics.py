#!/usr/bin/env python3
"""Independently aggregate an exported per-epoch CSV only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def aggregate_csv(path: Path) -> dict[str, float | int]:
    error_2d: list[float] = []
    error_3d: list[float] = []
    ols_2d: list[float] = []
    ols_3d: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            error_2d.append(float(row["error_2d_m"]))
            error_3d.append(float(row["error_3d_m"]))
            ols_2d.append(float(row["ols_error_2d_m"]))
            ols_3d.append(float(row["ols_error_3d_m"]))
    if not error_2d:
        raise RuntimeError("CSV has no data rows")
    # Deliberately uses math.fsum/len rather than the evaluator's NumPy mean.
    return {
        "rows": len(error_2d),
        "mean_2d_error_m": math.fsum(error_2d) / len(error_2d),
        "mean_3d_error_m": math.fsum(error_3d) / len(error_3d),
        "ols_mean_2d_error_m": math.fsum(ols_2d) / len(ols_2d),
        "ols_mean_3d_error_m": math.fsum(ols_3d) / len(ols_3d),
    }


def compare_summary(
    metrics: dict[str, float | int], summary_path: Path, tolerance: float
) -> None:
    summary = json.loads(summary_path.read_text())
    expected = {
        "rows": summary["per_epoch_csv"]["rows"],
        "mean_2d_error_m": summary["tdl_w"]["mean_2d_error_m"],
        "mean_3d_error_m": summary["tdl_w"]["mean_3d_error_m"],
        "ols_mean_2d_error_m": summary["equal_weight_ols_sanity_baseline"]["mean_2d_error_m"],
        "ols_mean_3d_error_m": summary["equal_weight_ols_sanity_baseline"]["mean_3d_error_m"],
    }
    if metrics["rows"] != expected["rows"]:
        raise RuntimeError(
            f"row-count mismatch: CSV={metrics['rows']}, summary={expected['rows']}"
        )
    for name in expected:
        if name == "rows":
            continue
        difference = abs(float(metrics[name]) - float(expected[name]))
        if difference > tolerance:
            raise RuntimeError(
                f"{name} mismatch {difference:.17g} exceeds tolerance {tolerance:.17g}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()
    metrics = aggregate_csv(args.csv.resolve())
    if args.summary:
        compare_summary(metrics, args.summary.resolve(), args.tolerance)
        metrics["summary_agreement_within_tolerance"] = True
        metrics["tolerance"] = args.tolerance
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
