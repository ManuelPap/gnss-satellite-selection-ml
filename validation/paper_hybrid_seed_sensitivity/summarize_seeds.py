#!/usr/bin/env python3
"""Aggregate completed predefined-seed TDL-BW runs without ranking them."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from validation.paper_hybrid_seed_sensitivity.experiment import (  # noqa: E402
    PREDEFINED_SEEDS,
    TRAINING_EPOCHS,
)


BIAS_STATUS_DEFINITIONS = {
    "dead from initialization": "initial fraction == 0 and final fraction == 0",
    "initially active, later dead": "initial fraction > 0 and final fraction == 0",
    "initially dead, later active": "initial fraction == 0 and final fraction > 0",
    "active": "initial fraction > 0 and final fraction > 0",
}


def classify_bias_status(initial_fraction: float, final_fraction: float) -> str:
    if not (
        math.isfinite(initial_fraction)
        and math.isfinite(final_fraction)
        and 0.0 <= initial_fraction <= 1.0
        and 0.0 <= final_fraction <= 1.0
    ):
        raise ValueError("positive fractions must be finite and in [0, 1]")
    initially_active = initial_fraction > 0.0
    finally_active = final_fraction > 0.0
    if not initially_active and not finally_active:
        return "dead from initialization"
    if initially_active and not finally_active:
        return "initially active, later dead"
    if not initially_active and finally_active:
        return "initially dead, later active"
    return "active"


def descriptive_statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("aggregate metric values must be finite and nonempty")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def row_from_result(result: dict[str, object]) -> dict[str, object]:
    if result.get("status") != "completed":
        raise ValueError(f"seed {result.get('seed')} is not a completed run")
    if result.get("executed_training_epochs") != TRAINING_EPOCHS:
        raise ValueError(f"seed {result.get('seed')} did not execute 100 epochs")
    if not result.get("finite", {}).get("all_finite", False):
        raise ValueError(f"seed {result.get('seed')} is not entirely finite")
    initial = float(result["initial_positive_bias_preactivation_fraction"])
    final = float(result["final_positive_bias_fraction"])
    held_out = result["held_out"]
    return {
        "seed": int(result["seed"]),
        "initial_positive_preactivation_fraction": initial,
        "final_positive_bias_fraction": final,
        "bias_status": classify_bias_status(initial, final),
        "final_training_loss_sum_3d_m": float(
            result["training_loss"]["final_post_update_sum_3d_m"]
        ),
        "KLT1_mean_2d_error_m": float(held_out["KLT1"]["mean_2d_error_m"]),
        "KLT1_mean_3d_error_m": float(held_out["KLT1"]["mean_3d_error_m"]),
        "KLT2_mean_2d_error_m": float(held_out["KLT2"]["mean_2d_error_m"]),
        "KLT2_mean_3d_error_m": float(held_out["KLT2"]["mean_3d_error_m"]),
    }


def build_summary(results: list[dict[str, object]]) -> dict[str, object]:
    rows = [row_from_result(result) for result in results]
    rows.sort(key=lambda row: row["seed"])
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed results")

    for control_field in (
        "configuration_sha256",
        "normalization_sha256",
        "training_data_identity_sha256",
    ):
        values = {str(result[control_field]) for result in results}
        if len(values) != 1:
            raise ValueError(f"control differs across results: {control_field}")

    observed_counts = Counter(str(row["bias_status"]) for row in rows)
    category_counts = {
        category: observed_counts.get(category, 0)
        for category in BIAS_STATUS_DEFINITIONS
    }
    return {
        "schema_version": 1,
        "status": "completed",
        "seeds": seeds,
        "seed_order": "numeric; no metric-based ranking or selection",
        "bias_status_definitions": BIAS_STATUS_DEFINITIONS,
        "rows": rows,
        "category_counts": category_counts,
        "aggregate_statistics": {
            "final_training_loss_sum_3d_m": descriptive_statistics(
                [float(row["final_training_loss_sum_3d_m"]) for row in rows]
            ),
            "KLT1_mean_3d_error_m": descriptive_statistics(
                [float(row["KLT1_mean_3d_error_m"]) for row in rows]
            ),
            "KLT2_mean_3d_error_m": descriptive_statistics(
                [float(row["KLT2_mean_3d_error_m"]) for row in rows]
            ),
        },
        "control_hashes": {
            field: results[0][field]
            for field in (
                "configuration_sha256",
                "normalization_sha256",
                "training_data_identity_sha256",
            )
        },
    }


def load_results(input_dir: Path, seeds: list[int]) -> list[dict[str, object]]:
    results = []
    for seed in seeds:
        path = input_dir.resolve() / f"seed_{seed}.json"
        result = json.loads(path.read_text())
        if int(result.get("seed", -1)) != seed:
            raise ValueError(f"result seed does not match filename: {path}")
        results.append(result)
    return results


def markdown_table(summary: dict[str, object]) -> str:
    lines = [
        "# TDL-BW seed-sensitivity summary",
        "",
        "Rows are in numeric seed order. They are not ranked by any outcome.",
        "",
        "| Seed | Initial positive preactivation fraction | Final positive bias fraction | Bias status | Final training loss | KLT1 2D | KLT1 3D | KLT2 2D | KLT2 3D |",
        "|---:|---:|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {seed} | {initial:.9g} | {final:.9g} | {status} | "
            "{loss:.9g} | {k1_2d:.9g} | {k1_3d:.9g} | "
            "{k2_2d:.9g} | {k2_3d:.9g} |".format(
                seed=row["seed"],
                initial=row["initial_positive_preactivation_fraction"],
                final=row["final_positive_bias_fraction"],
                status=row["bias_status"],
                loss=row["final_training_loss_sum_3d_m"],
                k1_2d=row["KLT1_mean_2d_error_m"],
                k1_3d=row["KLT1_mean_3d_error_m"],
                k2_2d=row["KLT2_mean_2d_error_m"],
                k2_3d=row["KLT2_mean_3d_error_m"],
            )
        )
    lines.extend(("", "## Bias status definitions", ""))
    for category, definition in summary["bias_status_definitions"].items():
        lines.append(f"- {category}: {definition}.")
    lines.extend(("", "## Category counts", ""))
    for category, count in summary["category_counts"].items():
        lines.append(f"- {category}: {count}")
    lines.extend(("", "## Aggregate statistics", ""))
    for metric, statistics in summary["aggregate_statistics"].items():
        lines.append(
            f"- {metric}: mean={statistics['mean']:.9g}, "
            f"median={statistics['median']:.9g}, std={statistics['std']:.9g}, "
            f"min={statistics['minimum']:.9g}, max={statistics['maximum']:.9g}"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_dir = REPOSITORY / "results/paper_hybrid_seed_sensitivity"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PREDEFINED_SEEDS))
    parser.add_argument("--input-dir", type=Path, default=default_dir)
    parser.add_argument("--json-output", type=Path, default=default_dir / "summary.json")
    parser.add_argument("--table-output", type=Path, default=default_dir / "summary.md")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must not contain duplicates")
    outside = sorted(set(args.seeds) - set(PREDEFINED_SEEDS))
    if outside:
        raise ValueError(f"summary seeds are fixed to 0..9: {outside}")
    if set(args.seeds) != set(PREDEFINED_SEEDS):
        raise ValueError("final aggregation requires every predefined seed 0..9")
    outputs = (args.json_output.resolve(), args.table_output.resolve())
    occupied = [path for path in outputs if path.exists()]
    if occupied and not args.overwrite:
        raise FileExistsError(
            "output already exists (use --overwrite): "
            + ", ".join(str(path) for path in occupied)
        )
    results = load_results(args.input_dir, args.seeds)
    summary = build_summary(results)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    outputs[1].write_text(markdown_table(summary))
    print(markdown_table(summary), end="")
    print(f"JSON summary: {outputs[0]}")
    print(f"Markdown table: {outputs[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
