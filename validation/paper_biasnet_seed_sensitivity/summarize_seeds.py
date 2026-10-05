#!/usr/bin/env python3
"""Aggregate all completed BiasNet seed runs without ranking or selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from validation.paper_biasnet_seed_sensitivity.experiment import (  # noqa: E402
    PREDEFINED_SEEDS,
    TRAINING_EPOCHS,
)


HISTORICAL_REFERENCE = {
    "seed": 20_260_929,
    "aggregate_membership": False,
    "description": "existing corrected-GT historical reference; not rerun",
    "KLT1_mean_2d_error_m": 2.1032,
    "KLT1_mean_3d_error_m": 5.5924,
    "KLT2_mean_2d_error_m": 2.4328,
    "KLT2_mean_3d_error_m": 5.7879,
}


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
    seed = result.get("seed")
    if result.get("status") != "completed":
        raise ValueError(f"seed {seed} is not a completed scientific run")
    if result.get("scientific_aggregate_eligible") is not True:
        raise ValueError(f"seed {seed} is ineligible for scientific aggregation")
    if result.get("executed_training_epochs") != TRAINING_EPOCHS:
        raise ValueError(f"seed {seed} did not execute 500 epochs")
    if not result.get("finite", {}).get("all_finite", False):
        raise ValueError(f"seed {seed} is not entirely finite")
    training = result["training_loss"]
    final = result["final_diagnostics"]
    held_out = result["held_out"]
    return {
        "seed": int(seed),
        "initial_loss_sum_3d_m": float(training["epoch_1_pre_update_sum_3d_m"]),
        "final_loss_sum_3d_m": float(training["final_post_update_sum_3d_m"]),
        "minimum_loss_sum_3d_m": float(training["minimum_evaluated"]["sum_3d_m"]),
        "minimum_loss_epoch": int(training["minimum_evaluated"]["epoch"]),
        "minimum_loss_stage": str(training["minimum_evaluated"]["stage"]),
        "dead_hidden_units_layer_1": int(
            final["hidden_layer_1"]["dead_neuron_count"]
        ),
        "dead_hidden_units_layer_2": int(
            final["hidden_layer_2"]["dead_neuron_count"]
        ),
        "dead_hidden_fraction_layer_1": float(
            final["hidden_layer_1"]["dead_neuron_fraction"]
        ),
        "dead_hidden_fraction_layer_2": float(
            final["hidden_layer_2"]["dead_neuron_fraction"]
        ),
        "final_bias_mean_m": float(final["bias_output"]["mean"]),
        "final_bias_std_m": float(final["bias_output"]["std"]),
        "KLT1_mean_2d_error_m": float(held_out["KLT1"]["mean_2d_error_m"]),
        "KLT1_mean_3d_error_m": float(held_out["KLT1"]["mean_3d_error_m"]),
        "KLT2_mean_2d_error_m": float(held_out["KLT2"]["mean_2d_error_m"]),
        "KLT2_mean_3d_error_m": float(held_out["KLT2"]["mean_3d_error_m"]),
    }


def _loss_curve_summary(results: list[dict[str, object]]) -> dict[str, object]:
    curves = np.asarray(
        [
            result["training_loss"]["pre_update_history_sum_3d_m"]
            for result in results
        ],
        dtype=np.float64,
    )
    if curves.shape != (len(results), TRAINING_EPOCHS):
        raise ValueError(f"unexpected loss-curve matrix shape: {curves.shape}")
    if not np.all(np.isfinite(curves)):
        raise ValueError("loss curves contain non-finite values")
    return {
        "epoch": list(range(1, TRAINING_EPOCHS + 1)),
        "individual_by_seed": {
            str(result["seed"]): curves[index].tolist()
            for index, result in enumerate(results)
        },
        "median": np.median(curves, axis=0).tolist(),
        "p25": np.percentile(curves, 25.0, axis=0).tolist(),
        "p75": np.percentile(curves, 75.0, axis=0).tolist(),
    }


def build_summary(results: list[dict[str, object]]) -> dict[str, object]:
    rows = [row_from_result(result) for result in results]
    ordered = sorted(zip(rows, results, strict=True), key=lambda item: item[0]["seed"])
    rows = [item[0] for item in ordered]
    results = [item[1] for item in ordered]
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed results")

    control_fields = (
        "configuration_sha256",
        "training_data_identity_sha256",
        "normalization_sha256",
        "held_out_data_identity_sha256",
    )
    for field in control_fields:
        values = {str(result[field]) for result in results}
        if len(values) != 1:
            raise ValueError(f"control differs across results: {field}")

    aggregate_metrics = {
        "final_training_loss_sum_3d_m": "final_loss_sum_3d_m",
        "KLT1_mean_3d_error_m": "KLT1_mean_3d_error_m",
        "KLT2_mean_3d_error_m": "KLT2_mean_3d_error_m",
        "final_bias_mean_m": "final_bias_mean_m",
        "final_bias_std_m": "final_bias_std_m",
        "dead_hidden_fraction_layer_1": "dead_hidden_fraction_layer_1",
        "dead_hidden_fraction_layer_2": "dead_hidden_fraction_layer_2",
    }
    return {
        "schema_version": 1,
        "status": "completed",
        "sample_size": len(rows),
        "interpretation": (
            "fixed n=10 initialization-sensitivity sample; not population-level proof"
        ),
        "seeds": seeds,
        "seed_order": "numeric; no metric-based ranking or seed selection",
        "rows": rows,
        "aggregate_statistics": {
            label: descriptive_statistics([float(row[field]) for row in rows])
            for label, field in aggregate_metrics.items()
        },
        "loss_curves": _loss_curve_summary(results),
        "control_hashes": {field: results[0][field] for field in control_fields},
        "historical_reference": HISTORICAL_REFERENCE,
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
        "# Corrected-GT BiasNet seed-sensitivity summary",
        "",
        "Rows are in numeric seed order. They are not ranked by any outcome.",
        "",
        "| Seed | Initial loss | Final loss | Minimum loss | Dead H1 | Dead H2 | Final bias mean | Final bias std | KLT1 2D | KLT1 3D | KLT2 2D | KLT2 3D |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {seed} | {initial:.9g} | {final:.9g} | {minimum:.9g} | "
            "{dead1} | {dead2} | {bias_mean:.9g} | {bias_std:.9g} | "
            "{k1_2d:.9g} | {k1_3d:.9g} | {k2_2d:.9g} | {k2_3d:.9g} |".format(
                seed=row["seed"],
                initial=row["initial_loss_sum_3d_m"],
                final=row["final_loss_sum_3d_m"],
                minimum=row["minimum_loss_sum_3d_m"],
                dead1=row["dead_hidden_units_layer_1"],
                dead2=row["dead_hidden_units_layer_2"],
                bias_mean=row["final_bias_mean_m"],
                bias_std=row["final_bias_std_m"],
                k1_2d=row["KLT1_mean_2d_error_m"],
                k1_3d=row["KLT1_mean_3d_error_m"],
                k2_2d=row["KLT2_mean_2d_error_m"],
                k2_3d=row["KLT2_mean_3d_error_m"],
            )
        )
    lines.extend(
        (
            "",
            "The predefined n=10 sample characterizes sensitivity; it is not population-level statistical proof.",
            "",
            "## Aggregate statistics",
            "",
        )
    )
    for metric, statistics in summary["aggregate_statistics"].items():
        lines.append(
            f"- {metric}: mean={statistics['mean']:.9g}, "
            f"median={statistics['median']:.9g}, std={statistics['std']:.9g}, "
            f"min={statistics['minimum']:.9g}, max={statistics['maximum']:.9g}"
        )
    lines.extend(
        (
            "",
            "## Historical reference (excluded from aggregates)",
            "",
            "| Seed | KLT1 2D | KLT1 3D | KLT2 2D | KLT2 3D |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {HISTORICAL_REFERENCE['seed']} | "
                f"{HISTORICAL_REFERENCE['KLT1_mean_2d_error_m']:.4f} | "
                f"{HISTORICAL_REFERENCE['KLT1_mean_3d_error_m']:.4f} | "
                f"{HISTORICAL_REFERENCE['KLT2_mean_2d_error_m']:.4f} | "
                f"{HISTORICAL_REFERENCE['KLT2_mean_3d_error_m']:.4f} |"
            ),
            "",
            "Seed 20260929 is an external historical row and is excluded from every statistic above.",
        )
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_dir = REPOSITORY / "results/paper_biasnet_seed_sensitivity"
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
    if set(args.seeds) != set(PREDEFINED_SEEDS):
        raise ValueError("final aggregation requires every predefined seed 0..9")
    outputs = (args.json_output.resolve(), args.table_output.resolve())
    occupied = [path for path in outputs if path.exists()]
    if occupied and not args.overwrite:
        raise FileExistsError(
            "output already exists (use --overwrite): "
            + ", ".join(str(path) for path in occupied)
        )
    summary = build_summary(load_results(args.input_dir, args.seeds))
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
