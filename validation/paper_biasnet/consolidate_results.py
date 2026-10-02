#!/usr/bin/env python3
"""Consolidate BiasNet KLT1/KLT2 summaries and independently check their CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .check_exported_metrics import aggregate_csv, compare_summary
except ImportError:
    from check_exported_metrics import aggregate_csv, compare_summary


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=root / "results/paper_biasnet"
    )
    parser.add_argument("--training-metrics", type=Path, default=here / "training_metrics.json")
    parser.add_argument(
        "--output", type=Path, default=here / "held_out_evaluation_summary.json"
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()

    training = json.loads(args.training_metrics.resolve().read_text())
    datasets: dict[str, object] = {}
    for name in ("klt1", "klt2"):
        summary_path = args.results_dir.resolve() / f"{name}_summary.json"
        csv_path = args.results_dir.resolve() / f"{name}_per_epoch.csv"
        summary = json.loads(summary_path.read_text())
        independent = aggregate_csv(csv_path)
        compare_summary(independent, summary_path, args.tolerance)
        datasets[name.upper()] = {
            "cardinality": summary["cardinality"],
            "tdl_b": summary["tdl_b"],
            "equal_weight_ols_sanity_baseline": summary[
                "equal_weight_ols_sanity_baseline"
            ],
            "bias_outputs": summary["bias_outputs"],
            "wls_failures_included_by_released_predictor": summary[
                "wls_failures_included_by_released_predictor"
            ],
            "independent_csv_checker": {
                **independent,
                "agreed_with_evaluator_summary": True,
                "tolerance": args.tolerance,
            },
        }
    output = {
        "status": "passed",
        "reference_commit": "dd5eac669676ba0a922102047e58c2dfc9be9267",
        "checkpoint_sha256": training["checkpoint"]["sha256"],
        "training": {
            "seed": training["configuration"]["seed"],
            "epochs": training["configuration"]["training_epochs"],
            "learning_rate": training["configuration"]["learning_rate"],
            "optimizer": training["configuration"]["optimizer"],
            "loss": training["configuration"]["loss"],
            "ground_truth_alignment": training["configuration"][
                "ground_truth_alignment"
            ],
            "klt3_bias_outputs": training["bias_output_diagnostics_klt3"],
        },
        "datasets": datasets,
        "whampoa": {
            "status": "not_evaluated_exact_historical_2021_07_14_inputs_not_public",
            "different_public_run_substituted": False,
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
