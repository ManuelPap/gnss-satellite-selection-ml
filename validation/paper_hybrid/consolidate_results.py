#!/usr/bin/env python3
"""Consolidate frozen hybrid results and prior independent learned baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_exported_metrics import aggregate_csv, compare_summary


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results/paper_hybrid")
    parser.add_argument(
        "--output", type=Path, default=here / "held_out_evaluation_summary.json"
    )
    args = parser.parse_args()
    weight_prior = json.loads(
        (root / "validation/paper_weightnet/held_out_evaluation_summary.json").read_text()
    )
    bias_prior = json.loads(
        (root / "validation/paper_biasnet_corrected_gt/held_out_evaluation_summary.json").read_text()
    )
    datasets: dict[str, object] = {}
    comparison: list[dict[str, object]] = []
    for name in ("KLT1", "KLT2"):
        stem = name.lower()
        summary_path = args.results.resolve() / f"{stem}_summary.json"
        csv_path = args.results.resolve() / f"{stem}_per_epoch.csv"
        summary = json.loads(summary_path.read_text())
        independent = aggregate_csv(csv_path)
        compare_summary(independent, summary_path, 1.0e-12)
        independent["agreed_with_evaluator_summary"] = True
        independent["tolerance"] = 1.0e-12
        summary["independent_csv_checker"] = independent
        datasets[name] = summary

        prior_weight = weight_prior["datasets"][name]
        prior_bias = bias_prior["datasets"][name]
        comparison.extend(
            [
                {
                    "dataset": name,
                    "method": "our frozen TDL-W",
                    "mean_2d_error_m": prior_weight["tdl_w_mean_2d_m"],
                    "mean_3d_error_m": prior_weight["tdl_w_mean_3d_m"],
                },
                {
                    "dataset": name,
                    "method": "our frozen corrected-GT TDL-B",
                    "mean_2d_error_m": prior_bias["corrected"]["mean_2d_error_m"],
                    "mean_3d_error_m": prior_bias["corrected"]["mean_3d_error_m"],
                },
                {
                    "dataset": name,
                    "method": "our frozen TDL-BW",
                    "mean_2d_error_m": summary["tdl_bw"]["mean_2d_error_m"],
                    "mean_3d_error_m": summary["tdl_bw"]["mean_3d_error_m"],
                },
                {
                    "dataset": name,
                    "method": "paper TDL-W",
                    "mean_2d_error_m": prior_weight["paper_tdl_w_mean_2d_m"],
                    "mean_3d_error_m": prior_weight["paper_tdl_w_mean_3d_m"],
                },
                {
                    "dataset": name,
                    "method": "paper TDL-B",
                    "mean_2d_error_m": prior_bias["paper"]["mean_2d_error_m"],
                    "mean_3d_error_m": prior_bias["paper"]["mean_3d_error_m"],
                },
                {
                    "dataset": name,
                    "method": "paper TDL-BW",
                    "mean_2d_error_m": summary["tdl_bw"]["paper_mean_2d_error_m"],
                    "mean_3d_error_m": summary["tdl_bw"]["paper_mean_3d_error_m"],
                },
            ]
        )
    output = {
        "status": "passed",
        "reference_commit": "dd5eac669676ba0a922102047e58c2dfc9be9267",
        "held_out_data_used_for_training": False,
        "datasets": datasets,
        "learned_baseline_comparison": comparison,
        "comparison_interpretation": "descriptive only; random initialization, paper/code discrepancies, and unavailable author runnable checkpoints prevent strict model-parameter equivalence",
        "whampoa": {
            "status": "not_evaluated",
            "reason": "exact July 2021 historical dataset is not open-source",
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": output["status"],
        "comparison": comparison,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
