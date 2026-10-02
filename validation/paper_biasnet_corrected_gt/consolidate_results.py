#!/usr/bin/env python3
"""Consolidate corrected KLT1/KLT2 results with defective and paper values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation.paper_biasnet.check_exported_metrics import (
    aggregate_csv,
    compare_summary,
)


DEFECTIVE = {
    "KLT1": {"mean_2d_error_m": 102.0353, "mean_3d_error_m": 184.7434},
    "KLT2": {"mean_2d_error_m": 92.7961, "mean_3d_error_m": 166.7303},
}


def difference(value: float, reference: float) -> dict[str, float]:
    delta = value - reference
    return {
        "difference_m": delta,
        "relative_difference_percent": delta / reference * 100.0,
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results/paper_biasnet_corrected_gt",
    )
    parser.add_argument("--training-metrics", type=Path, default=here / "training_metrics.json")
    parser.add_argument(
        "--output", type=Path, default=here / "held_out_evaluation_summary.json"
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()
    training = json.loads(args.training_metrics.resolve().read_text())
    datasets: dict[str, object] = {}
    for name in ("KLT1", "KLT2"):
        stem = name.lower()
        summary_path = args.results_dir.resolve() / f"{stem}_summary.json"
        csv_path = args.results_dir.resolve() / f"{stem}_per_epoch.csv"
        summary = json.loads(summary_path.read_text())
        independent = aggregate_csv(csv_path)
        compare_summary(independent, summary_path, args.tolerance)
        corrected_2d = float(summary["tdl_b"]["mean_2d_error_m"])
        corrected_3d = float(summary["tdl_b"]["mean_3d_error_m"])
        paper_2d = float(summary["tdl_b"]["paper_mean_2d_error_m"])
        paper_3d = float(summary["tdl_b"]["paper_mean_3d_error_m"])
        defective = DEFECTIVE[name]
        datasets[name] = {
            "cardinality": summary["cardinality"],
            "corrected": {
                "mean_2d_error_m": corrected_2d,
                "mean_3d_error_m": corrected_3d,
            },
            "paper": {"mean_2d_error_m": paper_2d, "mean_3d_error_m": paper_3d},
            "historical_defective": defective,
            "corrected_minus_paper_2d": difference(corrected_2d, paper_2d),
            "corrected_minus_paper_3d": difference(corrected_3d, paper_3d),
            "corrected_minus_defective_2d": difference(
                corrected_2d, defective["mean_2d_error_m"]
            ),
            "corrected_minus_defective_3d": difference(
                corrected_3d, defective["mean_3d_error_m"]
            ),
            "bias_outputs": summary["bias_outputs"],
            "wls_failures": summary["wls_failures_included_by_released_predictor"],
            "independent_csv_checker": {
                **independent,
                "agreed_with_evaluator_summary": True,
                "tolerance": args.tolerance,
            },
        }
    output = {
        "status": "passed",
        "experiment": "Corrected-GT BiasNet controlled experiment",
        "checkpoint_sha256": training["checkpoint"]["sha256"],
        "training_datasets": training["ab_control"]["training_datasets"],
        "held_out_datasets": training["ab_control"]["held_out_datasets"],
        "held_out_data_used_for_training": False,
        "datasets": datasets,
    }
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
