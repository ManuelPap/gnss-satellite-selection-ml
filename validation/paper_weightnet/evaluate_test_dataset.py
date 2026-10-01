#!/usr/bin/env python3
"""Evaluate one held-out dataset with the frozen paper-era WeightNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from held_out import (
    common_input_arguments,
    evaluate_prepared_dataset,
    inputs_from_args,
    load_frozen_weightnet,
    prepare_dataset,
    repository_root,
    summarize_results,
    write_results_csv,
    write_summary,
)


def main() -> int:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    common_input_arguments(parser)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "checkpoints/paper_weightnet/weightnet_3d.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/paper_weightnet",
    )
    args = parser.parse_args()

    spec, inputs = inputs_from_args(args)
    prepared = prepare_dataset(spec, inputs)
    model = load_frozen_weightnet(args.checkpoint)
    evaluations = evaluate_prepared_dataset(model, prepared)

    output_dir = args.output_dir.resolve()
    stem = spec.name.lower()
    csv_path = output_dir / f"{stem}_per_epoch.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    write_results_csv(csv_path, spec.name, evaluations)
    summary = summarize_results(
        prepared,
        evaluations,
        csv_path=csv_path,
        checkpoint=args.checkpoint,
    )
    write_summary(summary_path, summary)

    concise = {
        "dataset": spec.name,
        "released_code_epochs": len(prepared.epochs),
        "retained_measurements": prepared.measurement_count,
        "mean_2d_error_m": summary["tdl_w"]["mean_2d_error_m"],
        "mean_3d_error_m": summary["tdl_w"]["mean_3d_error_m"],
        "ols_mean_2d_error_m": summary["equal_weight_ols_sanity_baseline"]["mean_2d_error_m"],
        "ols_mean_3d_error_m": summary["equal_weight_ols_sanity_baseline"]["mean_3d_error_m"],
        "wls_failure_count": len(summary["wls_failures_included_by_released_predictor"]),
        "csv": str(csv_path),
        "summary": str(summary_path),
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
