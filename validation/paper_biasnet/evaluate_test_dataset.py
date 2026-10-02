#!/usr/bin/env python3
"""Evaluate the frozen reproduced BiasNet on KLT1 or KLT2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .held_out import (
        common_input_arguments,
        evaluate_prepared_dataset,
        inputs_from_args,
        load_frozen_biasnet,
        prepare_dataset,
        repository_root,
        summarize_results,
        write_results_csv,
        write_summary,
    )
except ImportError:
    from held_out import (
        common_input_arguments,
        evaluate_prepared_dataset,
        inputs_from_args,
        load_frozen_biasnet,
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
        default=root / "checkpoints/paper_biasnet/biasnet_3d.pth",
    )
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=Path(__file__).resolve().parent / "training_metrics.json",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "results/paper_biasnet")
    args = parser.parse_args()

    spec, inputs = inputs_from_args(args)
    prepared = prepare_dataset(spec, inputs)
    model = load_frozen_biasnet(args.checkpoint, args.training_metrics)
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
        training_metrics=args.training_metrics,
    )
    write_summary(summary_path, summary)
    print(
        json.dumps(
            {
                "dataset": spec.name,
                "released_code_epochs": len(prepared.epochs),
                "retained_measurements": prepared.measurement_count,
                "mean_2d_error_m": summary["tdl_b"]["mean_2d_error_m"],
                "mean_3d_error_m": summary["tdl_b"]["mean_3d_error_m"],
                "wls_failure_count": len(
                    summary["wls_failures_included_by_released_predictor"]
                ),
                "csv": str(csv_path),
                "summary": str(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
