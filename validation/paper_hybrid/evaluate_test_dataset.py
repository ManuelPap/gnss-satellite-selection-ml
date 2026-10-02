#!/usr/bin/env python3
"""Evaluate KLT1 or KLT2 with the frozen paper-era shared TDL-BW model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from held_out import (
    common_input_arguments,
    evaluate_prepared_dataset,
    inputs_from_args,
    load_frozen_hybrid,
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
        default=root / "checkpoints/paper_hybrid/hybrid_share_3d.pth",
    )
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=root / "validation/paper_hybrid/training_metrics.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results/paper_hybrid"
    )
    args = parser.parse_args()
    spec, inputs = inputs_from_args(args)
    if spec.name not in ("KLT1", "KLT2"):
        raise ValueError("this reproduction intentionally evaluates only KLT1/KLT2")
    prepared = prepare_dataset(spec, inputs)
    model = load_frozen_hybrid(args.checkpoint, args.training_metrics)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    evaluations = evaluate_prepared_dataset(model, prepared)
    for name, value in model.named_parameters():
        if not value.detach().equal(before[name]):
            raise RuntimeError("model parameter changed during frozen inference")
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / f"{spec.name.lower()}_per_epoch.csv"
    summary_path = output_dir / f"{spec.name.lower()}_summary.json"
    write_results_csv(csv_path, spec.name, evaluations)
    summary = summarize_results(
        prepared,
        evaluations,
        csv_path=csv_path,
        checkpoint=args.checkpoint,
        training_metrics=args.training_metrics,
    )
    write_summary(summary_path, summary)
    print(json.dumps({
        "dataset": spec.name,
        "epochs": len(prepared.epochs),
        "measurements": prepared.measurement_count,
        "mean_2d_error_m": summary["tdl_bw"]["mean_2d_error_m"],
        "mean_3d_error_m": summary["tdl_bw"]["mean_3d_error_m"],
        "summary": str(summary_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
