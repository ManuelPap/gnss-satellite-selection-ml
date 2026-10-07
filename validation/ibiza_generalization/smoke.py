"""Run and print the intentionally small seed-0 Ibiza inference smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from validation.ibiza_generalization.inference import (
    DEFAULT_IBIZA_NPZ,
    IBIZA_NPZ_SHA256,
    SMOKE_EPOCH_INDICES,
    run_seed_zero_smoke,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_IBIZA_NPZ)
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=list(SMOKE_EPOCH_INDICES),
        help="Accepted-epoch indices (default: 0 1 2).",
    )
    return parser.parse_args(argv)


def _range(values: object) -> list[float] | None:
    if values is None:
        return None
    return [float(values.min().cpu()), float(values.max().cpu())]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_seed_zero_smoke(
        dataset_path=args.dataset, accepted_epoch_indices=args.epochs
    )
    first = results[0]
    report = {
        "scope": "infrastructure/numerical smoke only; no positioning-accuracy claim",
        "dataset": "../external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz",
        "dataset_sha256": IBIZA_NPZ_SHA256,
        "common_rows_identical": True,
        "epoch_selection": [
            {
                "accepted_epoch_index": epoch.accepted_epoch_index,
                "split_epoch_index": epoch.split_epoch_index,
                "epoch_time_gpst_like_s": epoch.epoch_time_gpst_like_s,
                "row_range_half_open": [epoch.row_start, epoch.row_stop],
                "row_count": epoch.row_stop - epoch.row_start,
            }
            for epoch in first.epochs
        ],
        "architectures": {},
    }
    for architecture in results:
        report["architectures"][architecture.architecture] = {
            "seed": architecture.seed,
            "checkpoint_sha256": architecture.checkpoint_sha256,
            "output_semantics": list(architecture.output_semantics),
            "checkpoint_parameters_unchanged": (
                architecture.checkpoint_parameters_unchanged
            ),
            "epochs": [
                {
                    "accepted_epoch_index": epoch.accepted_epoch_index,
                    "receiver_state": epoch.receiver_state.cpu().tolist(),
                    "predicted_bias_m_range": _range(
                        epoch.neural_outputs.predicted_bias_m
                    ),
                    "predicted_weight_range": _range(
                        epoch.neural_outputs.predicted_weight
                    ),
                    "wls": {
                        "solution_status": epoch.wls.solution_status,
                        "convergence_status": epoch.wls.convergence_status,
                        "iterations": len(epoch.wls.iterations),
                        "rank_status": epoch.wls.rank_status,
                        "final_rank": epoch.wls.final_rank,
                        "active_state_count": epoch.wls.active_state_count,
                        "conditioning_status": epoch.wls.conditioning_status,
                        "final_condition_number": (
                            epoch.wls.final_condition_number
                        ),
                    },
                }
                for epoch in architecture.epochs
            ],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
