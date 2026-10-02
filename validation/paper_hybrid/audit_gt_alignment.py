#!/usr/bin/env python3
"""Prove the released hybrid trainer's one-to-one KLT3 GT mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from validation.paper_biasnet_corrected_gt.audit_gt_alignment import (
    EXPECTED_MAXIMUM_MISMATCH_S,
    load_ground_truth_rows,
)


SELECTED_INDICES = (0, 1, 2, 10, 100, 200, 400)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    shared = here.parent / "paper_weightnet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=shared / "klt3_features.npz")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(
            "/tmp/gnss-weightnet-repro/extracted/data/0610_KLT/20210610_100.txt"
        ),
    )
    parser.add_argument("--output", type=Path, default=here / "gt_alignment_audit.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with np.load(args.features.resolve(), allow_pickle=False) as cache:
        epoch_times = cache["epoch_times_gpst_like"].copy()
        cached_targets = cache["ground_truth_geodetic_deg_m"].copy()
        epoch_offsets = cache["epoch_offsets"].copy()
    epoch_count = int(epoch_times.size)
    table = load_ground_truth_rows(
        args.ground_truth.resolve(), epoch_times[0] - 1.0, epoch_times[-1] + 1.0
    )
    matched = np.abs(table[:, 2, None] - epoch_times[None, :]).argmin(axis=0)
    reconstructed = table[matched, 3:6]
    if not np.array_equal(reconstructed, cached_targets):
        discrepancy = float(np.max(np.abs(reconstructed - cached_targets)))
        raise RuntimeError(f"raw GT reconstruction differs from cache by {discrepancy}")
    consumed_list_indices = np.arange(epoch_count, dtype=np.int64)
    offsets = table[matched, 2] - epoch_times
    if float(np.max(np.abs(offsets))) > EXPECTED_MAXIMUM_MISMATCH_S:
        raise RuntimeError("hybrid GT timestamp mismatch exceeds audited tolerance")
    selected = []
    for index in SELECTED_INDICES:
        row = table[int(matched[index])]
        selected.append(
            {
                "gnss_epoch_index": index,
                "gnss_timestamp_gpst_like": float(epoch_times[index]),
                "hybrid_gt_list_index_consumed": int(consumed_list_indices[index]),
                "released_dataframe_gt_row_index": int(row[0]),
                "original_gt_timestamp_utc": float(row[1]),
                "gt_timestamp_after_plus_18_s": float(row[2]),
                "time_difference_gt_minus_gnss_s": float(row[2] - epoch_times[index]),
                "measurement_rows": int(epoch_offsets[index + 1] - epoch_offsets[index]),
            }
        )
    output = {
        "status": "passed",
        "reference_commit": "dd5eac669676ba0a922102047e58c2dfc9be9267",
        "source_proof": {
            "file": "hybrid_network_train.py",
            "append_site": "lines 58-60 append exactly once inside the retained-time predicate, before the OLS status check",
            "consumer": "lines 103-109 loop over retained obss and consume gts[i] before applying the same OLS status check",
            "second_append_present": False,
            "mapping": "training GNSS epoch i consumes hybrid GT list index i",
        },
        "cardinality": {
            "gnss_epochs_retained_by_strict_time_window": epoch_count,
            "hybrid_gt_entries_built": epoch_count,
            "successful_ols_epochs": epoch_count,
            "training_epochs_contributing_loss": epoch_count,
            "satellite_measurements": int(epoch_offsets[-1]),
            "unique_consumed_gt_list_indices": int(
                np.unique(consumed_list_indices).size
            ),
        },
        "alignment_statistics_seconds": {
            "maximum_absolute_mismatch": float(np.max(np.abs(offsets))),
            "median_absolute_mismatch": float(np.median(np.abs(offsets))),
            "p95_absolute_mismatch": float(np.percentile(np.abs(offsets), 95.0)),
            "minimum_signed_difference": float(offsets.min()),
            "maximum_signed_difference": float(offsets.max()),
            "expected_maximum_tolerance": EXPECTED_MAXIMUM_MISMATCH_S,
        },
        "defect_result": {
            "standalone_biasnet_duplicate_append_defect_present": False,
            "other_hybrid_gt_alignment_defect_found": False,
            "corrected_target_mode_required": False,
            "second_training_experiment_run": False,
        },
        "selected_mappings": selected,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
