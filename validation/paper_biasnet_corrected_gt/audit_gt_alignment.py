#!/usr/bin/env python3
"""Audit released duplicate GT appends and corrected one-to-one KLT3 alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from validation.paper_biasnet_corrected_gt.experiment import (
    corrected_target_indices,
    historical_target_indices,
)


LEAP_SECONDS = 18.0
EXPECTED_MAXIMUM_MISMATCH_S = 0.0041
SELECTED_AB_INDICES = (0, 1, 2, 10, 100, 200, 400)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=root / "validation/paper_weightnet/klt3_features.npz",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(
            "/tmp/gnss-weightnet-repro/extracted/data/0610_KLT/20210610_100.txt"
        ),
    )
    parser.add_argument("--output", type=Path, default=here / "gt_alignment_audit.json")
    return parser.parse_args()


def load_ground_truth_rows(
    path: Path, minimum_aligned_time: float, maximum_aligned_time: float
) -> np.ndarray:
    """Return [released dataframe index, UTC, UTC+18, lat, lon, altitude]."""

    rows: list[list[float]] = []
    dataframe_index = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream):
            if line_number < 30:
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            try:
                values = [float(item) for item in fields[:10]]
            except ValueError:
                continue
            original_utc = values[0]
            aligned = original_utc + LEAP_SECONDS
            if minimum_aligned_time <= aligned <= maximum_aligned_time:
                latitude = values[3] + values[4] / 60.0 + values[5] / 3600.0
                longitude = values[6] + values[7] / 60.0 + values[8] / 3600.0
                rows.append(
                    [
                        float(dataframe_index),
                        original_utc,
                        aligned,
                        latitude,
                        longitude,
                        values[9],
                    ]
                )
            dataframe_index += 1
    result = np.asarray(rows, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 6 or result.shape[0] == 0:
        raise RuntimeError("ground-truth audit window is empty or malformed")
    if np.any(np.diff(result[:, 2]) < 0.0):
        raise RuntimeError("ground-truth rows are not chronological")
    return result


def alignment_row(
    epoch_index: int,
    epoch_times: np.ndarray,
    table: np.ndarray,
    matched_rows: np.ndarray,
) -> dict[str, object]:
    row = table[int(matched_rows[epoch_index])]
    difference = float(row[2] - epoch_times[epoch_index])
    return {
        "gnss_epoch_index": epoch_index,
        "gnss_timestamp_gpst_like": float(epoch_times[epoch_index]),
        "original_gt_timestamp_utc": float(row[1]),
        "gt_timestamp_after_plus_18_s": float(row[2]),
        "time_difference_after_alignment_s": difference,
        "absolute_time_difference_after_alignment_s": abs(difference),
        "gt_row_index": int(row[0]),
    }


def main() -> int:
    args = parse_args()
    with np.load(args.features.resolve(), allow_pickle=False) as cache:
        epoch_times = cache["epoch_times_gpst_like"].copy()
        cached_targets = cache["ground_truth_geodetic_deg_m"].copy()
    epoch_count = epoch_times.size
    table = load_ground_truth_rows(
        args.ground_truth.resolve(), epoch_times[0] - 1.0, epoch_times[-1] + 1.0
    )
    matched_rows = np.abs(table[:, 2, None] - epoch_times[None, :]).argmin(axis=0)
    matched_targets = table[matched_rows, 3:6]
    if not np.array_equal(matched_targets, cached_targets):
        maximum = float(np.max(np.abs(matched_targets - cached_targets)))
        raise RuntimeError(f"raw GT reconstruction differs from cache by {maximum}")

    corrected_indices = corrected_target_indices(epoch_count)
    historical_indices = historical_target_indices(epoch_count)
    if np.unique(corrected_indices).size != epoch_count:
        raise RuntimeError("corrected target indices are not unique")
    if int(corrected_indices[-1]) != epoch_count - 1:
        raise RuntimeError("final corrected epoch does not map to final GT region")
    corrected_offsets = table[matched_rows[corrected_indices], 2] - epoch_times
    historical_offsets = table[matched_rows[historical_indices], 2] - epoch_times
    maximum_mismatch = float(np.max(np.abs(corrected_offsets)))
    if maximum_mismatch > EXPECTED_MAXIMUM_MISMATCH_S:
        raise RuntimeError(
            f"corrected GT mismatch {maximum_mismatch} exceeds expected tolerance"
        )

    middle_start = epoch_count // 2 - 5
    sections = {
        "first_10": [
            alignment_row(i, epoch_times, table, matched_rows) for i in range(10)
        ],
        "middle_10": [
            alignment_row(i, epoch_times, table, matched_rows)
            for i in range(middle_start, middle_start + 10)
        ],
        "last_10": [
            alignment_row(i, epoch_times, table, matched_rows)
            for i in range(epoch_count - 10, epoch_count)
        ],
    }
    ab_rows = []
    for epoch_index in SELECTED_AB_INDICES:
        historical_source = int(historical_indices[epoch_index])
        corrected_source = int(corrected_indices[epoch_index])
        historical_row = table[int(matched_rows[historical_source])]
        corrected_row = table[int(matched_rows[corrected_source])]
        ab_rows.append(
            {
                "gnss_epoch_index": epoch_index,
                "gnss_timestamp_gpst_like": float(epoch_times[epoch_index]),
                "historical_source_epoch_index": historical_source,
                "historical_target_timestamp_after_plus_18_s": float(
                    historical_row[2]
                ),
                "corrected_source_epoch_index": corrected_source,
                "corrected_target_timestamp_after_plus_18_s": float(corrected_row[2]),
                "historical_temporal_offset_s": float(
                    historical_row[2] - epoch_times[epoch_index]
                ),
                "corrected_temporal_offset_s": float(
                    corrected_row[2] - epoch_times[epoch_index]
                ),
            }
        )

    corrected_abs = np.abs(corrected_offsets)
    historical_abs = np.abs(historical_offsets)
    x = np.arange(epoch_count, dtype=np.float64)
    historical_slope = float(np.polyfit(x, historical_abs, 1)[0])
    corrected_slope = float(np.polyfit(x, corrected_abs, 1)[0])
    output = {
        "status": "passed",
        "experiment": "Corrected-GT BiasNet controlled experiment",
        "released_defect_proof": {
            "source_file": "bias_network_train.py at dd5eac669676ba0a922102047e58c2dfc9be9267",
            "function": "module-level preprocessing and training loops",
            "first_append": "lines 57-58: nearest gt_row; gts.append(...) before OLS status check",
            "second_append": "lines 74-76: nearest gt_row; gts.append(...) after successful OLS solve",
            "consumer": "lines 101-107: loop i over obss and select gt_row = gts[i]",
            "successful_retained_epochs": epoch_count,
            "historical_gt_list_length": epoch_count * 2,
            "historical_training_target_source_epoch_indices_first_12": historical_indices[
                :12
            ].tolist(),
            "mapping": "GNSS epoch i -> matched GT for source epoch floor(i/2)",
            "temporal_misalignment_explanation": (
                "the GNSS time advances one second per epoch while the consumed duplicated "
                "target advances only every two epochs, so lag grows by about 0.5 s/epoch"
            ),
        },
        "correction_contract": {
            "training_epoch_count": epoch_count,
            "training_gt_target_count": int(corrected_indices.size),
            "one_target_per_epoch": True,
            "unique_target_index_count": int(np.unique(corrected_indices).size),
            "corrected_target_indices_first_12": corrected_indices[:12].tolist(),
            "final_epoch_index": epoch_count - 1,
            "final_target_source_epoch_index": int(corrected_indices[-1]),
            "leap_second_treatment": "+18 s applied to original GT UTC timestamp",
            "nearest_match_tie_behavior": "first row, matching NumPy/Pandas argmin",
        },
        "corrected_alignment_statistics_seconds": {
            "maximum_absolute_mismatch": maximum_mismatch,
            "median_absolute_mismatch": float(np.median(corrected_abs)),
            "p95_absolute_mismatch": float(np.percentile(corrected_abs, 95.0)),
            "expected_maximum_tolerance": EXPECTED_MAXIMUM_MISMATCH_S,
        },
        "historical_alignment_statistics_seconds": {
            "maximum_absolute_mismatch": float(np.max(historical_abs)),
            "median_absolute_mismatch": float(np.median(historical_abs)),
            "p95_absolute_mismatch": float(np.percentile(historical_abs, 95.0)),
        },
        "offset_growth_comparison": {
            "historical_first_absolute_offset_s": float(historical_abs[0]),
            "historical_last_absolute_offset_s": float(historical_abs[-1]),
            "historical_absolute_offset_slope_per_epoch": historical_slope,
            "corrected_first_absolute_offset_s": float(corrected_abs[0]),
            "corrected_last_absolute_offset_s": float(corrected_abs[-1]),
            "corrected_absolute_offset_slope_per_epoch": corrected_slope,
            "corrected_error_grows_with_epoch_index": bool(
                abs(corrected_slope) > 1.0e-12
            ),
        },
        "selected_ab_comparison": ab_rows,
        "alignment_samples": sections,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
