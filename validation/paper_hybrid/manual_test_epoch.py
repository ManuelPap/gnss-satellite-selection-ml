#!/usr/bin/env python3
"""Print one end-to-end physical TDL-BW epoch and every WLS iteration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from held_out import (
    DATASET_SPECS,
    evaluate_epoch,
    load_frozen_hybrid,
    prepare_dataset,
    repository_root,
    resolve_input_paths,
)


def show(name: str, value: object) -> None:
    print(f"{name} =")
    print(np.asarray(value))


def main() -> int:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("KLT1", "KLT2"), default="KLT1")
    parser.add_argument("--epoch-index", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--tdl-dir", type=Path)
    parser.add_argument("--pyrtklib-site", type=Path)
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
    args = parser.parse_args()
    np.set_printoptions(precision=15, linewidth=240, suppress=False)
    spec = DATASET_SPECS[args.dataset]
    inputs = resolve_input_paths(
        spec,
        data_root=args.data_root,
        tdl_dir=args.tdl_dir,
        pyrtklib_site=args.pyrtklib_site,
    )
    prepared = prepare_dataset(spec, inputs)
    epoch = prepared.epochs[args.epoch_index]
    result = evaluate_epoch(
        load_frozen_hybrid(args.checkpoint, args.training_metrics), epoch
    )

    print(f"dataset={args.dataset} epoch_index={args.epoch_index} timestamp={epoch.epoch_time:.9f}")
    print("Per-satellite neural measurement controls:")
    for row, prn in enumerate(epoch.satellite_ids):
        print(
            f"row={row:02d} PRN={prn} "
            f"RTKLIB_corrected_P_m={epoch.corrected_pseudorange_m[row]:.15g} "
            f"C/N0={epoch.features[row,0]:.15g} "
            f"elevation_rad={epoch.features[row,1]:.15g} "
            f"OLS_residual_m={epoch.features[row,2]:.15g} "
            f"raw_features={epoch.features[row].tolist()} "
            f"normalized_features={result.normalized_features[row].tolist()} "
            f"predicted_bias_m={result.predicted_bias_m[row]:.15g} "
            f"P_minus_b_m={result.bias_corrected_pseudorange_m[row]:.15g} "
            f"predicted_weight={result.weights[row]:.15g}"
        )

    state_before = epoch.initial_ols_state.copy()
    raw_pseudorange = epoch.corrected_pseudorange_m
    for index, iteration in enumerate(result.solution.iterations):
        print(f"\nWLS iteration {index}")
        raw_residual = raw_pseudorange - iteration.predicted_observation_m.detach().numpy()
        show("state before", state_before)
        show("raw residual P-h(x) [m]", raw_residual)
        show("effective residual P-h(x)-b [m]", iteration.residual_v_m.detach().numpy())
        show("H", iteration.jacobian_H.detach().numpy())
        show("W", iteration.weight_matrix_W.detach().numpy())
        show("H^T W H", iteration.normal_matrix_HTWH.detach().numpy())
        show("H^T W v", iteration.rhs_HTWv.detach().numpy())
        show("delta state", iteration.delta_state.detach().numpy())
        show("state after", iteration.updated_state.detach().numpy())
        state_before = iteration.updated_state.detach().numpy()

    print("\nFinal physical result")
    show("estimated ECEF [m]", result.estimated_ecef_m)
    show("GT ECEF [m]", result.ground_truth_ecef_m)
    show("ENU error [m]", result.enu_error_m)
    print(f"2D error [m] = {result.error_2d_m:.15g}")
    print(f"3D error [m] = {result.error_3d_m:.15g}")
    print(f"WLS rank = {result.normal_matrix_rank}")
    print(f"WLS condition number = {result.normal_matrix_condition_number:.15g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
