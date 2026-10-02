#!/usr/bin/env python3
"""Print a non-black-box BiasNet trace for one held-out KLT epoch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .held_out import (
        common_input_arguments,
        evaluate_epoch,
        inputs_from_args,
        load_frozen_biasnet,
        prepare_dataset,
        repository_root,
    )
except ImportError:
    from held_out import (
        common_input_arguments,
        evaluate_epoch,
        inputs_from_args,
        load_frozen_biasnet,
        prepare_dataset,
        repository_root,
    )


def render(name: str, value: object) -> None:
    print(name)
    print(np.asarray(value))


def main() -> int:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    common_input_arguments(parser)
    parser.set_defaults(dataset="KLT1")
    parser.add_argument("--epoch-index", type=int, default=0)
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
    args = parser.parse_args()
    np.set_printoptions(precision=15, linewidth=240, threshold=np.inf, suppress=False)

    spec, inputs = inputs_from_args(args)
    prepared = prepare_dataset(spec, inputs)
    if not 0 <= args.epoch_index < len(prepared.epochs):
        raise IndexError(args.epoch_index)
    epoch = prepared.epochs[args.epoch_index]
    model = load_frozen_biasnet(args.checkpoint, args.training_metrics)
    result = evaluate_epoch(model, epoch)

    print(f"dataset: {spec.name}")
    print(f"valid epoch index: {epoch.valid_epoch_index}")
    print(f"split epoch index: {epoch.split_epoch_index}")
    print(f"GNSS timestamp: {epoch.epoch_time:.9f}")
    print("bias convention: positive b subtracts metres; P_bias_corrected = P_RTKLIB - b")
    print("satellite rows:")
    for row, satellite in enumerate(epoch.satellite_ids):
        feature = epoch.features[row]
        normalized = result.normalized_features[row]
        print(
            f"  row={row:02d} PRN={satellite} "
            f"raw_RINEX_P1_m={epoch.raw_pseudorange_m[row]:.15g} "
            f"RTKLIB_corrected_P_m={epoch.corrected_pseudorange_m[row]:.15g} "
            f"C/N0={feature[0]:.15g} elevation_rad={feature[1]:.15g} "
            f"OLS_residual_m={feature[2]:.15g} "
            f"raw_feature={feature.tolist()} "
            f"normalized_feature={normalized.tolist()} "
            f"predicted_bias_m={result.predicted_bias_m[row]:.15g} "
            f"BiasNet_corrected_P_m={result.bias_corrected_pseudorange_m[row]:.15g}"
        )

    previous_state = epoch.initial_ols_state
    for index, iteration in enumerate(result.solution.iterations):
        print(f"\nsolver iteration {index + 1}")
        render("state before [x,y,z,b_G,b_C,b_E,b_R] m", previous_state)
        render("v = (P - b) - predicted observation [m]", iteration.residual_v_m.detach().numpy())
        render("H", iteration.jacobian_H.detach().numpy())
        render("W (identity in standalone bias path)", iteration.weight_matrix_W.detach().numpy())
        render("H^T W H", iteration.normal_matrix_HTWH.detach().numpy())
        render("H^T W v", iteration.rhs_HTWv.detach().numpy())
        render("delta active state [m]", iteration.delta_state.detach().numpy())
        render("state after [x,y,z,b_G,b_C,b_E,b_R] m", iteration.updated_state.detach().numpy())
        previous_state = iteration.updated_state.detach().numpy()

    print()
    render("estimated ECEF [m]", result.estimated_ecef_m)
    render("GT ECEF [m]", result.ground_truth_ecef_m)
    render("ENU error [m]", result.enu_error_m)
    print(f"2D error [m]: {result.error_2d_m:.15g}")
    print(f"3D error [m]: {result.error_3d_m:.15g}")
    print(f"historical WLS status: {result.historical_wls_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
