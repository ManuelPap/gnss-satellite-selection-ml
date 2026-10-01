#!/usr/bin/env python3
"""Print the complete numerical path for one valid held-out epoch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from held_out import (
    FROZEN_MODEL_MEAN,
    FROZEN_MODEL_STD,
    SPEED_OF_LIGHT_M_S,
    common_input_arguments,
    evaluate_epoch,
    inputs_from_args,
    load_frozen_weightnet,
    prepare_dataset,
    repository_root,
)


def vector(values: np.ndarray) -> str:
    return np.array2string(
        np.asarray(values),
        precision=15,
        suppress_small=False,
        separator=", ",
        max_line_width=240,
    )


def matrix(values: np.ndarray) -> str:
    return np.array2string(
        np.asarray(values),
        precision=15,
        suppress_small=False,
        separator=", ",
        max_line_width=240,
        threshold=np.inf,
    )


def print_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    rendered = [[str(item) for item in headers]]
    rendered.extend([[str(item) for item in row] for row in rows])
    widths = [max(len(row[index]) for row in rendered) for index in range(len(headers))]
    for row_index, row in enumerate(rendered):
        print("  ".join(item.rjust(widths[index]) for index, item in enumerate(row)))
        if row_index == 0:
            print("  ".join("-" * width for width in widths))


def main() -> int:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    common_input_arguments(parser)
    parser.add_argument("--epoch-index", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "checkpoints/paper_weightnet/weightnet_3d.pth",
    )
    args = parser.parse_args()

    spec, inputs = inputs_from_args(args)
    prepared = prepare_dataset(spec, inputs)
    if not 0 <= args.epoch_index < len(prepared.epochs):
        raise IndexError(
            f"--epoch-index must be in [0, {len(prepared.epochs) - 1}] after OLS filtering"
        )
    epoch = prepared.epochs[args.epoch_index]
    model = load_frozen_weightnet(args.checkpoint)
    result = evaluate_epoch(model, epoch)

    print("A. epoch timestamp")
    print(f"GNSS timestamp [GPST-like Unix seconds]: {epoch.epoch_time:.9f}")
    print(f"valid evaluation epoch index: {epoch.valid_epoch_index}")
    print(f"raw candidate epoch index: {epoch.candidate_epoch_index}")
    print(f"split observation epoch index: {epoch.split_epoch_index}")

    print("\nB. ground-truth timestamp and time difference")
    print(f"GT timestamp after historical +18 s [s]: {epoch.gt_time:.9f}")
    print(f"GT minus GNSS time [s]: {epoch.gt_time_difference_s:.12g}")
    print(
        "matched GT geodetic [latitude deg, longitude deg, ellipsoidal height m]: "
        + vector(epoch.ground_truth_geodetic_deg_m)
    )

    print("\nC. valid satellite IDs")
    print(" ".join(epoch.satellite_ids.tolist()))

    print("\nRaw satellite observations retained by OLS")
    print_table(
        ("PRN", "raw P[0] [m]", "raw SNR[0]", "C/N0=SNR[0]/1000"),
        [
            (
                sat,
                f"{raw_pr:.6f}",
                f"{raw_snr:.3f}",
                f"{feature[0]:.9f}",
            )
            for sat, raw_pr, raw_snr, feature in zip(
                epoch.satellite_ids,
                epoch.raw_pseudorange_m,
                epoch.raw_snr_units,
                epoch.features,
                strict=True,
            )
        ],
    )

    print("\nD. corrected pseudorange per satellite")
    print_table(
        ("PRN", "corrected pseudorange [m]"),
        [
            (sat, f"{pseudorange:.9f}")
            for sat, pseudorange in zip(
                epoch.satellite_ids, epoch.corrected_pseudorange_m, strict=True
            )
        ],
    )

    print("\nE. satellite ECEF coordinates and satellite clock correction")
    print_table(
        (
            "PRN",
            "sat X [m]",
            "sat Y [m]",
            "sat Z [m]",
            "dts [s]",
            "-c*dts [m]",
        ),
        [
            (
                sat,
                f"{position[0]:.9f}",
                f"{position[1]:.9f}",
                f"{position[2]:.9f}",
                f"{clock:.15e}",
                f"{-SPEED_OF_LIGHT_M_S * clock:.9f}",
            )
            for sat, position, clock in zip(
                epoch.satellite_ids,
                epoch.satellite_positions_ecef_m,
                epoch.satellite_clock_bias_s,
                strict=True,
            )
        ],
    )

    print("\nF. equal-weight OLS state")
    print("[x m, y m, z m, b_G m, b_C m, b_E m, b_R m]")
    print(vector(epoch.initial_ols_state))

    print("\nG. raw feature table")
    print_table(
        ("PRN", "C/N0", "elevation [rad]", "OLS residual [m]"),
        [
            (sat, f"{row[0]:.9f}", f"{row[1]:.15g}", f"{row[2]:.15g}")
            for sat, row in zip(epoch.satellite_ids, epoch.features, strict=True)
        ],
    )

    print("\nH. frozen KLT3 normalization values used")
    print("checkpoint mean (float32 then double): " + vector(FROZEN_MODEL_MEAN))
    print("checkpoint population std (float32 then double): " + vector(FROZEN_MODEL_STD))
    print("formula: x_normalized = (float32(x) - mean_KLT3) / std_KLT3")

    print("\nI. normalized feature table")
    print_table(
        ("PRN", "normalized C/N0", "normalized elevation", "normalized OLS residual"),
        [
            (sat, f"{row[0]:.15g}", f"{row[1]:.15g}", f"{row[2]:.15g}")
            for sat, row in zip(
                epoch.satellite_ids, result.normalized_features, strict=True
            )
        ],
    )

    print("\nJ. NN-generated weight for every satellite")
    print_table(
        ("PRN", "WeightNet weight"),
        [
            (sat, f"{weight:.17g}")
            for sat, weight in zip(epoch.satellite_ids, result.weights, strict=True)
        ],
    )

    print("\nK. WLS diagonal weight matrix values")
    print("diag(W) = " + vector(result.weights))

    print("\nL. every released weighted Gauss-Newton/WLS iteration")
    state_before = epoch.initial_ols_state.copy()
    for index, iteration in enumerate(result.solution.iterations):
        print(f"\niteration {index}")
        print("active seven-state indices: " + str(iteration.active_state_indices))
        print("state before update: " + vector(state_before))
        print("residual v [m]: " + vector(iteration.residual_v_m.detach().numpy()))
        print("H:")
        print(matrix(iteration.jacobian_H.detach().numpy()))
        print("H^T W H:")
        print(matrix(iteration.normal_matrix_HTWH.detach().numpy()))
        print("H^T W v: " + vector(iteration.rhs_HTWv.detach().numpy()))
        print("delta (active-state order): " + vector(iteration.delta_state.detach().numpy()))
        state_after = iteration.updated_state.detach().numpy()
        print("state after update: " + vector(state_after))
        state_before = state_after.copy()
    print("historical WLS status: " + result.historical_wls_status)
    print(
        "last pre-update residual norm [m]: "
        f"{result.historical_wls_residual_norm_m:.15g}"
    )

    print("\nM. final ECEF position")
    print("[X m, Y m, Z m] = " + vector(result.estimated_ecef_m))
    print(
        "historical intermediate estimated geodetic [lat deg, lon deg, h m] = "
        + vector(result.estimated_geodetic_deg_m)
    )

    print("\nN. matched ground-truth position")
    print("geodetic [lat deg, lon deg, h m] = " + vector(epoch.ground_truth_geodetic_deg_m))
    print("ECEF [X m, Y m, Z m] = " + vector(result.ground_truth_ecef_m))

    print("\nO. ENU error")
    print("[East m, North m, Up m] = " + vector(result.enu_error_m))

    print("\nP. 2D error")
    print("sqrt(E^2 + N^2) [m] = " + f"{result.error_2d_m:.15g}")

    print("\nQ. 3D error")
    print("sqrt(E^2 + N^2 + U^2) [m] = " + f"{result.error_3d_m:.15g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
