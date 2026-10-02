#!/usr/bin/env python3
"""Compare archived and controlled BiasNet/WLS algebra on one real KLT1 epoch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .core import solve_paper_bias_position
except ImportError:
    from core import solve_paper_bias_position

from validation.paper_weightnet.held_out import (
    DATASET_SPECS,
    load_historical_modules,
    prepare_dataset,
    resolve_input_paths,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--tdl-dir", type=Path)
    parser.add_argument("--pyrtklib-site", type=Path)
    parser.add_argument("--epoch-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=here / "forward_equivalence.json")
    parser.add_argument("--trace", type=Path, default=here / "forward_equivalence_trace.npz")
    return parser.parse_args()


def maximum_absolute(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(difference))) if difference.size else 0.0


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    spec = DATASET_SPECS["KLT1"]
    inputs = resolve_input_paths(
        spec,
        data_root=args.data_root,
        tdl_dir=args.tdl_dir,
        pyrtklib_site=args.pyrtklib_site,
    )
    prepared = prepare_dataset(spec, inputs)
    epoch = prepared.epochs[args.epoch_index]
    prl, util = load_historical_modules(inputs)
    obs, nav, _station = util.read_obs(
        str(inputs.observation), inputs.ephemeris_patterns[0]
    )
    prl.sortobs(obs)
    native_epoch = util.split_obs(obs)[epoch.split_epoch_index]
    ols = util.get_ls_pnt_pos(native_epoch, nav)
    if not ols["status"]:
        raise RuntimeError("frozen comparison epoch no longer has a valid OLS solution")
    data = ols["data"]
    excluded = list(data["exclude"])
    satellites = [int(item) for item in data["sats"]]
    included = list(set(range(len(satellites))) - set(excluded))
    ordered = [index for index in range(len(satellites)) if index not in excluded]
    if included != ordered:
        raise RuntimeError("historical set row order differs from Jacobian row order")
    if not np.array_equal(np.asarray(ols["pos"]), epoch.initial_ols_state):
        raise RuntimeError("reloaded OLS state differs from prepared epoch")

    count = epoch.features.shape[0]
    # A deterministic, nonuniform supplied correction exercises row alignment
    # and sign independently of any checkpoint or network initialization.
    supplied_bias = np.linspace(-4.0, 4.0, count, dtype=np.float64)
    corrected_pseudorange = epoch.corrected_pseudorange_m - supplied_bias
    state = torch.as_tensor(epoch.initial_ols_state, dtype=torch.float64)
    delta = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
    raw_pr = torch.as_tensor(epoch.corrected_pseudorange_m, dtype=torch.float64).reshape(-1, 1)
    bias = torch.as_tensor(supplied_bias, dtype=torch.float64).reshape(-1, 1)
    archived: list[dict[str, np.ndarray]] = []
    while float(torch.linalg.vector_norm(delta)) > 1.0e-4 and len(archived) < 10:
        before = state.clone()
        h, predicted, _azel, ex, active, _counts, _vels, _vions, _vtrps = (
            util.H_matrix_prl_torch(
                data["eph"],
                state,
                data["dts"],
                native_epoch.data[0].time,
                nav,
                satellites,
                excluded,
            )
        )
        residual_rows = list(set(range(len(satellites))) - set(ex))
        if residual_rows != ordered:
            raise RuntimeError("satellite rows changed during archived BiasNet solve")
        uncorrected_residual = raw_pr - predicted
        residual = uncorrected_residual - bias
        weight = torch.eye(count, dtype=torch.float64)
        normal = h.T @ weight @ h
        rhs = h.T @ weight @ residual
        delta = util.wls_solve_torch(h, uncorrected_residual, weight, bias)
        updated = state.clone()
        updated[active] = updated[active] + delta.reshape(-1)
        archived.append(
            {
                "state_before": before.numpy().copy(),
                "corrected_pseudorange": corrected_pseudorange.copy(),
                "predicted_observation": predicted.detach().numpy().reshape(-1),
                "residual_v": residual.detach().numpy().reshape(-1),
                "H": h.detach().numpy().copy(),
                "W": weight.numpy().copy(),
                "normal_HTWH": normal.detach().numpy().copy(),
                "rhs_HTWv": rhs.detach().numpy().reshape(-1),
                "delta_state": delta.detach().numpy().reshape(-1),
                "updated_state": updated.detach().numpy().copy(),
            }
        )
        state = updated
    archived_final = state.detach().numpy().copy()

    controlled = solve_paper_bias_position(
        epoch.satellite_positions_ecef_m,
        epoch.satellite_clock_bias_s,
        epoch.corrected_pseudorange_m,
        epoch.system_clock_indices,
        supplied_bias,
        epoch.initial_ols_state,
        return_trace=True,
    )
    if len(controlled.iterations) != len(archived):
        raise RuntimeError("archived and controlled solvers used different iteration counts")
    names = (
        "corrected_pseudorange",
        "predicted_observation",
        "residual_v",
        "H",
        "W",
        "normal_HTWH",
        "rhs_HTWv",
        "delta_state",
        "updated_state",
    )
    maxima = {name: 0.0 for name in names}
    trace: dict[str, np.ndarray] = {
        "satellite_ids": epoch.satellite_ids,
        "features": epoch.features,
        "supplied_bias_m": supplied_bias,
        "raw_corrected_pseudorange_m": epoch.corrected_pseudorange_m,
        "bias_corrected_pseudorange_m": corrected_pseudorange,
        "archived_final_state": archived_final,
        "controlled_final_state": controlled.state.detach().numpy(),
    }
    for index, (historical, iteration) in enumerate(
        zip(archived, controlled.iterations, strict=True)
    ):
        current = {
            "state_before": (
                epoch.initial_ols_state
                if index == 0
                else controlled.iterations[index - 1].updated_state.detach().numpy()
            ),
            "corrected_pseudorange": controlled.bias_corrected_pseudorange_m.detach().numpy(),
            "predicted_observation": iteration.predicted_observation_m.detach().numpy(),
            "residual_v": iteration.residual_v_m.detach().numpy(),
            "H": iteration.jacobian_H.detach().numpy(),
            "W": iteration.weight_matrix_W.detach().numpy(),
            "normal_HTWH": iteration.normal_matrix_HTWH.detach().numpy(),
            "rhs_HTWv": iteration.rhs_HTWv.detach().numpy(),
            "delta_state": iteration.delta_state.detach().numpy(),
            "updated_state": iteration.updated_state.detach().numpy(),
        }
        for name in names:
            maxima[name] = max(maxima[name], maximum_absolute(historical[name], current[name]))
        for name, value in historical.items():
            trace[f"archived_iteration_{index}_{name}"] = value
        for name, value in current.items():
            trace[f"controlled_iteration_{index}_{name}"] = np.asarray(value)
    final_discrepancy = maximum_absolute(archived_final, controlled.state.detach().numpy())
    if max(maxima.values()) > 1.0e-8 or final_discrepancy > 1.0e-8:
        raise RuntimeError("forward-equivalence discrepancy exceeds 1e-8")

    args.trace.resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.trace.resolve(), **trace)
    output = {
        "status": "passed",
        "bias_source": "deterministic supplied vector, independent of checkpoint",
        "bias_sign_convention": "corrected pseudorange = RTKLIB-corrected pseudorange - bias",
        "bias_units": "metres",
        "epoch": {
            "dataset": "KLT1",
            "valid_epoch_index": args.epoch_index,
            "split_epoch_index": epoch.split_epoch_index,
            "time_gpst_like": epoch.epoch_time,
            "satellites": epoch.satellite_ids.tolist(),
        },
        "iterations": len(archived),
        "weighting": "identity W (equal weight)",
        "maximum_absolute_discrepancies": maxima,
        "final_state_maximum_absolute_discrepancy": final_discrepancy,
        "trace": {
            "path": str(args.trace.resolve()),
            "sha256": sha256(args.trace.resolve()),
        },
        "preserved_historical_defect": (
            "Torch H path gives satazel an allocated but unpopulated LOS vector, "
            "making atmospheric corrections zero"
        ),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
