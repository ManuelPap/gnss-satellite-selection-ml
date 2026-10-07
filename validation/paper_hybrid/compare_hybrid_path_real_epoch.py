#!/usr/bin/env python3
"""Compare archived and controlled hybrid/WLS algebra on one real KLT1 epoch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from validation.ibiza_generalization.runtime_cache import (  # noqa: E402
    DEFAULT_RUNTIME_DIR,
)

try:
    from .core import HybridShareNet, solve_paper_hybrid_position
except ImportError:
    from core import HybridShareNet, solve_paper_hybrid_position

from validation.paper_weightnet.held_out import (
    DATASET_SPECS,
    load_historical_modules,
    prepare_dataset,
    resolve_input_paths,
)


MODEL_SHA256 = "12bab5692b899e07681c5b130aaf69b54334328d3689426db1187b8594e9a2d2"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--epoch-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=here / "forward_equivalence.json")
    parser.add_argument("--trace", type=Path, default=here / "forward_equivalence_trace.npz")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maximum_absolute(left: object, right: object) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(difference))) if difference.size else 0.0


def load_archived_model_module(path: Path) -> object:
    if sha256(path) != MODEL_SHA256:
        raise RuntimeError("historical model.py hash mismatch")
    spec = importlib.util.spec_from_file_location("tdl_dd5eac6_model_hybrid_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load archived model module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    spec = DATASET_SPECS["KLT1"]
    inputs = resolve_input_paths(
        spec,
        data_root=args.data_root,
        runtime_dir=args.runtime_dir,
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
        raise RuntimeError("frozen comparison epoch lacks a valid OLS solution")
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
    # The supplied pre-transform columns exercise both released output
    # transformations independently of any checkpoint or random seed.
    supplied_raw = np.column_stack(
        (
            np.linspace(-2.0, 2.0, count, dtype=np.float64),
            np.linspace(0.25, 5.0, count, dtype=np.float64),
        )
    )
    archived_model = load_archived_model_module(inputs.tdl_dir / "model.py")
    archived_transform = archived_model.HybridShareNet()
    archived_transform.seq = nn.Identity()
    archived_transform.double()
    archived_weight_t, archived_bias_t = archived_transform(
        torch.as_tensor(supplied_raw, dtype=torch.float64)
    )
    controlled_weight_t, controlled_bias_t = HybridShareNet.transform_raw(
        torch.as_tensor(supplied_raw, dtype=torch.float64)
    )
    weight = archived_weight_t.detach().numpy()
    bias = archived_bias_t.detach().numpy()
    corrected = epoch.corrected_pseudorange_m - bias

    state = torch.as_tensor(epoch.initial_ols_state, dtype=torch.float64)
    delta = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
    raw_pr = torch.as_tensor(
        epoch.corrected_pseudorange_m, dtype=torch.float64
    ).reshape(-1, 1)
    bias_column = torch.as_tensor(bias, dtype=torch.float64).reshape(-1, 1)
    weight_matrix = torch.diag(torch.as_tensor(weight, dtype=torch.float64))
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
            raise RuntimeError("satellite rows changed during archived hybrid solve")
        raw_residual = raw_pr - predicted
        effective_residual = raw_residual - bias_column
        normal = h.T @ weight_matrix @ h
        rhs = h.T @ weight_matrix @ effective_residual
        delta = util.wls_solve_torch(
            h, raw_residual, weight_matrix, bias_column
        )
        updated = state.clone()
        updated[active] = updated[active] + delta.reshape(-1)
        archived.append(
            {
                "state_before": before.numpy().copy(),
                "corrected_pseudorange": corrected.copy(),
                "predicted_observation": predicted.detach().numpy().reshape(-1),
                "raw_residual": raw_residual.detach().numpy().reshape(-1),
                "effective_residual": effective_residual.detach().numpy().reshape(-1),
                "H": h.detach().numpy().copy(),
                "W": weight_matrix.numpy().copy(),
                "normal_HTWH": normal.detach().numpy().copy(),
                "rhs_HTWv": rhs.detach().numpy().reshape(-1),
                "delta_state": delta.detach().numpy().reshape(-1),
                "updated_state": updated.detach().numpy().copy(),
            }
        )
        state = updated
    archived_final = state.detach().numpy().copy()

    controlled = solve_paper_hybrid_position(
        epoch.satellite_positions_ecef_m,
        epoch.satellite_clock_bias_s,
        epoch.corrected_pseudorange_m,
        epoch.system_clock_indices,
        controlled_weight_t,
        controlled_bias_t,
        epoch.initial_ols_state,
        return_trace=True,
    )
    if len(controlled.iterations) != len(archived):
        raise RuntimeError("archived and controlled solvers used different iterations")
    names = (
        "state_before",
        "corrected_pseudorange",
        "predicted_observation",
        "raw_residual",
        "effective_residual",
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
        "supplied_raw_output": supplied_raw,
        "predicted_weight": weight,
        "predicted_bias_m": bias,
        "raw_corrected_pseudorange_m": epoch.corrected_pseudorange_m,
        "bias_corrected_pseudorange_m": corrected,
        "archived_final_state": archived_final,
        "controlled_final_state": controlled.state.detach().numpy(),
    }
    previous = np.asarray(epoch.initial_ols_state)
    for index, (historical, iteration) in enumerate(
        zip(archived, controlled.iterations, strict=True)
    ):
        raw_residual = (
            controlled.raw_corrected_pseudorange_m
            - iteration.predicted_observation_m
        ).detach().numpy()
        current = {
            "state_before": previous,
            "corrected_pseudorange": controlled.bias_corrected_pseudorange_m.detach().numpy(),
            "predicted_observation": iteration.predicted_observation_m.detach().numpy(),
            "raw_residual": raw_residual,
            "effective_residual": iteration.residual_v_m.detach().numpy(),
            "H": iteration.jacobian_H.detach().numpy(),
            "W": iteration.weight_matrix_W.detach().numpy(),
            "normal_HTWH": iteration.normal_matrix_HTWH.detach().numpy(),
            "rhs_HTWv": iteration.rhs_HTWv.detach().numpy(),
            "delta_state": iteration.delta_state.detach().numpy(),
            "updated_state": iteration.updated_state.detach().numpy(),
        }
        previous = current["updated_state"]
        for name in names:
            maxima[name] = max(maxima[name], maximum_absolute(historical[name], current[name]))
            trace[f"archived_iteration_{index}_{name}"] = historical[name]
            trace[f"controlled_iteration_{index}_{name}"] = np.asarray(current[name])
    transformations = {
        "weight": maximum_absolute(archived_weight_t.detach(), controlled_weight_t.detach()),
        "bias": maximum_absolute(archived_bias_t.detach(), controlled_bias_t.detach()),
    }
    final_discrepancy = maximum_absolute(
        archived_final, controlled.state.detach().numpy()
    )
    if max((*maxima.values(), *transformations.values(), final_discrepancy)) > 1.0e-8:
        raise RuntimeError("forward-equivalence discrepancy exceeds 1e-8")

    args.trace.resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.trace.resolve(), **trace)
    output = {
        "status": "passed",
        "hybrid_output_source": "deterministic supplied raw two-column values, independent of checkpoint",
        "output_order": ["weight", "bias"],
        "raw_column_semantics": {"0": "weight logit", "1": "bias pre-ReLU in metres"},
        "transformations": {
            "weight": "sigmoid(raw[:,0]) then clamp [0,1]",
            "bias": "ReLU(raw[:,1])",
        },
        "transformation_maximum_absolute_discrepancies": transformations,
        "bias_sign_convention": "corrected pseudorange = RTKLIB-corrected pseudorange - bias",
        "bias_units": "metres",
        "weight_units": "dimensionless relative WLS coefficient",
        "epoch": {
            "dataset": "KLT1",
            "valid_epoch_index": args.epoch_index,
            "split_epoch_index": epoch.split_epoch_index,
            "time_gpst_like": epoch.epoch_time,
            "satellites": epoch.satellite_ids.tolist(),
        },
        "iterations": len(archived),
        "maximum_absolute_discrepancies": maxima,
        "final_state_maximum_absolute_discrepancy": final_discrepancy,
        "trace": {
            "path": str(args.trace.resolve()),
            "sha256": sha256(args.trace.resolve()),
        },
        "preserved_historical_defect": "Torch H path supplies an unpopulated LOS vector to atmosphere/elevation calls, producing zero atmospheric corrections",
    }
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
