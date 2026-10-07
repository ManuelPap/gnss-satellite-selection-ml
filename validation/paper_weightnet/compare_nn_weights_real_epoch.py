#!/usr/bin/env python3
"""Generate KLT1 NN weights and compare archived versus controlled WLS."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from core import WeightNet, construct_features  # noqa: E402
from gnss_satellite_selection_ml.paper_observation_model import (  # noqa: E402
    paper_wls_iteration,
)
from validation.ibiza_generalization.runtime_cache import (  # noqa: E402
    DEFAULT_RUNTIME_DIR,
    import_pyrtklib,
    load_rtk_util,
    resolve_runtime,
)


EXPECTED_EPOCH = 1623296154.005
EXPECTED_EPOCH_INDEX = 2382
EXPECTED_SATELLITES = ("G01", "G03", "G07", "G14", "G21", "G22", "G28", "G30")
EXPECTED_OLS_STATE = np.asarray(
    [
        -2417850.6248949184,
        5384790.762720588,
        2408319.1245824904,
        1581580.6444318756,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
OBSERVATION_SHA256 = "f722557326d1d32c42e023d4e78515e885d21c8ae824e79460bef61c67b9b5c4"
NAVIGATION_SHA256 = {
    "hksc161d.21f": "64e8e3ec2f4a9eeb17379a499e5779d378978b834a1a1441240e487e7ce23768",
    "hksc161d.21g": "0bddf6292d39f00845e9038f7f87dcecc403ec9944ca144b7345bcb233d8e660",
    "hksc161d.21l": "795de82407d394097628a7a2595e284d666df3dc62c4ad5a34896d6de05b84e3",
    "hksc161d.21m": "9583d5e061d46f3de29b6f783332dda9e7ff61d62041f2c8a031f6b45628e376",
    "hksc161d.21n": "a5bc8ab35fe0c80f91d0e57517b495be6563385d235835fd6aa063d73bb7072c",
    "hksc161d.21o": "7335032796f9b46176c359e8a39cc3f8496dd1f3fd6003fcfb9a794aafe88dd6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--ephemeris-glob", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "checkpoints/paper_weightnet/weightnet_3d.pth",
    )
    parser.add_argument(
        "--training-metrics", type=Path, default=HERE / "training_metrics.json"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "klt1_nn_weight_comparison.json"
    )
    parser.add_argument("--trace", type=Path, default=HERE / "klt1_nn_weight_trace.npz")
    parser.add_argument("--start-time", type=float, default=1623296154.0)
    parser.add_argument("--end-time", type=float, default=1623296357.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def satellite_id(prl: object, satellite: int) -> str:
    text = prl.Arr1Dchar(4)
    prl.satno2id(satellite, text)
    return str(text[0])


def gps_subset(prl: object, epoch: object) -> tuple[object, list[int]]:
    rows = [
        index
        for index in range(epoch.n)
        if satellite_id(prl, epoch.data[index].sat).startswith("G")
    ]
    result = prl.obs_t()
    result.data = prl.Arr1Dobsd_t(len(rows))
    for target, source in enumerate(rows):
        result.data[target] = epoch.data[source]
    result.n = len(rows)
    result.nmax = len(rows)
    return result, rows


def native_array(values: object, length: int) -> np.ndarray:
    return np.asarray([values[index] for index in range(length)], dtype=np.float64)


def maximum_absolute(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(difference))) if difference.size else 0.0


def main() -> int:
    args = parse_args()
    observation_path = args.observation.resolve()
    checkpoint_path = args.checkpoint.resolve()
    metrics_path = args.training_metrics.resolve()
    runtime, _runtime_manifest = resolve_runtime(args.runtime_dir)
    tdl_dir = runtime.tdl_dir
    verify_hash(observation_path, OBSERVATION_SHA256, "KLT rover observation")
    ephemeris_paths = [Path(item).resolve() for item in sorted(glob.glob(args.ephemeris_glob))]
    if {path.name for path in ephemeris_paths} != set(NAVIGATION_SHA256):
        raise RuntimeError("unexpected navigation wildcard contents")
    for path in ephemeris_paths:
        verify_hash(path, NAVIGATION_SHA256[path.name], f"navigation {path.name}")
    if not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("trained checkpoint or training metrics are missing")
    training_metrics = json.loads(metrics_path.read_text())
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != training_metrics["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint hash does not match the training record")

    prl = import_pyrtklib(runtime.pyrtklib_site)
    util = load_rtk_util(tdl_dir, module_name="paper_weight_comparison_rtk_util")

    if ".to('cuda')" in (tdl_dir / "rtk_util.py").read_text():
        raise RuntimeError(
            "cached TDL runtime contains CUDA placement; rebuild the paper runtime"
        )
    obs, nav, _station = util.read_obs(str(observation_path), args.ephemeris_glob)
    prl.sortobs(obs)
    epochs = util.split_obs(obs)
    selected = None
    for epoch_index, epoch in enumerate(epochs):
        epoch_time = float(epoch.data[0].time.time + epoch.data[0].time.sec)
        if not (epoch_time > args.start_time and epoch_time < args.end_time):
            continue
        gps, source_rows = gps_subset(prl, epoch)
        if gps.n < 4:
            continue
        result = util.get_ls_pnt_pos(gps, nav)
        if result["status"]:
            selected = epoch_index, epoch_time, gps, source_rows, result
            break
    if selected is None:
        raise RuntimeError("no valid KLT1 GPS epoch found")
    epoch_index, epoch_time, epoch, source_rows, ols_result = selected
    if epoch_index != EXPECTED_EPOCH_INDEX or epoch_time != EXPECTED_EPOCH:
        raise RuntimeError(f"selected wrong epoch: {epoch_index}/{epoch_time}")
    ols_state = np.asarray(ols_result["pos"], dtype=np.float64)
    if not np.allclose(ols_state, EXPECTED_OLS_STATE, rtol=0.0, atol=1.0e-7):
        raise RuntimeError("GPS-only OLS initializer differs from frozen validation")

    data = ols_result["data"]
    excluded = list(data["exclude"])
    satellites = list(data["sats"])
    included_rows = list(set(range(len(satellites))) - set(excluded))
    ordered_rows = [index for index in range(len(satellites)) if index not in excluded]
    if included_rows != ordered_rows:
        raise RuntimeError("legacy residual row order differs from Jacobian row order")
    satellite_ids = tuple(satellite_id(prl, satellites[index]) for index in included_rows)
    if satellite_ids != EXPECTED_SATELLITES:
        raise RuntimeError(f"unexpected satellite order: {satellite_ids}")
    elevation = np.delete(
        np.asarray(data["azel"], dtype=np.float64).reshape(-1, 2), excluded, axis=0
    )[:, 1]
    features = construct_features(
        np.asarray(data["SNR"], dtype=np.float64),
        elevation,
        np.asarray(data["residual"], dtype=np.float64),
    )

    model = WeightNet()
    model.double()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        weights_torch = model(feature_tensor).squeeze(-1)
    if not bool(torch.all(torch.isfinite(weights_torch))):
        raise RuntimeError("WeightNet generated a non-finite KLT1 weight")
    if not bool(torch.all((weights_torch >= 0.0) & (weights_torch <= 10.0))):
        raise RuntimeError("WeightNet generated a weight outside [0, 10]")
    weights = weights_torch.numpy().copy()
    weight_by_satellite = dict(zip(satellite_ids, weights.tolist(), strict=True))

    positions_raw = data["eph"]
    clock_raw = data["dts"]
    positions_full = native_array(positions_raw, len(satellites) * 6).reshape(-1, 6)
    clock_full = native_array(clock_raw, len(satellites) * 2).reshape(-1, 2)
    corrected_pseudorange = np.asarray(data["prs"], dtype=np.float64).reshape(-1)
    positions = positions_full[included_rows, :3]
    clock_bias = clock_full[included_rows, 0]

    paper_iterations: list[dict[str, np.ndarray]] = []
    state = torch.as_tensor(ols_state, dtype=torch.float64)
    delta = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
    while float(torch.linalg.vector_norm(delta)) > 1.0e-4 and len(paper_iterations) < 10:
        state_before = state.clone()
        h, predicted, _azel, ex, active, _counts, _vels, _vions, _vtrps = (
            util.H_matrix_prl_torch(
                positions_raw,
                state,
                clock_raw,
                epoch.data[0].time,
                nav,
                satellites,
                excluded,
            )
        )
        residual_rows = list(set(range(len(satellites))) - set(ex))
        h_rows = [index for index in range(len(satellites)) if index not in ex]
        if residual_rows != h_rows or h_rows != included_rows:
            raise RuntimeError("satellite-row alignment changed during paper WLS")
        aligned_weights = np.asarray(
            [weight_by_satellite[satellite_id(prl, satellites[row])] for row in h_rows],
            dtype=np.float64,
        )
        weight_matrix = torch.diag(torch.as_tensor(aligned_weights, dtype=torch.float64))
        residual = torch.as_tensor(
            corrected_pseudorange, dtype=torch.float64
        ).reshape(-1, 1) - predicted
        normal = h.T @ weight_matrix @ h
        rhs = h.T @ weight_matrix @ residual
        delta = util.wls_solve_torch(h, residual, weight_matrix)
        updated = state.clone()
        updated[active] = updated[active] + delta.reshape(-1)
        paper_iterations.append(
            {
                "state_before": state_before.numpy().copy(),
                "predicted_observation": predicted.detach().numpy().reshape(-1),
                "H": h.detach().numpy().copy(),
                "residual_v": residual.detach().numpy().reshape(-1),
                "weight_vector": aligned_weights,
                "W": weight_matrix.numpy().copy(),
                "normal_HTWH": normal.detach().numpy().copy(),
                "rhs_HTWv": rhs.detach().numpy().reshape(-1),
                "delta_state": delta.detach().numpy().reshape(-1),
                "updated_state": updated.numpy().copy(),
            }
        )
        state = updated
    if len(paper_iterations) >= 10:
        raise RuntimeError("paper-era WLS reached its iteration limit")
    paper_final_state = state.numpy().copy()

    comparison_names = (
        "predicted_observation",
        "H",
        "residual_v",
        "weight_vector",
        "W",
        "normal_HTWH",
        "rhs_HTWv",
        "delta_state",
        "updated_state",
    )
    maxima = {name: 0.0 for name in comparison_names}
    controlled_iterations: list[dict[str, np.ndarray]] = []
    controlled_state = ols_state.copy()
    for paper in paper_iterations:
        controlled = paper_wls_iteration(
            positions,
            clock_bias,
            corrected_pseudorange,
            weights,
            controlled_state,
            satellite_ids,
            satellite_rows=np.arange(len(satellite_ids), dtype=np.int64),
        )
        record = {
            "state_before": controlled_state.copy(),
            "predicted_observation": controlled.observation.predicted_observation_m,
            "H": controlled.observation.jacobian_H,
            "residual_v": controlled.observation.residual_v_m,
            "weight_vector": weights,
            "W": controlled.weight_matrix_W,
            "normal_HTWH": controlled.normal_matrix_HTWH,
            "rhs_HTWv": controlled.rhs_HTWv,
            "delta_state": controlled.delta_state,
            "updated_state": controlled.updated_state,
        }
        controlled_iterations.append(record)
        for name in comparison_names:
            maxima[name] = max(maxima[name], maximum_absolute(paper[name], record[name]))
        controlled_state = controlled.updated_state
    final_state_discrepancy = maximum_absolute(paper_final_state, controlled_state)

    trace_values: dict[str, np.ndarray] = {
        "features": features,
        "weights": weights,
        "satellite_ids": np.asarray(satellite_ids, dtype="U3"),
        "satellite_positions_ecef_m": positions,
        "satellite_clock_bias_s": clock_bias,
        "corrected_pseudorange_m": corrected_pseudorange,
        "ols_initial_state": ols_state,
        "paper_final_state": paper_final_state,
        "controlled_final_state": controlled_state,
    }
    for index, (paper, controlled) in enumerate(
        zip(paper_iterations, controlled_iterations, strict=True)
    ):
        for name, value in paper.items():
            trace_values[f"paper_iteration_{index}_{name}"] = value
        for name, value in controlled.items():
            trace_values[f"controlled_iteration_{index}_{name}"] = value
    args.trace.resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.trace.resolve(), **trace_values)

    output = {
        "status": "passed",
        "epoch": {
            "split_index": epoch_index,
            "time_gpst_like": epoch_time,
            "source_rows": source_rows,
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash},
        "features": {
            satellite: row.tolist()
            for satellite, row in zip(satellite_ids, features, strict=True)
        },
        "weights": weight_by_satellite,
        "weight_range_check": {
            "all_finite": True,
            "expected_closed_range": [0.0, 10.0],
            "minimum": float(np.min(weights)),
            "maximum": float(np.max(weights)),
        },
        "satellite_row_alignment_exact": True,
        "iterations": len(paper_iterations),
        "paper_final_state": paper_final_state.tolist(),
        "controlled_final_state": controlled_state.tolist(),
        "maximum_absolute_discrepancies": maxima,
        "final_state_maximum_absolute_discrepancy": final_state_discrepancy,
        "trace": {"path": str(args.trace.resolve()), "sha256": sha256(args.trace.resolve())},
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("KLT1 NN weights:")
    for satellite in satellite_ids:
        print(f"  {satellite}: {weight_by_satellite[satellite]:.17g}")
    print(f"iterations: {len(paper_iterations)}")
    print("maximum absolute discrepancies:")
    for name, value in maxima.items():
        print(f"  {name}: {value:.17e}")
    print(f"  final_state: {final_state_discrepancy:.17e}")
    print(f"result: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
