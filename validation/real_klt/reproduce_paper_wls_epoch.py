#!/usr/bin/env python3
"""Reproduce one paper-era TDL-GNSS WLS trace on the original KLT1 data."""

from __future__ import annotations

import argparse
import glob
import hashlib
import inspect
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from validation.ibiza_generalization.runtime_cache import (  # noqa: E402
    DEFAULT_RUNTIME_DIR,
    PYRTKLIB_COMMIT,
    PYRTKLIB_VERSION,
    TDL_COMMIT,
    import_pyrtklib,
    load_rtk_util,
    resolve_runtime,
)


DATASET_URL = (
    "https://www.dropbox.com/scl/fi/d3urwaquf5ema5j0unmt4/"
    "data.zip?rlkey=tuwpx9pdzqtdvoeoqwhcc5gi8&st=wh5qhg6e&dl=1"
)
DATASET_ARCHIVE_SHA256 = (
    "2afd7b1e395f8494e6992d1e109f9e8d9d83d992bf6d446d83d91aaf46cfc721"
)
OBSERVATION_SHA256 = (
    "f722557326d1d32c42e023d4e78515e885d21c8ae824e79460bef61c67b9b5c4"
)
NAVIGATION_SHA256 = {
    "hksc161d.21f": "64e8e3ec2f4a9eeb17379a499e5779d378978b834a1a1441240e487e7ce23768",
    "hksc161d.21g": "0bddf6292d39f00845e9038f7f87dcecc403ec9944ca144b7345bcb233d8e660",
    "hksc161d.21l": "795de82407d394097628a7a2595e284d666df3dc62c4ad5a34896d6de05b84e3",
    "hksc161d.21m": "9583d5e061d46f3de29b6f783332dda9e7ff61d62041f2c8a031f6b45628e376",
    "hksc161d.21n": "a5bc8ab35fe0c80f91d0e57517b495be6563385d235835fd6aa063d73bb7072c",
    "hksc161d.21o": "7335032796f9b46176c359e8a39cc3f8496dd1f3fd6003fcfb9a794aafe88dd6",
}
EXPECTED_EPOCH = 1623296154.005
EXPECTED_EPOCH_INDEX = 2382
EXPECTED_SATELLITES = ("G01", "G03", "G07", "G14", "G21", "G22", "G28", "G30")
EXPECTED_OLS_STATE = np.array(
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
EXPECTED_FINAL_STATE = np.array(
    [
        -2417858.878882711,
        5384804.946954405,
        2408324.985628578,
        1581600.17818899,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
# These are the float64 values produced by the successful audit's
# torch.arange(0.5, 2.0, 0.2). Keeping the two expanded literals preserves
# that exact vector across both Torch and NumPy.
DETERMINISTIC_WEIGHTS = np.asarray(
    [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7000000000000002, 1.9000000000000001],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run archived TDL-GNSS H_matrix_prl_torch() and wls_solve_torch() "
            "for the first valid GPS-only KLT1 epoch."
        )
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME_DIR,
        help="Persistent generated paper runtime cache.",
    )
    parser.add_argument(
        "--observation",
        required=True,
        type=Path,
        help="Original COM38_210610_025603.obs from the public archive.",
    )
    parser.add_argument(
        "--ephemeris-glob",
        required=True,
        help="Quoted RTKLIB wildcard, for example extracted/.../hksc161d.21*.",
    )
    parser.add_argument(
        "--dataset-archive",
        required=True,
        type=Path,
        help="Downloaded original data.zip, retained for provenance hashing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here,
        help="Artifact directory (default: next to this script).",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=1623296154.0,
        help="Strict lower GPST-like time bound, matching the paper config.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=1623296357.0,
        help="Strict upper GPST-like time bound, matching the paper config.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_hash: str, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": actual_hash,
    }


def satellite_id(prl: object, sat: int) -> str:
    value = prl.Arr1Dchar(4)
    prl.satno2id(sat, value)
    return str(value[0])


def gps_subset(prl: object, epoch: object) -> tuple[object, list[int]]:
    rows = [
        index
        for index in range(epoch.n)
        if satellite_id(prl, epoch.data[index].sat).startswith("G")
    ]
    gps = prl.obs_t()
    gps.data = prl.Arr1Dobsd_t(len(rows))
    for compact_index, source_index in enumerate(rows):
        gps.data[compact_index] = epoch.data[source_index]
    gps.n = len(rows)
    gps.nmax = len(rows)
    return gps, rows


def as_numpy(values: object, length: int) -> np.ndarray:
    return np.array([values[index] for index in range(length)], dtype=np.float64)


def legacy_atmosphere_components(
    prl: object,
    satellite_positions: object,
    state: np.ndarray,
    epoch_time: object,
    nav: object,
    satellites: list[int],
    exclude: list[int],
) -> dict[str, np.ndarray]:
    """Replay instrumentation around the archived zero-LOS atmosphere calls."""
    count = len(satellites)
    receiver = prl.Arr1Ddouble(3)
    receiver[0], receiver[1], receiver[2] = state[:3]
    position = prl.Arr1Ddouble(3)
    prl.ecef2pos(receiver, position)

    azel_rows: list[list[float]] = []
    ionosphere_delay: list[float] = []
    troposphere_delay: list[float] = []
    ionosphere_variance: list[float] = []
    troposphere_variance: list[float] = []
    compact_rows: list[int] = []

    for index in range(count):
        if index in exclude:
            continue
        # This vector is deliberately never populated. It is the archived Torch
        # behavior that makes azimuth/elevation and both delays zero.
        line_of_sight = prl.Arr1Ddouble(3)
        azel = prl.Arr1Ddouble(2)
        prl.satazel(position, line_of_sight, azel)
        if azel[1] < 0:
            continue

        dion = prl.Arr1Ddouble(1)
        vion = prl.Arr1Ddouble(1)
        dtrp = prl.Arr1Ddouble(1)
        vtrp = prl.Arr1Ddouble(1)
        prl.ionocorr(
            epoch_time,
            nav,
            satellites[index],
            position,
            azel,
            prl.IONOOPT_BRDC,
            dion,
            vion,
        )
        prl.tropcorr(
            epoch_time,
            nav,
            position,
            azel,
            prl.TROPOPT_SAAS,
            dtrp,
            vtrp,
        )
        compact_rows.append(index)
        azel_rows.append([float(azel[0]), float(azel[1])])
        ionosphere_delay.append(float(dion[0]))
        troposphere_delay.append(float(dtrp[0]))
        ionosphere_variance.append(float(vion[0]))
        troposphere_variance.append(float(vtrp[0]))

    return {
        "compact_rows": np.asarray(compact_rows, dtype=np.int64),
        "azel_rad": np.asarray(azel_rows, dtype=np.float64),
        "ionosphere_delay_m": np.asarray(ionosphere_delay, dtype=np.float64),
        "troposphere_delay_m": np.asarray(troposphere_delay, dtype=np.float64),
        "ionosphere_variance_m2": np.asarray(
            ionosphere_variance, dtype=np.float64
        ),
        "troposphere_variance_m2": np.asarray(
            troposphere_variance, dtype=np.float64
        ),
    }


def main() -> int:
    args = parse_args()
    observation_path = args.observation.resolve()
    archive_path = args.dataset_archive.resolve()
    runtime, _runtime_manifest = resolve_runtime(args.runtime_dir)
    tdl_dir = runtime.tdl_dir
    output_dir = args.output_dir.resolve()

    archive_record = verify_file(
        archive_path, DATASET_ARCHIVE_SHA256, "dataset archive"
    )
    observation_record = verify_file(
        observation_path, OBSERVATION_SHA256, "rover observation"
    )
    ephemeris_paths = [Path(path).resolve() for path in sorted(glob.glob(args.ephemeris_glob))]
    if {path.name for path in ephemeris_paths} != set(NAVIGATION_SHA256):
        raise RuntimeError(
            "ephemeris wildcard resolved to the wrong files: "
            + ", ".join(path.name for path in ephemeris_paths)
        )
    ephemeris_records = [
        verify_file(path, NAVIGATION_SHA256[path.name], f"navigation file {path.name}")
        for path in ephemeris_paths
    ]

    if not (tdl_dir / "rtk_util.py").is_file():
        raise FileNotFoundError(f"missing archived rtk_util.py under {tdl_dir}")
    source_text = (tdl_dir / "rtk_util.py").read_text()
    if ".to('cuda')" in source_text:
        raise RuntimeError(
            "cached TDL runtime still targets CUDA; rebuild it with "
            "validation.ibiza_generalization.prepare_runtime"
        )

    prl = import_pyrtklib(runtime.pyrtklib_site)
    import torch
    util = load_rtk_util(tdl_dir, module_name="real_klt_rtk_util")
    if ".to('cpu')" not in inspect.getsource(util.H_matrix_prl_torch):
        raise RuntimeError("archived Torch matrix function lacks the CPU substitution")

    obs, nav, _station = util.read_obs(
        str(observation_path), args.ephemeris_glob
    )
    prl.sortobs(obs)
    epochs = util.split_obs(obs)

    selected = None
    for epoch_index, epoch in enumerate(epochs):
        epoch_time = float(epoch.data[0].time.time + epoch.data[0].time.sec)
        if not (epoch_time > args.start_time and epoch_time < args.end_time):
            continue
        gps_epoch, source_rows = gps_subset(prl, epoch)
        if gps_epoch.n < 4:
            continue
        try:
            ols_result = util.get_ls_pnt_pos(gps_epoch, nav)
        except Exception:
            continue
        if ols_result["status"]:
            selected = (
                epoch_index,
                epoch_time,
                gps_epoch,
                source_rows,
                ols_result,
            )
            break

    if selected is None:
        raise RuntimeError("no valid GPS-only epoch found in the requested interval")
    epoch_index, epoch_time, epoch, source_rows, ols_result = selected

    if epoch_index != EXPECTED_EPOCH_INDEX:
        raise RuntimeError(
            f"wrong selected epoch index: expected {EXPECTED_EPOCH_INDEX}, got {epoch_index}"
        )
    if epoch_time != EXPECTED_EPOCH:
        raise RuntimeError(
            f"wrong selected epoch: expected {EXPECTED_EPOCH}, got {epoch_time}"
        )

    satellite_positions_raw, no_ephemeris, clock_raw, _variance = util.get_sat_pos(
        epoch.data, epoch.n, nav
    )
    if no_ephemeris:
        raise RuntimeError(f"selected GPS epoch lacks ephemeris rows: {no_ephemeris}")

    corrected_pseudorange: list[float] = []
    raw_pseudorange: list[float] = []
    satellites: list[int] = []
    satellite_ids: list[str] = []
    preprocessing_exclude: list[int] = []
    vmeas = prl.Arr1Ddouble(1)
    system_name = prl.Arr1Dchar(4)
    for index in range(epoch.n):
        observation = epoch.data[index]
        raw = float(observation.P[0])
        prl.satno2id(observation.sat, system_name)
        sat_id = str(system_name[0])
        if raw == 0.0 or sat_id[0] not in ["G", "C", "R", "E"]:
            preprocessing_exclude.append(index)
            corrected = 0.0
        else:
            corrected = float(
                util.prange(observation, nav, prl.prcopt_default, vmeas)
            )
        raw_pseudorange.append(raw)
        corrected_pseudorange.append(corrected)
        satellites.append(int(observation.sat))
        satellite_ids.append(sat_id)

    if tuple(satellite_ids) != EXPECTED_SATELLITES:
        raise RuntimeError(
            f"wrong satellite set/order: expected {EXPECTED_SATELLITES}, "
            f"got {tuple(satellite_ids)}"
        )
    if preprocessing_exclude:
        raise RuntimeError(
            f"unexpected preprocessing exclusions: {preprocessing_exclude}"
        )

    satellite_positions_6 = as_numpy(
        satellite_positions_raw, epoch.n * 6
    ).reshape(epoch.n, 6)
    satellite_positions = satellite_positions_6[:, :3]
    satellite_clock_bias = as_numpy(clock_raw, epoch.n * 2)[::2]
    satellite_clock_correction = -prl.CLIGHT * satellite_clock_bias
    corrected = np.asarray(corrected_pseudorange, dtype=np.float64).reshape(-1, 1)
    raw = np.asarray(raw_pseudorange, dtype=np.float64)

    ols_initial = np.asarray(ols_result["pos"], dtype=np.float64)
    if not np.allclose(ols_initial, EXPECTED_OLS_STATE, rtol=0.0, atol=1e-7):
        raise RuntimeError(
            "GPS-only OLS state differs from the validated result: "
            f"{ols_initial.tolist()}"
        )

    weight_by_satellite = dict(zip(EXPECTED_SATELLITES, DETERMINISTIC_WEIGHTS))
    state = torch.tensor(ols_initial, dtype=torch.float64, device="cpu")
    delta = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
    trace: dict[str, np.ndarray] = {}
    iteration = 0
    all_rows_aligned = True
    reconstruction_max_abs = 0.0
    last_raw_residual = np.empty((0, 1), dtype=np.float64)
    active_indices_per_iteration: list[list[int]] = []

    while torch.norm(delta) > 0.0001 and iteration < 10:
        state_before = state.clone()
        (
            h,
            predicted,
            azel,
            excluded,
            active_indices,
            _system_count,
            _vels,
            returned_vion,
            returned_vtrp,
        ) = util.H_matrix_prl_torch(
            satellite_positions_raw,
            state,
            clock_raw,
            epoch.data[0].time,
            nav,
            satellites,
            preprocessing_exclude,
        )

        # These two constructions intentionally match the archived code. The
        # list(set(...)) order is hazardous, so equality is checked explicitly.
        residual_rows = list(set(range(epoch.n - len(no_ephemeris))) - set(excluded))
        h_rows = [
            index
            for index in range(epoch.n - len(no_ephemeris))
            if index not in excluded
        ]
        rows_aligned = h_rows == residual_rows
        all_rows_aligned = all_rows_aligned and rows_aligned
        if not rows_aligned:
            raise RuntimeError(
                "legacy list(set(...)) row order differs from Jacobian row order: "
                f"H={h_rows}, residual={residual_rows}"
            )
        if len(residual_rows) < 4:
            raise RuntimeError("selected iteration has fewer than four observations")

        iteration_ids = [satellite_ids[index] for index in h_rows]
        weight_vector = np.asarray(
            [weight_by_satellite[name] for name in iteration_ids],
            dtype=np.float64,
        )
        weight = torch.diag(torch.tensor(weight_vector, dtype=torch.float64))
        raw_residual = (
            torch.tensor(corrected[residual_rows], dtype=torch.float64)
            - predicted
        )
        bias = torch.zeros_like(raw_residual)
        effective_residual = raw_residual - bias

        # These exported reference values use the same Torch operations as the
        # paper-era solve. The independent script recomputes them with NumPy.
        normal = torch.matmul(torch.matmul(h.T, weight), h)
        rhs = torch.matmul(torch.matmul(h.T, weight), effective_residual)
        delta = util.wls_solve_torch(h, raw_residual, weight, bias)
        state_after = state.clone()
        state_after[active_indices] = (
            state_after[active_indices] + delta.squeeze()
        )

        legacy = legacy_atmosphere_components(
            prl,
            satellite_positions_raw,
            state_before.detach().cpu().numpy(),
            epoch.data[0].time,
            nav,
            satellites,
            preprocessing_exclude,
        )
        if h_rows != legacy["compact_rows"].tolist():
            raise RuntimeError("legacy atmosphere instrumentation row mismatch")

        compact_positions = satellite_positions[h_rows]
        receiver_xyz = state_before[:3].detach().cpu().numpy()
        geometric_range = np.linalg.norm(
            compact_positions - receiver_xyz, axis=1
        )
        sagnac = (
            prl.OMGE
            * (
                compact_positions[:, 0] * receiver_xyz[1]
                - compact_positions[:, 1] * receiver_xyz[0]
            )
            / prl.CLIGHT
        )
        predicted_reconstructed = (
            geometric_range
            + sagnac
            + float(state_before[3])
            + satellite_clock_correction[h_rows]
            + legacy["ionosphere_delay_m"]
            + legacy["troposphere_delay_m"]
        ).reshape(-1, 1)
        reconstruction_diff = float(
            np.max(
                np.abs(
                    predicted_reconstructed
                    - predicted.detach().cpu().numpy()
                )
            )
        )
        reconstruction_max_abs = max(
            reconstruction_max_abs, reconstruction_diff
        )
        if reconstruction_diff != 0.0:
            raise RuntimeError(
                "predicted-observation component reconstruction is not exact: "
                f"{reconstruction_diff}"
            )

        prefix = f"iteration_{iteration}_"
        trace[prefix + "active_state_indices"] = np.asarray(
            active_indices, dtype=np.int64
        )
        trace[prefix + "state_before"] = (
            state_before.detach().cpu().numpy().copy()
        )
        trace[prefix + "predicted_observation_m"] = (
            predicted.detach().cpu().numpy().copy()
        )
        trace[prefix + "H"] = h.detach().cpu().numpy().copy()
        trace[prefix + "raw_residual_m"] = (
            raw_residual.detach().cpu().numpy().copy()
        )
        trace[prefix + "effective_residual_m"] = (
            effective_residual.detach().cpu().numpy().copy()
        )
        trace[prefix + "W"] = weight.detach().cpu().numpy().copy()
        trace[prefix + "normal_matrix"] = (
            normal.detach().cpu().numpy().copy()
        )
        trace[prefix + "rhs"] = rhs.detach().cpu().numpy().copy()
        trace[prefix + "delta_state"] = delta.detach().cpu().numpy().copy()
        trace[prefix + "state_after"] = (
            state_after.detach().cpu().numpy().copy()
        )
        trace[prefix + "torch_azel_rad"] = as_numpy(
            azel, epoch.n * 2
        ).reshape(epoch.n, 2)[h_rows]
        trace[prefix + "ionosphere_m"] = np.asarray(
            returned_vion, dtype=np.float64
        )
        trace[prefix + "troposphere_m"] = np.asarray(
            returned_vtrp, dtype=np.float64
        )
        trace[prefix + "h_compact_rows"] = np.asarray(
            h_rows, dtype=np.int64
        )
        trace[prefix + "residual_compact_rows"] = np.asarray(
            residual_rows, dtype=np.int64
        )
        trace[prefix + "geometric_range_m"] = geometric_range
        trace[prefix + "sagnac_m"] = sagnac
        trace[prefix + "ionosphere_delay_m"] = legacy[
            "ionosphere_delay_m"
        ]
        trace[prefix + "troposphere_delay_m"] = legacy[
            "troposphere_delay_m"
        ]
        trace[prefix + "ionosphere_variance_m2"] = legacy[
            "ionosphere_variance_m2"
        ]
        trace[prefix + "troposphere_variance_m2"] = legacy[
            "troposphere_variance_m2"
        ]
        trace[prefix + "reconstructed_azel_rad"] = legacy["azel_rad"]

        state = state_after
        last_raw_residual = trace[prefix + "raw_residual_m"]
        active_indices_per_iteration.append(
            np.asarray(active_indices, dtype=np.int64).tolist()
        )
        iteration += 1

    if iteration >= 10:
        raise RuntimeError("paper-era WLS exceeded its maximum iteration count")
    if float(np.linalg.norm(last_raw_residual)) > 1000.0:
        raise RuntimeError("paper-era WLS final residual norm exceeds 1000 m")

    # The archived wrapper performs this NumPy-model evaluation after convergence.
    util.H_matrix_prl(
        satellite_positions_raw,
        state.detach().cpu().numpy(),
        clock_raw,
        epoch.data[0].time,
        nav,
        satellites,
        preprocessing_exclude,
    )

    final_state = state.detach().cpu().numpy().copy()
    if not np.allclose(final_state, EXPECTED_FINAL_STATE, rtol=0.0, atol=1e-7):
        raise RuntimeError(
            "final WLS state differs from the validated result: "
            f"{final_state.tolist()}"
        )

    trace.update(
        {
            "epoch_time": np.asarray(epoch_time, dtype=np.float64),
            "source_row_indices": np.asarray(source_rows, dtype=np.int64),
            "h_compact_rows": trace["iteration_0_h_compact_rows"],
            "residual_compact_rows": trace[
                "iteration_0_residual_compact_rows"
            ],
            "satellite_ids": np.asarray(satellite_ids, dtype="U3"),
            "satellite_positions_ecef_m": satellite_positions,
            "satellite_clock_bias_s": satellite_clock_bias,
            "satellite_clock_correction_m": satellite_clock_correction,
            "raw_pseudorange_m": raw,
            "corrected_pseudorange_m": corrected,
            "ols_initial_state": ols_initial,
            "weight_diagonal": DETERMINISTIC_WEIGHTS,
            "final_state": final_state,
        }
    )

    legacy_flags = {
        "torch_azel_all_zero": all(
            bool(np.all(trace[f"iteration_{i}_torch_azel_rad"] == 0.0))
            for i in range(iteration)
        ),
        "ionosphere_delay_all_zero": all(
            bool(np.all(trace[f"iteration_{i}_ionosphere_delay_m"] == 0.0))
            for i in range(iteration)
        ),
        "troposphere_delay_all_zero": all(
            bool(np.all(trace[f"iteration_{i}_troposphere_delay_m"] == 0.0))
            for i in range(iteration)
        ),
        "uninitialized_los_used_for_satazel": True,
        "returned_vion_vtrp_are_variances_not_delays": True,
        "predicted_observation_component_reconstruction_max_abs_m": (
            reconstruction_max_abs
        ),
    }
    if not (
        legacy_flags["torch_azel_all_zero"]
        and legacy_flags["ionosphere_delay_all_zero"]
        and legacy_flags["troposphere_delay_all_zero"]
    ):
        raise RuntimeError("the expected archived zero atmosphere behavior changed")

    manifest = {
        "status": True,
        "reference_commit": TDL_COMMIT,
        "pyrtklib_hypothesis_commit": PYRTKLIB_COMMIT,
        "pyrtklib_version": PYRTKLIB_VERSION,
        "pyrtklib_publication_version_uncertain": True,
        "dataset_archive_url": DATASET_URL,
        "dataset_archive_sha256": archive_record["sha256"],
        "dataset_archive_size_bytes": archive_record["size_bytes"],
        "rover_observation": {
            "archive_path": "data/0610_KLT/COM38_210610_025603.obs",
            "sha256": observation_record["sha256"],
            "size_bytes": observation_record["size_bytes"],
        },
        "wildcard_inputs": ephemeris_records,
        "epoch_index_after_split": epoch_index,
        "epoch_time_gpst_unix_like": epoch_time,
        "epoch_time_iso_gpst_like": datetime.fromtimestamp(
            epoch_time, tz=timezone.utc
        ).isoformat(),
        "selection_note": (
            "First valid GPS-only epoch satisfying paper config "
            "t > 1623296154; fractional .005 makes 03:35:54.005 eligible."
        ),
        "satellite_ids": satellite_ids,
        "source_row_indices": source_rows,
        "h_compact_rows": trace["h_compact_rows"].tolist(),
        "residual_compact_rows_from_list_set": trace[
            "residual_compact_rows"
        ].tolist(),
        "row_alignment_equal": all_rows_aligned,
        "weight_by_satellite": {
            name: float(value)
            for name, value in zip(satellite_ids, DETERMINISTIC_WEIGHTS)
        },
        "ols_initial_state": ols_initial.tolist(),
        "active_state_indices_per_iteration": active_indices_per_iteration,
        "iterations": iteration,
        "last_raw_residual_norm_m": float(np.linalg.norm(last_raw_residual)),
        "final_state": final_state.tolist(),
        "legacy_torch_behavior_preserved": legacy_flags,
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pyrtklib": installed_version,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "paper_epoch_trace.npz"
    manifest_path = output_dir / "paper_epoch_manifest.json"
    np.savez(trace_path, **trace)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print(f"wrote {trace_path}")
    print(f"wrote {manifest_path}")
    print(
        "selected epoch: "
        f"{manifest['epoch_time_iso_gpst_like']} "
        f"({epoch_time:.3f}, split index {epoch_index})"
    )
    print("satellites: " + " ".join(satellite_ids))
    print(f"iterations: {iteration}")
    print("final state: " + np.array2string(final_state, precision=12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
