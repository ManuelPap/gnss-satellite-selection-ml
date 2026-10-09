#!/usr/bin/env python3
"""Run the separately labelled current-stack neutral TASGNSS Ibiza baseline.

Phase A is intentionally not implemented here: the frozen Ibiza artifact fails
the strict same-input compatibility gate documented in ``solver_audit.json``.
This runner uses original RINEX, current TASGNSS preprocessing, zero bias, and
uniform weights.  It serializes all positions before opening the EPN reference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

from .core import (
    IBIZA_NPZ_SHA256,
    accuracy_summary,
    distribution_summary,
    enu_errors,
    error_components,
    historical_neutral_rows,
    json_ready,
    paired_delta_summary,
    phase_a_compatibility,
    sha256_file,
    trusted_reference_ecef,
    write_csv,
    write_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = REPOSITORY_ROOT.parent
DEFAULT_TASGNSS = PHD_ROOT / "external_references/TASGNSS"
DEFAULT_TDL = PHD_ROOT / "external_references/TDL-GNSS"
DEFAULT_PYRTKLIB = PHD_ROOT / "external_references/pyrtklib"
DEFAULT_RTKLIB = PHD_ROOT / "external_references/RTKLIB"
DEFAULT_BACKEND_SITE = PHD_ROOT / "external_data/tasgnss_comparison/runtime/pyrtklib-0.2.7-site"
DEFAULT_DATASET = PHD_ROOT / "external_data/ibiza_2025_01_01"
DEFAULT_NPZ = DEFAULT_DATASET / "derived/ibiza_preprocessed.npz"
DEFAULT_OBSERVATION = DEFAULT_DATASET / "raw/observation/IBIZ00ESP_R_20250010000_01D_30S_MO.rnx"
DEFAULT_NAVIGATION = DEFAULT_DATASET / "raw/navigation/BRDM00DLR_S_20250010000_01D_MN.rnx"
DEFAULT_REFERENCE = REPOSITORY_ROOT / "validation/ibiza_generalization/ibiz00esp_reference.json"
DEFAULT_HISTORICAL_NEUTRAL = PHD_ROOT / "external_data/domain_shift/positioning_ablation/positioning_ablation_epoch_results.npz"
DEFAULT_OUTPUT = PHD_ROOT / "external_data/tasgnss_comparison/ibiza_neutral"

SYSTEM_CODE = {"G": 1, "R": 2, "E": 3, "C": 4, "J": 5, "I": 6, "1": 7}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasgnss-repo", type=Path, default=DEFAULT_TASGNSS)
    parser.add_argument("--backend-site", type=Path, default=DEFAULT_BACKEND_SITE)
    parser.add_argument("--ibiza-npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--navigation", type=Path, default=DEFAULT_NAVIGATION)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--historical-neutral", type=Path, default=DEFAULT_HISTORICAL_NEUTRAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args(argv)


def _run_text(arguments: list[str]) -> str:
    completed = subprocess.run(arguments, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def repository_provenance(path: Path) -> dict[str, object]:
    remotes: dict[str, dict[str, str]] = {}
    for line in _run_text(["git", "-C", str(path), "remote", "-v"]).splitlines():
        fields = line.split()
        if len(fields) >= 3:
            remotes.setdefault(fields[0], {})[fields[2].strip("()")] = fields[1]
    status = _run_text(["git", "-C", str(path), "status", "--short"])
    return {
        "path": str(path.resolve()),
        "head": _run_text(["git", "-C", str(path), "rev-parse", "HEAD"]),
        "status_short": status.splitlines() if status else [],
        "remotes": remotes,
    }


def _setup_version(path: Path) -> str:
    match = re.search(r"\bversion\s*=\s*[\"']([^\"']+)", path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"version is absent from {path}")
    return match.group(1)


def _import_external_stack(tasgnss_repo: Path, backend_site: Path) -> tuple[Any, Any]:
    os.environ["rtklib"] = "origin"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(backend_site.resolve()))
    sys.path.insert(0, str(tasgnss_repo.resolve()))
    import tasgnss  # type: ignore[import-not-found]
    from tasgnss import core as tas_core  # type: ignore[import-not-found]

    backend_file = Path(tas_core.prl.__file__).resolve()
    if not backend_file.is_relative_to(backend_site.resolve()):
        raise RuntimeError(f"TASGNSS imported an unexpected backend: {backend_file}")
    if tas_core.rtklib_version != "origin" or tas_core.prl.__name__ != "pyrtklib":
        raise RuntimeError("the baseline requires TASGNSS's origin pyrtklib backend")
    return tasgnss, tas_core


def _time_of_epoch(epoch: Any) -> float:
    value = epoch.data[0].time
    return float(value.time + value.sec)


def _active_design_diagnostics(h: np.ndarray) -> tuple[int, int, float]:
    active = np.any(h != 0.0, axis=0)
    active_h = h[:, active]
    rank = int(np.linalg.matrix_rank(active_h))
    condition = float(np.linalg.cond(active_h))
    return int(np.sum(active)), rank, condition


def _solve_all_epochs(
    tas: Any,
    observation_path: Path,
    navigation_path: Path,
    *,
    progress_every: int,
) -> tuple[dict[str, np.ndarray], list[np.ndarray], Counter[str], dict[str, int]]:
    observations, navigation, _station = tas.read_obs(
        str(observation_path), str(navigation_path), opt=""
    )
    epochs = tas.split_obs(observations, ref_obs=False)
    count = len(epochs)
    split_index = np.arange(count, dtype=np.int64)
    timestamps = np.empty(count, dtype=np.float64)
    raw_counts = np.empty(count, dtype=np.int64)
    processed_counts = np.zeros(count, dtype=np.int64)
    solved = np.zeros(count, dtype=np.uint8)
    positions = np.full((count, 3), np.nan, dtype=np.float64)
    clocks = np.full((count, 7), np.nan, dtype=np.float64)
    active_state_count = np.full(count, -1, dtype=np.int64)
    design_rank = np.full(count, -1, dtype=np.int64)
    design_condition = np.full(count, np.nan, dtype=np.float64)
    residual_norm = np.full(count, np.nan, dtype=np.float64)
    iteration_count = np.full(count, -1, dtype=np.int64)
    support: list[np.ndarray] = []
    support_system: list[np.ndarray] = []
    messages: Counter[str] = Counter()
    controls = {"uniform_weights_epochs": 0, "zero_bias_epochs": 0}

    for index, epoch in enumerate(epochs):
        timestamps[index] = _time_of_epoch(epoch)
        raw_counts[index] = int(epoch.n)
        result = tas.wls_pnt_pos(
            epoch,
            navigation,
            use_cache=True,
            return_residual=True,
            enable_torch=False,
            w=1,
            b=None,
            device="cpu",
        )
        message = str(result.get("msg", "missing status message"))
        messages[message] += 1
        solve_data = result.get("solve_data", {})
        epoch_support = np.asarray(
            [int(row[0]) for row in result.get("data", [])], dtype=np.int64
        )
        epoch_system = np.asarray(
            [SYSTEM_CODE[str(row[2])] for row in result.get("data", [])], dtype=np.uint8
        )
        support.append(epoch_support)
        support_system.append(epoch_system)
        processed_counts[index] = epoch_support.size
        if result.get("status", False):
            solved[index] = 1
            positions[index] = np.asarray(result["pos"], dtype=np.float64).reshape(3)
            clocks[index] = np.asarray(result["cb"], dtype=np.float64).reshape(7)
            residual_info = result["residual_info"]
            residual = np.asarray(residual_info["residual"], dtype=np.float64)
            h = np.asarray(residual_info["H"], dtype=np.float64)
            weight = np.asarray(residual_info["W"], dtype=np.float64)
            active_state_count[index], design_rank[index], design_condition[index] = (
                _active_design_diagnostics(h)
            )
            residual_norm[index] = float(np.linalg.norm(residual))
            if np.array_equal(weight, np.eye(weight.shape[0], dtype=np.float64)):
                controls["uniform_weights_epochs"] += 1
            controls["zero_bias_epochs"] += 1
        tas.cache_data.pop(id(epoch), None)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"solved {index + 1}/{count} raw epochs", flush=True)

    offsets = np.zeros(count + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([values.size for values in support])
    flat_support = np.concatenate(support) if support else np.empty(0, dtype=np.int64)
    flat_system = np.concatenate(support_system) if support_system else np.empty(0, dtype=np.uint8)
    arrays = {
        "schema_version": np.array([1], dtype=np.int64),
        "split_epoch_index": split_index,
        "epoch_time_gpst_like_s": timestamps,
        "raw_observation_count": raw_counts,
        "processed_observation_count": processed_counts,
        "solved": solved,
        "position_ecef_m": positions,
        "clock_bias_m_g_c_e_r_j_i_sbas": clocks,
        "active_state_count": active_state_count,
        "final_design_rank": design_rank,
        "final_design_condition_number": design_condition,
        "final_residual_norm_m": residual_norm,
        "iteration_count": iteration_count,
        "support_offsets": offsets,
        "support_rtklib_satellite_number": flat_support,
        "support_constellation_code": flat_system,
    }
    return arrays, support, messages, controls


def _state_from_result(result: dict[str, object]) -> np.ndarray:
    def convert(value: object) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()  # type: ignore[union-attr]
        return np.asarray(value, dtype=np.float64)

    return np.concatenate(
        (
            convert(result["pos"]).reshape(3),
            convert(result["cb"]).reshape(7),
        )
    )


def _uniform_scaling_check(
    tas: Any, observation_path: Path, navigation_path: Path
) -> dict[str, object]:
    observations, navigation, _station = tas.read_obs(
        str(observation_path), str(navigation_path), opt=""
    )
    epoch = tas.split_obs(observations, ref_obs=False)[0]
    baseline = tas.wls_pnt_pos(epoch, navigation, w=1, b=None, enable_torch=False)
    if not baseline["status"]:
        raise RuntimeError("uniform-scaling audit epoch did not solve")
    baseline_state = _state_from_result(baseline)
    observation_count = int(len(baseline["solve_data"]["pr"]))
    comparisons: list[dict[str, object]] = []
    for scale in (0.25, 7.0):
        tas.cache_data[id(epoch)][0] = None
        tas.cache_data[id(epoch)][1] = None
        scaled = tas.wls_pnt_pos(
            epoch,
            navigation,
            w=np.full(observation_count, scale, dtype=np.float64),
            b=None,
            enable_torch=False,
        )
        if not scaled["status"]:
            raise RuntimeError(f"uniform-scaling solve failed at scale {scale}")
        difference = float(np.max(np.abs(_state_from_result(scaled) - baseline_state)))
        comparisons.append(
            {"scale": scale, "maximum_absolute_state_difference": difference, "passed": difference <= 1e-7}
        )
    tas.cache_data.pop(id(epoch), None)
    return {
        "epoch_split_index": 0,
        "uniform_value_selected": 1.0,
        "effective_objective": "minimize ||diag(w) @ (H dx - residual)||_2^2",
        "comparisons": comparisons,
        "passed": all(bool(row["passed"]) for row in comparisons),
    }


def _numpy_torch_consistency_check(
    tas: Any, observation_path: Path, navigation_path: Path
) -> dict[str, object]:
    observations, navigation, _station = tas.read_obs(
        str(observation_path), str(navigation_path), opt=""
    )
    epoch = tas.split_obs(observations, ref_obs=False)[0]
    numpy_result = tas.wls_pnt_pos(
        epoch, navigation, w=1, b=None, enable_torch=False, device="cpu"
    )
    if not numpy_result["status"]:
        raise RuntimeError("NumPy consistency-check solve failed")
    tas.cache_data[id(epoch)][0] = None
    tas.cache_data[id(epoch)][1] = None
    torch_result = tas.wls_pnt_pos(
        epoch, navigation, w=1, b=None, enable_torch=True, device="cpu"
    )
    if not torch_result["status"]:
        raise RuntimeError("Torch consistency-check solve failed")
    maximum_difference = float(
        np.max(np.abs(_state_from_result(numpy_result) - _state_from_result(torch_result)))
    )
    tas.cache_data.pop(id(epoch), None)
    return {
        "epoch_split_index": 0,
        "maximum_absolute_state_difference": maximum_difference,
        "tolerance": 1.0e-7,
        "passed": maximum_difference <= 1.0e-7,
        "scope": "neutral CPU solve only; no model, gradients, optimizer, or backward pass",
    }


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _accuracy_row(name: str, enu: np.ndarray, total: int, experiment: str) -> dict[str, object]:
    return {"solution": name, "experiment": experiment, **accuracy_summary(enu, total_epochs=total)}


def _support_comparison(
    frozen: Any, current_support: list[np.ndarray], current_solved: np.ndarray
) -> dict[str, object]:
    offsets = frozen["epoch_offsets"]
    exact = 0
    common_satellite_fractions: list[float] = []
    for accepted_index, split_index_value in enumerate(frozen["epoch_split_index"]):
        split_index = int(split_index_value)
        if not current_solved[split_index]:
            continue
        expected = frozen["rtklib_satellite_number"][
            int(offsets[accepted_index]) : int(offsets[accepted_index + 1])
        ]
        actual = current_support[split_index]
        if np.array_equal(expected, actual):
            exact += 1
        union = np.union1d(expected, actual)
        intersection = np.intersect1d(expected, actual)
        common_satellite_fractions.append(float(intersection.size / union.size))
    return {
        "accepted_epochs": int(frozen["epoch_split_index"].size),
        "exact_same_order_support_epochs": exact,
        "all_exact": exact == int(frozen["epoch_split_index"].size),
        "mean_jaccard_satellite_support": float(np.mean(common_satellite_fractions)),
        "minimum_jaccard_satellite_support": float(np.min(common_satellite_fractions)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    input_hashes_before = {
        "ibiza_npz": sha256_file(args.ibiza_npz),
        "observation_rinex": sha256_file(args.observation),
        "navigation_rinex": sha256_file(args.navigation),
        "historical_neutral": sha256_file(args.historical_neutral),
    }
    if input_hashes_before["ibiza_npz"] != IBIZA_NPZ_SHA256:
        raise RuntimeError("frozen Ibiza NPZ SHA-256 mismatch")
    repositories = {
        "comparison": REPOSITORY_ROOT,
        "TASGNSS": args.tasgnss_repo,
        "TDL-GNSS": DEFAULT_TDL,
        "pyrtklib": DEFAULT_PYRTKLIB,
        "RTKLIB": DEFAULT_RTKLIB,
    }
    provenance_before = {
        name: repository_provenance(path) for name, path in repositories.items()
    }
    tas, tas_core = _import_external_stack(args.tasgnss_repo, args.backend_site)
    backend_binary = next((args.backend_site / "pyrtklib").glob("pyrtklib*.so"))
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch_distribution": importlib.metadata.version("torch"),
        "tasgnss_version": _setup_version(args.tasgnss_repo / "setup.py"),
        "tasgnss_commit": provenance_before["TASGNSS"]["head"],
        "backend_selector": tas_core.rtklib_version,
        "backend_module": tas_core.prl.__name__,
        "backend_distribution_version": importlib.metadata.version("pyrtklib"),
        "backend_binary": str(backend_binary.resolve()),
        "backend_binary_sha256": sha256_file(backend_binary),
        "backend_source_commit": provenance_before["pyrtklib"]["head"],
    }

    with np.load(args.ibiza_npz, allow_pickle=False) as frozen_for_gate:
        compatibility = phase_a_compatibility(frozen_for_gate.files)
    if compatibility["passed"]:
        raise RuntimeError("the recorded Phase-A gate unexpectedly changed")

    started = time.monotonic()
    arrays, current_support, messages, solve_controls = _solve_all_epochs(
        tas,
        args.observation,
        args.navigation,
        progress_every=args.progress_every,
    )
    solve_duration_s = time.monotonic() - started
    positions_path = output / "current_stack_positions_pre_ground_truth.npz"
    _save_npz(positions_path, arrays)
    positions_hash_before_ground_truth = sha256_file(positions_path)
    scaling = _uniform_scaling_check(tas, args.observation, args.navigation)
    if not scaling["passed"]:
        raise RuntimeError("constant positive uniform weight scaling changed the solution")
    numerical_consistency = _numpy_torch_consistency_check(
        tas, args.observation, args.navigation
    )
    if not numerical_consistency["passed"]:
        raise RuntimeError("NumPy and Torch neutral solves are inconsistent")

    # Scientific boundary: no EPN reference file is opened above this line.
    reference_document = json.loads(args.reference.read_text(encoding="utf-8"))
    reference_ecef = trusted_reference_ecef(reference_document)
    solved_mask = arrays["solved"].astype(bool)
    current_enu = enu_errors(arrays["position_ecef_m"][solved_mask], reference_ecef)
    current_enu_all = np.full((arrays["solved"].size, 3), np.nan, dtype=np.float64)
    current_enu_all[solved_mask] = current_enu

    with np.load(args.ibiza_npz, allow_pickle=False) as frozen:
        frozen_split = frozen["epoch_split_index"].astype(np.int64)
        ols_positions = frozen["epoch_ols_initial_state"][:, :3].astype(np.float64)
        ols_enu = enu_errors(ols_positions, reference_ecef)
        support_comparison = _support_comparison(frozen, current_support, solved_mask)
        frozen_timestamps = frozen["epoch_time_gpst_like_s"].astype(np.float64)
    historical = historical_neutral_rows(args.historical_neutral)
    if not np.array_equal(historical["split_epoch_index"], frozen_split):
        raise RuntimeError("historical neutral epochs do not match the frozen Ibiza epochs")
    if not np.array_equal(historical["timestamp"], frozen_timestamps):
        raise RuntimeError("historical neutral timestamps do not match the frozen Ibiza epochs")
    historical_solved = historical["solved"]
    historical_enu = enu_errors(historical["state"][historical_solved, :3], reference_ecef)
    common = solved_mask[frozen_split] & historical_solved
    current_common_enu = current_enu_all[frozen_split[common]]
    paper_rejected_mask = np.ones(arrays["solved"].size, dtype=bool)
    paper_rejected_mask[frozen_split] = False
    current_paper_rejected_enu = current_enu_all[paper_rejected_mask & solved_mask]

    accuracy_rows = [
        _accuracy_row("paper_ols_initializer", ols_enu, ols_enu.shape[0], "historical_frozen"),
        _accuracy_row(
            "historical_neutral_torch_wls",
            historical_enu,
            historical_solved.size,
            "historical_frozen",
        ),
        _accuracy_row(
            "current_tasgnss_neutral_common_frozen_epochs",
            current_common_enu,
            frozen_split.size,
            "CURRENT-STACK END-TO-END BASELINE; common paper-accepted epochs",
        ),
        _accuracy_row(
            "current_tasgnss_neutral_all_raw_epochs",
            current_enu,
            arrays["solved"].size,
            "CURRENT-STACK END-TO-END BASELINE",
        ),
    ]
    accuracy_path = output / "current_stack_accuracy.csv"
    write_csv(accuracy_path, accuracy_rows)

    paired_rows: list[dict[str, object]] = []
    for reference_name, reference_values in (
        ("paper_ols_initializer", ols_enu[common]),
        ("historical_neutral_torch_wls", historical_enu[common]),
    ):
        for row in paired_delta_summary(current_common_enu, reference_values):
            paired_rows.append(
                {
                    "candidate": "current_tasgnss_neutral_common_frozen_epochs",
                    "reference": reference_name,
                    "delta_definition": "candidate error - reference error; negative is better",
                    **row,
                }
            )
    paired_path = output / "current_stack_paired_summary.csv"
    write_csv(paired_path, paired_rows)

    displacement = np.linalg.norm(
        arrays["position_ecef_m"][frozen_split[common]] - ols_positions[common], axis=1
    )
    historical_displacement = np.linalg.norm(
        historical["state"][common, :3] - ols_positions[common], axis=1
    )
    displacement_summary = {
        "scope_warning": (
            "The current value is an end-to-end displacement with different preprocessing/support, "
            "not a strict solver-only displacement."
        ),
        "current_stack_minus_paper_ols": distribution_summary(displacement),
        "historical_neutral_torch_minus_paper_ols": distribution_summary(historical_displacement),
    }

    epoch_results = dict(arrays)
    epoch_results["enu_error_m"] = current_enu_all
    epoch_results_path = output / "current_stack_epoch_results.npz"
    _save_npz(epoch_results_path, epoch_results)

    solver_audit = {
        "classification": "TASGNSS is a later refactor, not the original Hu paper-era implementation.",
        "phase_a_compatibility_gate": compatibility,
        "current_solver": {
            "entry_point": "tasgnss.core.wls_pnt_pos",
            "observation_model": "tasgnss.core.pseudorange_observe_func",
            "state": "[x,y,z] plus seven metre-valued clocks ordered G,C,E,R,J,I,SBAS; inactive columns remain in H",
            "satellite_clock": "predicted pseudorange includes -CLIGHT*sdt",
            "sagnac": "computed once in preprocess_obs at RTKLIB pntpos position and then frozen",
            "ionosphere": "broadcast IONOOPT_BRDC, computed once at RTKLIB pntpos position and then frozen",
            "troposphere": "Saastamoinen TROPOPT_SAAS, computed once at RTKLIB pntpos position and then frozen",
            "code_bias": "prange applies C1/P1 DCB and constellation TGD/BGD before solve",
            "elevation_azimuth": "get_atmosphere_error calls geodist to populate LOS before satazel; feature az/el uses ECEF-to-ENU",
            "exclusions": "fewer than four raw rows; absent ephemeris; zero prange result; insufficient rows for active systems",
            "maximum_iterations": 20,
            "convergence": "norm(dp) <= 0.001 m; reaching maxiter is reported as failure",
            "rank_conditioning": "no explicit rank or condition rejection; this runner reports active-column diagnostics",
            "numpy_linear_algebra": "numpy.linalg.lstsq(W@H, W@residual, rcond=None)",
            "torch_linear_algebra": "torch.linalg.pinv(W@H) @ (W@residual)",
            "weight_semantics": "custom vector w becomes W=diag(w), hence effective precision is w^2",
            "uniform_weights": "w=1 selects identity and is ordinary least squares",
            "initial_state": "the public current solver starts position and all receiver clocks at zero",
            "atmosphere_recomputed_each_iteration": False,
            "atmosphere_in_differentiable_graph": False,
            "avoids_historical_unpopulated_los": True,
            "already_corrected_pseudorange_api": (
                "No public solver entry accepts frozen corrected pseudorange directly. Injecting it as raw P "
                "would make prange apply code-bias/TGD corrections again."
            ),
        },
        "runtime": runtime,
        "repository_provenance_before": provenance_before,
        "uniform_weight_scaling": scaling,
        "numpy_torch_consistency": numerical_consistency,
    }
    audit_path = output / "solver_audit.json"
    write_json(audit_path, json_ready(solver_audit))

    input_hashes_after = {
        "ibiza_npz": sha256_file(args.ibiza_npz),
        "observation_rinex": sha256_file(args.observation),
        "navigation_rinex": sha256_file(args.navigation),
        "historical_neutral": sha256_file(args.historical_neutral),
    }
    provenance_after = {
        name: repository_provenance(path) for name, path in repositories.items()
    }
    external_unchanged = all(
        provenance_after[name] == provenance_before[name]
        for name in ("TASGNSS", "TDL-GNSS", "pyrtklib", "RTKLIB")
    )
    manifest = {
        "schema_version": 1,
        "experiment": "CURRENT-STACK END-TO-END BASELINE",
        "strict_solver_only_comparison_completed": False,
        "phase_a": {"status": "stopped_at_compatibility_gate", "gate": compatibility},
        "phase_b": {
            "status": "completed",
            "raw_epochs": int(arrays["solved"].size),
            "solved_epochs": int(np.sum(solved_mask)),
            "failed_epochs": int(np.sum(~solved_mask)),
            "solve_duration_seconds": solve_duration_s,
            "messages": dict(messages),
            "support_comparison_with_frozen_paper_epochs": support_comparison,
            "paper_rejected_epoch_slice": {
                "epochs": int(np.sum(paper_rejected_mask)),
                "current_solved_epochs": int(np.sum(paper_rejected_mask & solved_mask)),
                "accuracy": accuracy_summary(
                    current_paper_rejected_enu,
                    total_epochs=int(np.sum(paper_rejected_mask)),
                ),
                "warning": (
                    "These epochs were rejected by the paper OLS preprocessing but accepted by current "
                    "TASGNSS; they are retained in the all-raw Phase-B result."
                ),
            },
        },
        "configuration": {
            "constellation_read_option": "empty RTKLIB readrnx option; observed file contains G/R/E",
            "correction_anchor_pntpos_defaults": (
                "prcopt_default: GPS navsys, 15 degree elevation mask, ionosphere/troposphere off; "
                "used only to obtain the approximate receiver position"
            ),
            "signal": "first RTKLIB pseudorange slot; current prange single-frequency branch",
            "ionosphere": "broadcast IONOOPT_BRDC",
            "troposphere": "Saastamoinen TROPOPT_SAAS",
            "tasgnss_level_elevation_mask": "none; RTKLIB pntpos used only for approximate correction position has its own defaults",
            "weighting": "uniform w=1 identity",
            "state": "ECEF plus G,C,E,R,J,I,SBAS metre-valued receiver clocks",
            "solver_initial_state": "all-zero ECEF and receiver-clock state",
            "backend": runtime,
        },
        "controls": {
            "neural_checkpoint_loaded": False,
            "learned_bias_applied": False,
            "learned_weights_applied": False,
            "uniform_weights": solve_controls["uniform_weights_epochs"] == int(np.sum(solved_mask)),
            "zero_bias": solve_controls["zero_bias_epochs"] == int(np.sum(solved_mask)),
            "torch_backend_enabled_for_phase_b": False,
            "torch_backend_used_for_consistency_check": True,
            "training_run": False,
            "optimizer_created": False,
            "backward_called": False,
            "ground_truth_read_after_positions_serialized": True,
            "ground_truth_passed_to_solver": False,
            "positions_pre_ground_truth_sha256": positions_hash_before_ground_truth,
            "uniform_global_weight_scaling_invariant": scaling["passed"],
            "numpy_torch_neutral_consistency": numerical_consistency["passed"],
            "input_hashes_unchanged": input_hashes_after == input_hashes_before,
            "external_repositories_unchanged": external_unchanged,
        },
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "repository_provenance_before": provenance_before,
        "repository_provenance_after": provenance_after,
        "accuracy": accuracy_rows,
        "paired_summary": paired_rows,
        "initializer_displacement": displacement_summary,
        "interpretation": (
            "Because Phase A failed its compatibility gate, these current-stack results cannot establish "
            "whether solver changes alone removed the historical degradation. The common-epoch and "
            "all-raw metrics use different explicit denominators; the latter retains current TASGNSS "
            "solutions on all paper-rejected epochs."
        ),
    }
    manifest_path = output / "tasgnss_comparison_manifest.json"
    write_json(manifest_path, json_ready(manifest))

    artifact_paths = [
        positions_path,
        epoch_results_path,
        accuracy_path,
        paired_path,
        audit_path,
        manifest_path,
    ]
    write_csv(
        output / "artifact_hashes.csv",
        [
            {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in artifact_paths
        ],
    )
    print(json.dumps({"output": str(output), "solved": int(np.sum(solved_mask)), "failed": int(np.sum(~solved_mask))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
