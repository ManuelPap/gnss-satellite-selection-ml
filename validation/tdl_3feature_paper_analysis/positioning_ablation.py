#!/usr/bin/env python3
"""Controlled Ibiza positioning ablation from immutable frozen outputs.

The positioning stage consumes only the frozen Ibiza preprocessing arrays and
the already-published per-row neural responses.  Ground truth is loaded only
after all positions and solver regressions have completed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np
import torch

from validation.ibiza_generalization.evaluate_ground_truth import (
    DEFAULT_FROZEN_RESULTS_DIR,
    REFERENCE_MANIFEST_PATH,
    assert_frozen_inputs_unchanged,
    compute_error_metrics,
    evaluate_ecef_positions,
    linear_quantile,
    load_reference_manifest,
    result_filename,
    sha256_file,
    validate_frozen_result_set,
)
from validation.ibiza_generalization.inference import _wls_diagnostics
from validation.paper_biasnet.core import solve_paper_bias_position
from validation.paper_hybrid.core import solve_paper_hybrid_position
from validation.paper_weightnet.core import solve_paper_weighted_position


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = REPOSITORY_ROOT.parent
DEFAULT_IBIZA = PHD_ROOT / "external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz"
DEFAULT_MODEL_RESPONSE = PHD_ROOT / "external_data/domain_shift/model_response"
DEFAULT_OUTPUT = PHD_ROOT / "external_data/domain_shift/positioning_ablation"
CHECKPOINT_MANIFEST = REPOSITORY_ROOT / "validation/ibiza_generalization/frozen_checkpoint_manifest.json"

IBIZA_SHA256 = "edb0189e9eadf3266d75984e3041a90306dd44b3ebecc0101eb188f933dc88c5"
MODEL_RESPONSE_MANIFEST_SHA256 = "a9844b905ae6fa616460da72d76c3412f1bb7d0e8fc01ced3d0a4fd0fa0eb0d2"
MODEL_RESPONSE_HASH_INDEX_SHA256 = "0ded6dea2046b83a21078055acd170397d51f30c7fd51f87e370fe8e8171efe9"
FROZEN_RESULT_MANIFEST_SHA256 = "722e43859504925f4917875b361f843907084358f7c7b7976abbc18e42acaecc"
CHECKPOINT_MANIFEST_SHA256 = "e40aa6190e9670b3f5a0ff3a33990827201f98f41302d857eb3ec4f6676f2ba4"
REFERENCE_MANIFEST_SHA256 = "801309b88988ee7407bf2568016b580ba592b7a15bac4789f7f9a720116b94cd"

ARCHITECTURES = ("TDL-B", "TDL-W", "TDL-BW")
SEEDS = tuple(range(10))
EXPECTED_EPOCHS = 2856
EXPECTED_ROWS = 73204
ARCHITECTURE_SLUGS = {"TDL-B": "tdl_b", "TDL-W": "tdl_w", "TDL-BW": "tdl_bw"}
VARIANTS = {
    "TDL-B": ("B-neutral", "B-original"),
    "TDL-W": ("W-neutral", "W-original"),
    "TDL-BW": ("BW-neutral", "BW-bias-only", "BW-weight-only", "BW-full"),
}
ORIGINAL_VARIANT = {"TDL-B": "B-original", "TDL-W": "W-original", "TDL-BW": "BW-full"}
PAIRED_COMPARISONS = {
    "TDL-B": (("B-original", "B-neutral"),),
    "TDL-W": (("W-original", "W-neutral"),),
    "TDL-BW": (
        ("BW-bias-only", "BW-neutral"),
        ("BW-weight-only", "BW-neutral"),
        ("BW-full", "BW-neutral"),
        ("BW-full", "BW-bias-only"),
        ("BW-full", "BW-weight-only"),
    ),
}
STATE_REGRESSION_ATOL = 5.0e-7
CONDITION_REGRESSION_RTOL = 5.0e-8
CONDITION_REGRESSION_ATOL = 1.0e-10
UNIFORM_SCALE_ATOL = 1.0e-8
PAIRED_EQUAL_ATOL_M = 1.0e-12

REQUIRED_DATASET_ARRAYS = (
    "features",
    "row_epoch_time_gpst_like_s",
    "epoch_index",
    "source_row_index",
    "source_observation_index",
    "rtklib_satellite_number",
    "satellite_prn",
    "constellation_code",
    "epoch_offsets",
    "epoch_split_index",
    "epoch_time_gpst_like_s",
    "satellite_position_ecef_m",
    "satellite_clock_bias_s",
    "corrected_pseudorange_m",
    "system_clock_index",
    "epoch_ols_initial_state",
    "epoch_active_state_count",
)


@dataclass
class PositionSeries:
    state: np.ndarray
    solved: np.ndarray
    status: np.ndarray
    rank: np.ndarray
    condition_number: np.ndarray
    iteration_count: np.ndarray


@dataclass
class EvaluatedSeries:
    positioning: PositionSeries
    enu_error_m: np.ndarray
    e2d_m: np.ndarray
    abs_up_m: np.ndarray
    e3d_m: np.ndarray


@dataclass
class AuthoritativeInputs:
    dataset: dict[str, np.ndarray]
    responses: dict[tuple[str, int], dict[str, np.ndarray]]
    provenance: dict[str, object]
    frozen_validation: object
    immutable_snapshot: dict[str, str]


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite scientific artifact: {path}")
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compresslevel=9)


def _snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path.resolve()) for path in sorted(set(paths))}


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    if sha256_file(path) != IBIZA_SHA256:
        raise RuntimeError("Ibiza source NPZ SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_DATASET_ARRAYS if name not in archive.files]
        if missing:
            raise RuntimeError(f"Ibiza source lacks required arrays: {missing}")
        dataset = {name: archive[name] for name in REQUIRED_DATASET_ARRAYS}
    if dataset["features"].shape != (EXPECTED_ROWS, 3):
        raise RuntimeError("Ibiza feature row count/schema changed")
    if dataset["epoch_offsets"].shape != (EXPECTED_EPOCHS + 1,):
        raise RuntimeError("Ibiza epoch offsets changed")
    if int(dataset["epoch_offsets"][-1]) != EXPECTED_ROWS:
        raise RuntimeError("Ibiza epoch offsets do not end at the row count")
    if dataset["epoch_ols_initial_state"].shape != (EXPECTED_EPOCHS, 7):
        raise RuntimeError("Ibiza initial-state schema changed")
    for name, values in dataset.items():
        if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
            raise RuntimeError(f"Ibiza source array {name} contains non-finite values")
    return dataset


def response_filename(architecture: str, seed: int) -> str:
    return f"ibiza_{ARCHITECTURE_SLUGS[architecture]}_seed_{seed:02d}.npz"


def _verify_response_rows(
    dataset: Mapping[str, np.ndarray],
    path: Path,
    architecture: str,
    seed: int,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if str(arrays["dataset"][0]) != "Ibiza":
        raise RuntimeError(f"wrong dataset in {path.name}")
    if str(arrays["architecture"][0]) != architecture or int(arrays["seed"][0]) != seed:
        raise RuntimeError(f"wrong architecture/seed in {path.name}")
    exact_pairs = {
        "features": "features",
        "epoch_index": "epoch_index",
        "source_row_index": "source_row_index",
        "source_observation_index": "source_observation_index",
        "rtklib_satellite_number": "rtklib_satellite_number",
        "satellite_prn": "satellite_prn",
        "constellation_code": "constellation_code",
    }
    for response_name, dataset_name in exact_pairs.items():
        if not np.array_equal(arrays[response_name], dataset[dataset_name]):
            raise RuntimeError(f"{path.name} row mapping differs at {response_name}")
    if not np.array_equal(arrays["row_index"], np.arange(EXPECTED_ROWS, dtype=np.int64)):
        raise RuntimeError(f"{path.name} row_index is not source order")
    outputs: dict[str, np.ndarray] = {}
    if architecture in ("TDL-B", "TDL-BW"):
        outputs["bias_m"] = np.asarray(arrays["predicted_bias_m"], dtype=np.float64)
    if architecture in ("TDL-W", "TDL-BW"):
        outputs["weight"] = np.asarray(arrays["predicted_weight"], dtype=np.float64)
        if not np.all(outputs["weight"] > 0.0):
            raise RuntimeError(f"{path.name} contains a non-positive released weight")
    if any(values.shape != (EXPECTED_ROWS,) for values in outputs.values()):
        raise RuntimeError(f"{path.name} output row count changed")
    if any(not np.all(np.isfinite(values)) for values in outputs.values()):
        raise RuntimeError(f"{path.name} output contains non-finite values")
    return outputs


def validate_authoritative_inputs(
    *,
    dataset_path: Path = DEFAULT_IBIZA,
    model_response_dir: Path = DEFAULT_MODEL_RESPONSE,
    frozen_results_dir: Path = DEFAULT_FROZEN_RESULTS_DIR,
) -> AuthoritativeInputs:
    """Validate immutable sources without loading ground truth or checkpoints."""

    dataset_path = dataset_path.resolve()
    model_response_dir = model_response_dir.resolve()
    frozen_results_dir = frozen_results_dir.resolve()
    response_manifest_path = model_response_dir / "model_response_manifest.json"
    hash_index_path = model_response_dir / "artifact_hashes.csv"
    frozen_manifest_path = frozen_results_dir / "run_manifest.json"
    fixed_hashes = {
        response_manifest_path: MODEL_RESPONSE_MANIFEST_SHA256,
        hash_index_path: MODEL_RESPONSE_HASH_INDEX_SHA256,
        frozen_manifest_path: FROZEN_RESULT_MANIFEST_SHA256,
        CHECKPOINT_MANIFEST.resolve(): CHECKPOINT_MANIFEST_SHA256,
    }
    for path, expected in fixed_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"immutable source hash mismatch for {path}: {actual} != {expected}")

    dataset = _load_dataset(dataset_path)
    response_manifest = _load_json(response_manifest_path)
    if response_manifest["inputs"]["Ibiza"]["sha256_after"] != IBIZA_SHA256:  # type: ignore[index]
        raise RuntimeError("model-response manifest points to a different Ibiza source")
    indexed = {row["path"]: row for row in _read_csv(hash_index_path)}
    responses: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    response_paths: list[Path] = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            relative = f"rows/{response_filename(architecture, seed)}"
            record = indexed.get(relative)
            if record is None:
                raise RuntimeError(f"model-response hash index lacks {relative}")
            path = model_response_dir / relative
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"frozen model-response hash mismatch: {relative}")
            responses[(architecture, seed)] = _verify_response_rows(
                dataset, path, architecture, seed
            )
            response_paths.append(path)

    checkpoint_section = response_manifest["checkpoint_manifest"]
    if checkpoint_section["sha256"] != CHECKPOINT_MANIFEST_SHA256:  # type: ignore[index]
        raise RuntimeError("model-response checkpoint-manifest provenance changed")
    checkpoint_paths: list[Path] = []
    for record in checkpoint_section["records"]:  # type: ignore[index]
        path = REPOSITORY_ROOT / str(record["path"])
        if sha256_file(path) != record["expected_sha256"]:
            raise RuntimeError(f"checkpoint provenance hash mismatch: {record['path']}")
        checkpoint_paths.append(path)

    frozen_validation = validate_frozen_result_set(frozen_results_dir)
    if frozen_validation.run_manifest_sha256 != FROZEN_RESULT_MANIFEST_SHA256:
        raise RuntimeError("frozen inference run manifest changed")
    immutable_paths = (
        [dataset_path, response_manifest_path, hash_index_path, frozen_manifest_path, CHECKPOINT_MANIFEST]
        + response_paths
        + checkpoint_paths
        + [item.path for item in frozen_validation.files]
    )
    immutable_snapshot = _snapshot(immutable_paths)
    provenance = {
        "ibiza_npz": {"path": str(dataset_path), "sha256": IBIZA_SHA256},
        "model_response_manifest": {
            "path": str(response_manifest_path),
            "sha256": MODEL_RESPONSE_MANIFEST_SHA256,
        },
        "model_response_hash_index": {
            "path": str(hash_index_path),
            "sha256": MODEL_RESPONSE_HASH_INDEX_SHA256,
        },
        "verified_ibiza_response_files": len(response_paths),
        "checkpoint_manifest": {
            "path": str(CHECKPOINT_MANIFEST.resolve()),
            "sha256": CHECKPOINT_MANIFEST_SHA256,
            "checkpoint_count": len(checkpoint_paths),
            "checkpoints_deserialized": False,
        },
        "frozen_inference_manifest": {
            "path": str(frozen_manifest_path),
            "sha256": FROZEN_RESULT_MANIFEST_SHA256,
            "result_file_count": len(frozen_validation.files),
        },
    }
    return AuthoritativeInputs(
        dataset=dataset,
        responses=responses,
        provenance=provenance,
        frozen_validation=frozen_validation,
        immutable_snapshot=immutable_snapshot,
    )


def _epoch_slice(dataset: Mapping[str, np.ndarray], epoch_index: int) -> slice:
    start = int(dataset["epoch_offsets"][epoch_index])
    stop = int(dataset["epoch_offsets"][epoch_index + 1])
    rows = slice(start, stop)
    if not np.all(dataset["epoch_index"][rows] == epoch_index):
        raise RuntimeError("epoch offsets and source epoch_index disagree")
    return rows


def _common_solver_arguments(dataset: Mapping[str, np.ndarray], rows: slice) -> tuple[np.ndarray, ...]:
    return (
        dataset["satellite_position_ecef_m"][rows],
        dataset["satellite_clock_bias_s"][rows],
        dataset["corrected_pseudorange_m"][rows],
        dataset["system_clock_index"][rows],
    )


def _solve_epoch(
    dataset: Mapping[str, np.ndarray],
    epoch_index: int,
    *,
    route: str,
    bias_m: np.ndarray | None,
    weight: np.ndarray | None,
) -> tuple[np.ndarray, bool, str, int, float, int]:
    rows = _epoch_slice(dataset, epoch_index)
    common = _common_solver_arguments(dataset, rows)
    initial = dataset["epoch_ols_initial_state"][epoch_index]
    with torch.inference_mode():
        if route == "bias":
            if bias_m is None or weight is not None:
                raise ValueError("bias route requires only bias")
            solution = solve_paper_bias_position(
                *common, bias_m[rows], initial, return_trace=True
            ).wls
        elif route == "weight":
            if weight is None or bias_m is not None:
                raise ValueError("weight route requires only weight")
            solution = solve_paper_weighted_position(
                *common, weight[rows], initial, return_trace=True
            )
        elif route == "hybrid":
            if weight is None or bias_m is None:
                raise ValueError("hybrid route requires weight and bias")
            solution = solve_paper_hybrid_position(
                *common, weight[rows], bias_m[rows], initial, return_trace=True
            ).wls
        else:
            raise ValueError(f"unknown historical solver route {route!r}")
    diagnostics = _wls_diagnostics(solution)
    state = solution.state.detach().cpu().numpy().copy()
    finite = bool(np.all(np.isfinite(state)))
    solved = diagnostics.solution_status == "solved" and finite
    status = diagnostics.solution_status if finite else "non_finite_state"
    return (
        state,
        solved,
        status,
        diagnostics.final_rank,
        diagnostics.final_condition_number,
        len(diagnostics.iterations),
    )


def solve_series(
    dataset: Mapping[str, np.ndarray],
    *,
    route: str,
    bias_m: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    epoch_indices: Sequence[int] | None = None,
) -> PositionSeries:
    indices = tuple(range(EXPECTED_EPOCHS)) if epoch_indices is None else tuple(epoch_indices)
    count = len(indices)
    state = np.full((count, 7), np.nan, dtype=np.float64)
    solved = np.zeros(count, dtype=bool)
    status = np.full(count, "not_run", dtype="U64")
    rank = np.full(count, -1, dtype=np.int64)
    condition = np.full(count, np.nan, dtype=np.float64)
    iterations = np.full(count, -1, dtype=np.int64)
    for output_index, epoch_index in enumerate(indices):
        try:
            values = _solve_epoch(
                dataset,
                epoch_index,
                route=route,
                bias_m=bias_m,
                weight=weight,
            )
            state[output_index], solved[output_index], status[output_index], rank[output_index], condition[output_index], iterations[output_index] = values
        except Exception as error:
            status[output_index] = f"exception:{type(error).__name__}"
    return PositionSeries(
        state=state,
        solved=solved,
        status=status,
        rank=rank,
        condition_number=condition,
        iteration_count=iterations,
    )


def _load_frozen_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def compare_original_to_frozen(
    architecture: str,
    seed: int,
    series: PositionSeries,
    frozen_results_dir: Path,
) -> dict[str, object]:
    path = frozen_results_dir / result_filename(architecture, seed)
    rows = _load_frozen_rows(path)
    if len(rows) != EXPECTED_EPOCHS:
        raise RuntimeError(f"published original result count changed: {path.name}")
    maximum_state_difference = 0.0
    maximum_ecef_difference = 0.0
    maximum_clock_difference = 0.0
    maximum_condition_difference = 0.0
    exact_state_count = 0
    for epoch_index, row in enumerate(rows):
        if int(row["accepted_epoch_index"]) != epoch_index:
            raise RuntimeError(f"published original epoch order changed: {path.name}")
        expected_solved = row["solution_status"] == "solved"
        if bool(series.solved[epoch_index]) != expected_solved:
            raise RuntimeError(f"original solved status mismatch: {path.name}/{epoch_index}")
        expected = np.asarray(row["estimated_receiver_state"], dtype=np.float64)
        if expected.shape != (7,):
            raise RuntimeError(f"published original state missing: {path.name}/{epoch_index}")
        difference = np.abs(series.state[epoch_index] - expected)
        maximum_state_difference = max(maximum_state_difference, float(np.max(difference)))
        maximum_ecef_difference = max(maximum_ecef_difference, float(np.max(difference[:3])))
        maximum_clock_difference = max(maximum_clock_difference, float(np.max(difference[3:])))
        exact_state_count += int(np.array_equal(series.state[epoch_index], expected))
        if not np.allclose(series.state[epoch_index], expected, rtol=0.0, atol=STATE_REGRESSION_ATOL):
            raise RuntimeError(f"original state regression failed: {path.name}/{epoch_index}")
        if int(series.rank[epoch_index]) != int(row["rank"]):
            raise RuntimeError(f"original final rank mismatch: {path.name}/{epoch_index}")
        if int(series.iteration_count[epoch_index]) != int(row["iteration_count"]):
            raise RuntimeError(f"original iteration count mismatch: {path.name}/{epoch_index}")
        expected_condition = float(row["condition_number"])
        actual_condition = float(series.condition_number[epoch_index])
        maximum_condition_difference = max(
            maximum_condition_difference, abs(actual_condition - expected_condition)
        )
        if not math.isclose(
            actual_condition,
            expected_condition,
            rel_tol=CONDITION_REGRESSION_RTOL,
            abs_tol=CONDITION_REGRESSION_ATOL,
        ):
            raise RuntimeError(f"original condition regression failed: {path.name}/{epoch_index}")
    return {
        "architecture": architecture,
        "seed": seed,
        "epoch_count": len(rows),
        "solved_count": int(np.count_nonzero(series.solved)),
        "failed_count": int(len(rows) - np.count_nonzero(series.solved)),
        "solved_status_exact": True,
        "rank_exact": True,
        "iteration_count_exact": True,
        "state_atol": STATE_REGRESSION_ATOL,
        "maximum_absolute_state_difference": maximum_state_difference,
        "maximum_absolute_ecef_difference_m": maximum_ecef_difference,
        "maximum_absolute_clock_state_difference_m": maximum_clock_difference,
        "exact_state_count": exact_state_count,
        "condition_rtol": CONDITION_REGRESSION_RTOL,
        "condition_atol": CONDITION_REGRESSION_ATOL,
        "maximum_absolute_condition_difference": maximum_condition_difference,
        "passed": True,
    }


def compute_original_positions(
    inputs: AuthoritativeInputs,
    frozen_results_dir: Path = DEFAULT_FROZEN_RESULTS_DIR,
) -> tuple[dict[tuple[str, int, str], PositionSeries], list[dict[str, object]]]:
    results: dict[tuple[str, int, str], PositionSeries] = {}
    regression: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            response = inputs.responses[(architecture, seed)]
            if architecture == "TDL-B":
                series = solve_series(inputs.dataset, route="bias", bias_m=response["bias_m"])
            elif architecture == "TDL-W":
                series = solve_series(inputs.dataset, route="weight", weight=response["weight"])
            else:
                series = solve_series(
                    inputs.dataset,
                    route="hybrid",
                    bias_m=response["bias_m"],
                    weight=response["weight"],
                )
            results[(architecture, seed, ORIGINAL_VARIANT[architecture])] = series
            regression.append(
                compare_original_to_frozen(
                    architecture, seed, series, frozen_results_dir.resolve()
                )
            )
            print(f"original regression passed: {architecture} seed {seed}", flush=True)
    return results, regression


def validate_uniform_weight_scaling(
    dataset: Mapping[str, np.ndarray], neutral: PositionSeries
) -> dict[str, object]:
    indices = tuple(np.linspace(0, EXPECTED_EPOCHS - 1, 64, dtype=np.int64).tolist())
    row_count = int(dataset["features"].shape[0])
    comparisons: list[dict[str, object]] = []
    for scale in (0.25, 7.0):
        scaled = solve_series(
            dataset,
            route="weight",
            weight=np.full(row_count, scale, dtype=np.float64),
            epoch_indices=indices,
        )
        reference = neutral.state[np.asarray(indices)]
        if not np.array_equal(scaled.solved, neutral.solved[np.asarray(indices)]):
            raise RuntimeError("constant weight scaling changed solved status")
        differences = np.abs(scaled.state - reference)
        maximum = float(np.nanmax(differences))
        if maximum > UNIFORM_SCALE_ATOL:
            raise RuntimeError(f"constant weight scaling changed WLS state by {maximum}")
        comparisons.append(
            {
                "scale": scale,
                "epoch_count": len(indices),
                "maximum_absolute_state_difference": maximum,
                "atol": UNIFORM_SCALE_ATOL,
                "passed": True,
            }
        )
    return {
        "algebra": (
            "For constant c>0, W=cI gives inv(H.T@cI@H)@H.T@cI@v "
            "= inv(H.T@H)@H.T@v in exact arithmetic."
        ),
        "uniform_value_selected": 1.0,
        "sample_policy": "64 evenly spaced accepted epochs, including both endpoints",
        "comparisons": comparisons,
        "passed": True,
    }


def compare_neutral_to_stored_initializer(
    dataset: Mapping[str, np.ndarray], neutral: PositionSeries
) -> dict[str, object]:
    initial = np.asarray(dataset["epoch_ols_initial_state"], dtype=np.float64)
    differences = neutral.state - initial
    position_distance = np.linalg.norm(differences[:, :3], axis=1)
    return {
        "expected_to_be_equal": False,
        "exact_state_count": int(sum(np.array_equal(a, b) for a, b in zip(neutral.state, initial, strict=True))),
        "maximum_absolute_state_difference": float(np.max(np.abs(differences))),
        "position_displacement_m": {
            "mean": float(np.mean(position_distance)),
            "median": float(np.median(position_distance)),
            "p95": linear_quantile(position_distance, 0.95),
            "maximum": float(np.max(position_distance)),
        },
        "reason": (
            "epoch_ols_initial_state is the converged preprocessing get_ls_pnt_pos result, "
            "which uses the RTKLIB ionosphere/troposphere correction path. The released "
            "Torch WLS reproduction starts from that state but uses its established "
            "zero-atmospheric-term observation model, so an additional neutral solve is not identical."
        ),
    }


def compute_ablation_positions(
    inputs: AuthoritativeInputs,
    original: dict[tuple[str, int, str], PositionSeries],
) -> tuple[dict[tuple[str, int, str], PositionSeries], dict[str, object], dict[str, object]]:
    results = dict(original)
    ones = np.ones(EXPECTED_ROWS, dtype=np.float64)
    neutral = solve_series(inputs.dataset, route="weight", weight=ones)
    uniform_validation = validate_uniform_weight_scaling(inputs.dataset, neutral)
    neutral_regression = compare_neutral_to_stored_initializer(inputs.dataset, neutral)
    for seed in SEEDS:
        results[("TDL-B", seed, "B-neutral")] = neutral
        results[("TDL-W", seed, "W-neutral")] = neutral
        results[("TDL-BW", seed, "BW-neutral")] = neutral
    for seed in SEEDS:
        response = inputs.responses[("TDL-BW", seed)]
        results[("TDL-BW", seed, "BW-bias-only")] = solve_series(
            inputs.dataset, route="bias", bias_m=response["bias_m"]
        )
        print(f"ablation positioning completed: TDL-BW seed {seed} bias-only", flush=True)
        results[("TDL-BW", seed, "BW-weight-only")] = solve_series(
            inputs.dataset, route="weight", weight=response["weight"]
        )
        print(f"ablation positioning completed: TDL-BW seed {seed} weight-only", flush=True)
    expected = {
        (architecture, seed, variant)
        for architecture in ARCHITECTURES
        for seed in SEEDS
        for variant in VARIANTS[architecture]
    }
    if set(results) != expected:
        raise RuntimeError("computed ablation plan does not match the declared 3 x 10 design")
    return results, uniform_validation, neutral_regression


def evaluate_positions(
    results: Mapping[tuple[str, int, str], PositionSeries],
    reference_manifest_path: Path = REFERENCE_MANIFEST_PATH,
) -> tuple[dict[tuple[str, int, str], EvaluatedSeries], list[dict[str, object]], dict[str, object]]:
    if sha256_file(reference_manifest_path) != REFERENCE_MANIFEST_SHA256:
        raise RuntimeError("trusted reference manifest SHA-256 changed")
    reference_manifest, reference = load_reference_manifest(reference_manifest_path)
    evaluated: dict[tuple[str, int, str], EvaluatedSeries] = {}
    per_seed: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            for variant in VARIANTS[architecture]:
                key = (architecture, seed, variant)
                series = results[key]
                solved = series.solved
                if not np.any(solved):
                    raise RuntimeError(f"no solved epochs for {key}")
                errors = evaluate_ecef_positions(series.state[solved, :3], reference)
                enu = np.full((EXPECTED_EPOCHS, 3), np.nan, dtype=np.float64)
                e2d = np.full(EXPECTED_EPOCHS, np.nan, dtype=np.float64)
                e3d = np.full(EXPECTED_EPOCHS, np.nan, dtype=np.float64)
                enu[solved] = errors["enu_error_m"]
                e2d[solved] = errors["e2d_m"]
                e3d[solved] = errors["e3d_m"]
                item = EvaluatedSeries(
                    positioning=series,
                    enu_error_m=enu,
                    e2d_m=e2d,
                    abs_up_m=np.abs(enu[:, 2]),
                    e3d_m=e3d,
                )
                evaluated[key] = item
                metrics = compute_error_metrics(enu[solved], EXPECTED_EPOCHS)
                per_seed.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "ablation": variant,
                        "solved_epochs": int(metrics["solved_epochs"]),
                        "failed_epochs": int(EXPECTED_EPOCHS - int(metrics["solved_epochs"])),
                        "e_mean_m": metrics["e_mean_m"],
                        "e_rms_m": metrics["e_rms_m"],
                        "n_mean_m": metrics["n_mean_m"],
                        "n_rms_m": metrics["n_rms_m"],
                        "u_mean_m": metrics["u_mean_m"],
                        "u_rms_m": metrics["u_rms_m"],
                        **{
                            name: metrics[name]
                            for name in (
                                "e2d_mean_m",
                                "e2d_median_m",
                                "e2d_rms_m",
                                "e2d_p68_m",
                                "e2d_p95_m",
                                "e2d_max_m",
                                "e3d_mean_m",
                                "e3d_median_m",
                                "e3d_rms_m",
                                "e3d_p68_m",
                                "e3d_p95_m",
                                "e3d_max_m",
                            )
                        },
                    }
                )
    reference_record = {
        "station_id": reference_manifest["station_id"],
        "manifest_path": str(reference_manifest_path.resolve()),
        "manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "trusted_ecef_m": reference.tolist(),
        "loaded_only_after_positioning_and_regression": True,
    }
    return evaluated, per_seed, reference_record


def across_seed_rows(per_seed: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    identifiers = {"architecture", "seed", "ablation"}
    output: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        for variant in VARIANTS[architecture]:
            selected = sorted(
                (row for row in per_seed if row["architecture"] == architecture and row["ablation"] == variant),
                key=lambda row: int(row["seed"]),
            )
            if [int(row["seed"]) for row in selected] != list(SEEDS):
                raise RuntimeError(f"incomplete seed set for {architecture}/{variant}")
            for metric in [name for name in selected[0] if name not in identifiers]:
                values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
                output.append(
                    {
                        "architecture": architecture,
                        "ablation": variant,
                        "per_seed_metric": metric,
                        "seed_count": 10,
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "population_sd": float(np.std(values, ddof=0)),
                        "iqr": linear_quantile(values, 0.75) - linear_quantile(values, 0.25),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                )
    return output


def paired_delta_summary(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired summary requires a finite non-empty vector")
    improved = values < -PAIRED_EQUAL_ATOL_M
    worsened = values > PAIRED_EQUAL_ATOL_M
    equal = ~(improved | worsened)
    return {
        "common_solved_epochs": int(values.size),
        "mean_delta_m": float(np.mean(values)),
        "median_delta_m": float(np.median(values)),
        "p05_delta_m": linear_quantile(values, 0.05),
        "p95_delta_m": linear_quantile(values, 0.95),
        "fraction_improved": float(np.mean(improved)),
        "fraction_worsened": float(np.mean(worsened)),
        "fraction_equal": float(np.mean(equal)),
        "equal_tolerance_m": PAIRED_EQUAL_ATOL_M,
    }


def write_paired_epoch_and_summaries(
    path: Path,
    evaluated: Mapping[tuple[str, int, str], EvaluatedSeries],
) -> list[dict[str, object]]:
    fields = (
        "architecture",
        "seed",
        "comparison",
        "candidate_variant",
        "reference_variant",
        "accepted_epoch_index",
        "delta_definition",
        "delta_e2d_m",
        "delta_abs_up_m",
        "delta_e3d_m",
    )
    summaries: list[dict[str, object]] = []
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                for candidate, reference in PAIRED_COMPARISONS[architecture]:
                    candidate_series = evaluated[(architecture, seed, candidate)]
                    reference_series = evaluated[(architecture, seed, reference)]
                    common = candidate_series.positioning.solved & reference_series.positioning.solved
                    indices = np.flatnonzero(common)
                    deltas = {
                        "e2d": candidate_series.e2d_m[common] - reference_series.e2d_m[common],
                        "abs_up": candidate_series.abs_up_m[common] - reference_series.abs_up_m[common],
                        "e3d": candidate_series.e3d_m[common] - reference_series.e3d_m[common],
                    }
                    comparison = f"{candidate}_minus_{reference}"
                    for position, epoch_index in enumerate(indices):
                        writer.writerow(
                            {
                                "architecture": architecture,
                                "seed": seed,
                                "comparison": comparison,
                                "candidate_variant": candidate,
                                "reference_variant": reference,
                                "accepted_epoch_index": int(epoch_index),
                                "delta_definition": "candidate error minus reference-variant error",
                                "delta_e2d_m": float(deltas["e2d"][position]),
                                "delta_abs_up_m": float(deltas["abs_up"][position]),
                                "delta_e3d_m": float(deltas["e3d"][position]),
                            }
                        )
                    for metric, values in deltas.items():
                        summaries.append(
                            {
                                "architecture": architecture,
                                "seed": seed,
                                "comparison": comparison,
                                "candidate_variant": candidate,
                                "reference_variant": reference,
                                "error_metric": metric,
                                "delta_definition": "candidate error minus reference-variant error",
                                **paired_delta_summary(values),
                            }
                        )
    return summaries


def paired_across_seed_rows(per_seed: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    identifiers = {
        "architecture",
        "seed",
        "comparison",
        "candidate_variant",
        "reference_variant",
        "error_metric",
        "delta_definition",
        "equal_tolerance_m",
    }
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in per_seed:
        groups[(str(row["architecture"]), str(row["comparison"]), str(row["error_metric"]))].append(row)
    output: list[dict[str, object]] = []
    for (architecture, comparison, metric), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise RuntimeError(f"paired aggregation lacks seeds 0..9: {comparison}/{metric}")
        for per_seed_metric in [name for name in rows[0] if name not in identifiers]:
            values = np.asarray([float(row[per_seed_metric]) for row in rows])
            output.append(
                {
                    "architecture": architecture,
                    "comparison": comparison,
                    "error_metric": metric,
                    "per_seed_metric": per_seed_metric,
                    "seed_count": 10,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "population_sd": float(np.std(values, ddof=0)),
                    "iqr": linear_quantile(values, 0.75) - linear_quantile(values, 0.25),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return output


def factorial_interaction(
    full_squared_error: np.ndarray,
    bias_only_squared_error: np.ndarray,
    weight_only_squared_error: np.ndarray,
    neutral_squared_error: np.ndarray,
) -> np.ndarray:
    """Exact numerical 2x2 contrast for per-epoch squared 3D error."""

    return (
        np.asarray(full_squared_error)
        - np.asarray(bias_only_squared_error)
        - np.asarray(weight_only_squared_error)
        + np.asarray(neutral_squared_error)
    )


def write_factorial_outputs(
    path: Path,
    evaluated: Mapping[tuple[str, int, str], EvaluatedSeries],
) -> list[dict[str, object]]:
    fields = ("seed", "accepted_epoch_index", "interaction_m2", "formula")
    summaries: list[dict[str, object]] = []
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for seed in SEEDS:
            full = evaluated[("TDL-BW", seed, "BW-full")]
            bias = evaluated[("TDL-BW", seed, "BW-bias-only")]
            weight = evaluated[("TDL-BW", seed, "BW-weight-only")]
            neutral = evaluated[("TDL-BW", seed, "BW-neutral")]
            common = (
                full.positioning.solved
                & bias.positioning.solved
                & weight.positioning.solved
                & neutral.positioning.solved
            )
            interaction = factorial_interaction(
                np.square(full.e3d_m[common]),
                np.square(bias.e3d_m[common]),
                np.square(weight.e3d_m[common]),
                np.square(neutral.e3d_m[common]),
            )
            for epoch_index, value in zip(np.flatnonzero(common), interaction, strict=True):
                writer.writerow(
                    {
                        "seed": seed,
                        "accepted_epoch_index": int(epoch_index),
                        "interaction_m2": float(value),
                        "formula": "L(full)-L(bias-only)-L(weight-only)+L(neutral)",
                    }
                )
            summaries.append(
                {
                    "seed": seed,
                    "common_solved_epochs": int(interaction.size),
                    "mean_m2": float(np.mean(interaction)),
                    "median_m2": float(np.median(interaction)),
                    "p05_m2": linear_quantile(interaction, 0.05),
                    "p95_m2": linear_quantile(interaction, 0.95),
                    "fraction_positive": float(np.mean(interaction > 0.0)),
                    "fraction_negative": float(np.mean(interaction < 0.0)),
                    "fraction_zero": float(np.mean(interaction == 0.0)),
                }
            )
    return summaries


def factorial_across_seed_rows(per_seed: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for metric in [name for name in per_seed[0] if name != "seed"]:
        values = np.asarray([float(row[metric]) for row in per_seed], dtype=np.float64)
        output.append(
            {
                "per_seed_metric": metric,
                "seed_count": 10,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "population_sd": float(np.std(values, ddof=0)),
                "iqr": linear_quantile(values, 0.75) - linear_quantile(values, 0.25),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return output


def epoch_result_arrays(
    evaluated: Mapping[tuple[str, int, str], EvaluatedSeries],
    dataset: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    architecture_values: list[np.ndarray] = []
    seed_values: list[np.ndarray] = []
    variant_values: list[np.ndarray] = []
    epoch_values: list[np.ndarray] = []
    state_values: list[np.ndarray] = []
    solved_values: list[np.ndarray] = []
    status_values: list[np.ndarray] = []
    rank_values: list[np.ndarray] = []
    condition_values: list[np.ndarray] = []
    iteration_values: list[np.ndarray] = []
    enu_values: list[np.ndarray] = []
    e2d_values: list[np.ndarray] = []
    abs_up_values: list[np.ndarray] = []
    e3d_values: list[np.ndarray] = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            for variant in VARIANTS[architecture]:
                item = evaluated[(architecture, seed, variant)]
                architecture_values.append(np.full(EXPECTED_EPOCHS, architecture, dtype="U6"))
                seed_values.append(np.full(EXPECTED_EPOCHS, seed, dtype=np.int64))
                variant_values.append(np.full(EXPECTED_EPOCHS, variant, dtype="U14"))
                epoch_values.append(np.arange(EXPECTED_EPOCHS, dtype=np.int64))
                state_values.append(item.positioning.state)
                solved_values.append(item.positioning.solved.astype(np.uint8))
                status_values.append(item.positioning.status)
                rank_values.append(item.positioning.rank)
                condition_values.append(item.positioning.condition_number)
                iteration_values.append(item.positioning.iteration_count)
                enu_values.append(item.enu_error_m)
                e2d_values.append(item.e2d_m)
                abs_up_values.append(item.abs_up_m)
                e3d_values.append(item.e3d_m)
    return {
        "schema_version": np.asarray([1], dtype=np.int64),
        "architecture": np.concatenate(architecture_values),
        "seed": np.concatenate(seed_values),
        "ablation": np.concatenate(variant_values),
        "accepted_epoch_index": np.concatenate(epoch_values),
        "source_split_epoch_index": np.tile(dataset["epoch_split_index"], 80),
        "timestamp_gpst_like_s": np.tile(dataset["epoch_time_gpst_like_s"], 80),
        "estimated_receiver_state": np.concatenate(state_values),
        "solved": np.concatenate(solved_values),
        "solution_status": np.concatenate(status_values),
        "final_rank": np.concatenate(rank_values),
        "final_condition_number": np.concatenate(condition_values),
        "iteration_count": np.concatenate(iteration_values),
        "enu_error_m": np.concatenate(enu_values),
        "e2d_m": np.concatenate(e2d_values),
        "abs_up_m": np.concatenate(abs_up_values),
        "e3d_m": np.concatenate(e3d_values),
    }


def _save_figure(figure: object, path: Path) -> None:
    figure.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "validation.tdl_3feature_paper_analysis.positioning_ablation"},
    )


def make_plots(
    evaluated: Mapping[tuple[str, int, str], EvaluatedSeries],
    per_seed: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir()
    created: list[Path] = []
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    for architecture in ARCHITECTURES:
        for value_name, label, suffix in (
            ("e3d_m", "3D error [m]", "3d_error_ecdf"),
            ("abs_up_m", "absolute vertical error [m]", "vertical_error_ecdf"),
        ):
            figure, axis = plt.subplots(figsize=(7.2, 4.6))
            for color, variant in zip(colors, VARIANTS[architecture], strict=False):
                for seed in SEEDS:
                    values = getattr(evaluated[(architecture, seed, variant)], value_name)
                    values = np.sort(values[np.isfinite(values)])
                    y = np.arange(1, values.size + 1) / values.size
                    axis.step(
                        values,
                        y,
                        where="post",
                        color=color,
                        alpha=0.22,
                        linewidth=0.65,
                        label=variant if seed == 0 else None,
                    )
            axis.set(xlabel=label, ylabel="ECDF", ylim=(0.0, 1.01), title=f"{architecture} ablation")
            axis.grid(alpha=0.2)
            axis.legend()
            path = plot_dir / f"{ARCHITECTURE_SLUGS[architecture]}_{suffix}.png"
            _save_figure(figure, path)
            plt.close(figure)
            created.append(path)

        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for color, (candidate, reference) in zip(colors, PAIRED_COMPARISONS[architecture], strict=False):
            for seed in SEEDS:
                first = evaluated[(architecture, seed, candidate)]
                second = evaluated[(architecture, seed, reference)]
                common = first.positioning.solved & second.positioning.solved
                delta = np.sort(first.e3d_m[common] - second.e3d_m[common])
                y = np.arange(1, delta.size + 1) / delta.size
                axis.step(
                    delta,
                    y,
                    where="post",
                    color=color,
                    alpha=0.22,
                    linewidth=0.65,
                    label=f"{candidate} - {reference}" if seed == 0 else None,
                )
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set(
            xlabel="paired 3D-error delta [m]",
            ylabel="ECDF",
            ylim=(0.0, 1.01),
            title=f"{architecture} paired ablation deltas",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
        path = plot_dir / f"{ARCHITECTURE_SLUGS[architecture]}_paired_3d_delta_ecdf.png"
        _save_figure(figure, path)
        plt.close(figure)
        created.append(path)

    labels: list[str] = []
    values: list[list[float]] = []
    for architecture in ARCHITECTURES:
        for variant in VARIANTS[architecture]:
            labels.append(variant)
            values.append(
                [
                    float(row["e3d_rms_m"])
                    for row in per_seed
                    if row["architecture"] == architecture and row["ablation"] == variant
                ]
            )
    figure, axis = plt.subplots(figsize=(10.2, 4.8))
    axis.boxplot(values, tick_labels=labels, showmeans=True)
    axis.set(ylabel="per-seed 3D RMS error [m]", title="Across-seed ablation comparison")
    axis.tick_params(axis="x", labelrotation=30)
    axis.grid(axis="y", alpha=0.2)
    path = plot_dir / "across_seed_3d_rms_comparison.png"
    _save_figure(figure, path)
    plt.close(figure)
    created.append(path)
    return created


def _artifact_records(paths: Iterable[Path], root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def run_ablation(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    dataset_path: Path = DEFAULT_IBIZA,
    model_response_dir: Path = DEFAULT_MODEL_RESPONSE,
    frozen_results_dir: Path = DEFAULT_FROZEN_RESULTS_DIR,
    reference_manifest_path: Path = REFERENCE_MANIFEST_PATH,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing scientific output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    inputs = validate_authoritative_inputs(
        dataset_path=dataset_path,
        model_response_dir=model_response_dir,
        frozen_results_dir=frozen_results_dir,
    )
    original, original_regression = compute_original_positions(inputs, frozen_results_dir)
    results, uniform_validation, neutral_regression = compute_ablation_positions(inputs, original)
    # This is the first point at which the propagated EPN reference is read.
    evaluated, per_seed, reference_record = evaluate_positions(results, reference_manifest_path)
    across = across_seed_rows(per_seed)

    with tempfile.TemporaryDirectory(prefix="positioning-ablation-", dir=output_dir.parent) as temporary:
        work = Path(temporary) / "publish"
        work.mkdir()
        epoch_path = work / "positioning_ablation_epoch_results.npz"
        write_deterministic_npz(epoch_path, epoch_result_arrays(evaluated, inputs.dataset))
        per_seed_path = work / "positioning_ablation_per_seed.csv"
        across_path = work / "positioning_ablation_across_seed.csv"
        _write_csv(per_seed_path, per_seed)
        _write_csv(across_path, across)
        paired_epoch_path = work / "positioning_ablation_paired_epoch.csv"
        paired_summary = write_paired_epoch_and_summaries(paired_epoch_path, evaluated)
        paired_summary_path = work / "positioning_ablation_paired_summary.csv"
        _write_csv(paired_summary_path, paired_summary)
        paired_across = paired_across_seed_rows(paired_summary)
        paired_across_path = work / "positioning_ablation_paired_across_seed.csv"
        _write_csv(paired_across_path, paired_across)
        factorial_epoch_path = work / "tdl_bw_factorial_interaction.csv"
        factorial_summary = write_factorial_outputs(factorial_epoch_path, evaluated)
        factorial_summary_path = work / "tdl_bw_factorial_interaction_summary.csv"
        _write_csv(factorial_summary_path, factorial_summary)
        factorial_across = factorial_across_seed_rows(factorial_summary)
        factorial_across_path = work / "tdl_bw_factorial_interaction_across_seed.csv"
        _write_csv(factorial_across_path, factorial_across)
        regression_path = work / "original_position_regression.json"
        regression_path.write_text(_canonical_json(original_regression), encoding="utf-8")
        neutral_path = work / "neutral_baseline_regression.json"
        neutral_path.write_text(
            _canonical_json(
                {
                    "uniform_weight_scaling": uniform_validation,
                    "stored_initializer_comparison": neutral_regression,
                }
            ),
            encoding="utf-8",
        )
        plot_paths = make_plots(evaluated, per_seed, work)

        assert_frozen_inputs_unchanged(inputs.frozen_validation)  # type: ignore[arg-type]
        after_snapshot = {path: sha256_file(Path(path)) for path in inputs.immutable_snapshot}
        if after_snapshot != inputs.immutable_snapshot:
            raise RuntimeError("an immutable preprocessing/model-response/checkpoint source changed")

        generated = [
            epoch_path,
            per_seed_path,
            across_path,
            paired_epoch_path,
            paired_summary_path,
            paired_across_path,
            factorial_epoch_path,
            factorial_summary_path,
            factorial_across_path,
            regression_path,
            neutral_path,
            *plot_paths,
        ]
        artifact_records = _artifact_records(generated, work)
        hash_index_path = work / "positioning_ablation_artifact_hashes.csv"
        _write_csv(hash_index_path, artifact_records)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "completed_controlled_frozen_output_positioning_ablation",
            "repository": {
                "commit": _git_value("rev-parse", "HEAD"),
                "dirty_at_generation": bool(_git_value("status", "--porcelain")),
            },
            "scientific_scope": (
                "descriptive association of already-frozen bias/weight outputs with Ibiza "
                "positioning under the unchanged historical solver; not full causal identification"
            ),
            "authoritative_inputs": inputs.provenance,
            "reference": reference_record,
            "solver": {
                "implementation": "validation.paper_weightnet.core.solve_paper_weighted_position",
                "bias_wrapper": "validation.paper_biasnet.core.solve_paper_bias_position",
                "hybrid_wrapper": "validation.paper_hybrid.core.solve_paper_hybrid_position",
                "bias_application": "corrected_pseudorange_m - predicted_bias_m",
                "weight_application": "W=diag(predicted_weight) used directly once in (H.T@W@H)^-1@H.T@W@v",
                "uniform_weight": 1.0,
                "initial_state": "epoch_ols_initial_state, seven states",
                "active_state_indices": "ECEF indices 0,1,2 plus sorted observed constellation clock indices 3..6",
                "convergence_tolerance": 1.0e-4,
                "maximum_iterations": 10,
                "success": "finite state, final normal matrix full rank with finite condition, and converged before maximum iterations",
                "implementation_modified": False,
            },
            "ablation_plan": VARIANTS,
            "original_position_regression": {
                "all_30_passed": True,
                "state_atol": STATE_REGRESSION_ATOL,
                "condition_rtol": CONDITION_REGRESSION_RTOL,
                "condition_atol": CONDITION_REGRESSION_ATOL,
                "tolerance_justification": (
                    "The immutable response artifacts were forwarded as one 73,204-row batch, "
                    "whereas the historical inference forwarded one epoch at a time. A targeted "
                    "validation-only TDL-BW seed-1 epoch-8 forward found maximum weight difference "
                    "4.743384504624082e-20 and exact bias. Across all TDL-BW originals this was "
                    "amplified by highly conditioned weighted normal matrices to at most "
                    "2.0209699869155884e-7 in the seven-state solution; statuses, ranks, and "
                    "iteration counts remained exact. The 5e-7 state tolerance is sub-micrometre."
                ),
                "validation_only_forward_diagnostic": {
                    "architecture": "TDL-BW",
                    "seed": 1,
                    "accepted_epoch_index": 8,
                    "maximum_absolute_bias_difference_m": 0.0,
                    "maximum_absolute_weight_difference": 4.743384504624082e-20,
                    "checkpoint_deserialized": True,
                    "purpose": "explain batched-versus-per-epoch floating-point regression discrepancy only",
                },
                "records": original_regression,
            },
            "neutral_baseline_regression": {
                "uniform_weight_scaling": uniform_validation,
                "stored_initializer_comparison": neutral_regression,
            },
            "metric_policy": {
                "implementation": "validation.ibiza_generalization.evaluate_ground_truth",
                "same_propagated_epn_reference_for_all_variants": True,
                "quantile_method": "numpy linear",
                "axis_errors": "signed estimated-minus-reference ENU",
                "paired_delta": "candidate/original error minus reference-variant error",
                "paired_equal_tolerance_m": PAIRED_EQUAL_ATOL_M,
                "factorial_interaction": "L(full)-L(bias-only)-L(weight-only)+L(neutral), L=squared 3D position error",
                "factorial_interaction_interpretation": "descriptive numerical contrast, not a physical causal interaction",
                "seed_aggregation": "summarize ten per-seed values; never pool seed x epoch rows as independent",
            },
            "counts": {
                "accepted_epochs": EXPECTED_EPOCHS,
                "satellite_rows": EXPECTED_ROWS,
                "per_seed_metric_rows": len(per_seed),
                "across_seed_metric_rows": len(across),
                "paired_per_seed_summary_rows": len(paired_summary),
                "paired_across_seed_rows": len(paired_across),
                "factorial_per_seed_summary_rows": len(factorial_summary),
                "factorial_across_seed_rows": len(factorial_across),
            },
            "controls": {
                "neural_inference_rerun_for_ablation": False,
                "validation_only_single_epoch_forward_performed_before_ablation": True,
                "checkpoint_deserialized_for_ablation": False,
                "checkpoint_deserialized_for_validation_only_diagnostic": True,
                "checkpoint_modified": False,
                "feature_cache_modified": False,
                "frozen_model_response_modified": False,
                "preprocessing_rerun_or_modified": False,
                "wls_modified": False,
                "optimizer_created": False,
                "backward_called": False,
                "training_or_retraining": False,
                "normalization_fitted": False,
                "ground_truth_read_during_positioning": False,
                "ground_truth_passed_to_solver": False,
                "ground_truth_loaded_only_after_original_and_neutral_regressions": True,
                "same_epoch_satellite_support_all_variants": True,
                "same_initial_states_all_variants": True,
                "same_corrected_pseudorange_before_bias_all_variants": True,
                "same_satellite_positions_clocks_system_indices_all_variants": True,
                "only_frozen_bias_and_or_weight_inputs_differ": True,
                "immutable_sources_byte_identical_after": True,
            },
            "generated_files_excluding_manifest_and_hash_index": artifact_records,
            "artifact_hash_index": {
                "path": hash_index_path.name,
                "sha256": sha256_file(hash_index_path),
            },
        }
        manifest_path = work / "positioning_ablation_manifest.json"
        manifest_path.write_text(_canonical_json(manifest), encoding="utf-8")
        os.rename(work, output_dir)
    print(f"published positioning ablation: {output_dir}", flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_IBIZA)
    parser.add_argument("--model-response", type=Path, default=DEFAULT_MODEL_RESPONSE)
    parser.add_argument("--frozen-results", type=Path, default=DEFAULT_FROZEN_RESULTS_DIR)
    parser.add_argument("--reference-manifest", type=Path, default=REFERENCE_MANIFEST_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    run_ablation(
        output_dir=arguments.output,
        dataset_path=arguments.dataset,
        model_response_dir=arguments.model_response,
        frozen_results_dir=arguments.frozen_results,
        reference_manifest_path=arguments.reference_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
