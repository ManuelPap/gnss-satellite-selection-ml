#!/usr/bin/env python3
"""Zero-shot Ibiza evaluation for the ten frozen KLT3 HybridShareSysNet models.

Inference is deliberately completed and serialized before the EPN reference is
opened.  The reference is consumed only by the final, pure metric pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from validation.tasgnss_ibiza_solver_audit.core import (
    IBIZA_NPZ_SHA256,
    enu_errors,
    trusted_reference_ecef,
)

from .core import (
    DEFAULT_OUTPUT_ROOT,
    FEATURE_NAMES,
    SEEDS,
    distribution_metrics,
    feature_tensor,
    import_current_stack,
    initialize_solver_cache,
    make_model,
    paired_delta_metrics,
    percentile_metrics,
    record_feature_parts,
    sha256_array,
    sha256_file,
    verify_provenance,
    write_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = REPOSITORY_ROOT.parent
IBIZA_DATA_ROOT = PHD_ROOT / "external_data/ibiza_2025_01_01"
DEFAULT_OBSERVATION = (
    IBIZA_DATA_ROOT / "raw/observation/IBIZ00ESP_R_20250010000_01D_30S_MO.rnx"
)
DEFAULT_NAVIGATION = (
    IBIZA_DATA_ROOT / "raw/navigation/BRDM00DLR_S_20250010000_01D_MN.rnx"
)
DEFAULT_HISTORICAL_DATASET = IBIZA_DATA_ROOT / "derived/ibiza_preprocessed.npz"
DEFAULT_REFERENCE = REPOSITORY_ROOT / "validation/ibiza_generalization/ibiz00esp_reference.json"
DEFAULT_NEUTRAL_AUDIT = (
    PHD_ROOT
    / "external_data/tasgnss_comparison/ibiza_neutral/current_stack_positions_pre_ground_truth.npz"
)
DEFAULT_EVALUATION_ROOT = DEFAULT_OUTPUT_ROOT / "ibiza_evaluation"
DEFAULT_CHECKPOINT_MANIFEST = DEFAULT_OUTPUT_ROOT / "frozen_checkpoint_manifest.json"

EXPECTED_CURRENT_EPOCHS = 2880
EXPECTED_HISTORICAL_COMMON_EPOCHS = 2856
PRIMARY_SUPPORT = "primary_current_stack"
SECONDARY_SUPPORT = "secondary_historical_common"
SUPPORT_NAMES = (PRIMARY_SUPPORT, SECONDARY_SUPPORT)
POSITION_ARRAY_KEYS = (
    "feature_hash",
    "neutral_solver_input_hash",
    "learned_solver_input_hash",
    "model_output_hash",
    "neutral_position_ecef_m",
    "learned_position_ecef_m",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--navigation", type=Path, default=DEFAULT_NAVIGATION)
    parser.add_argument("--historical-dataset", type=Path, default=DEFAULT_HISTORICAL_DATASET)
    parser.add_argument("--neutral-audit", type=Path, default=DEFAULT_NEUTRAL_AUDIT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=None,
        help="non-scientific prefix-only smoke; requires exactly one --seed",
    )
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _hash_named_arrays(items: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode("utf-8"))
        digest.update(sha256_array(np.asarray(value)).encode("ascii"))
    return digest.hexdigest()


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_checkpoint_inventory(
    checkpoint_root: Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Load the authoritative manifest and verify all ten checkpoint hashes."""

    root = checkpoint_root.resolve()
    path = (manifest_path or root / "frozen_checkpoint_manifest.json").resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("seeds")
    if manifest.get("seed_count") != len(SEEDS) or not isinstance(entries, list):
        raise RuntimeError("frozen manifest must declare exactly ten checkpoints")
    if [entry.get("seed") for entry in entries] != list(SEEDS):
        raise RuntimeError("frozen manifest must contain ordered seeds 0-9")

    verified: list[dict[str, Any]] = []
    for seed, entry in zip(SEEDS, entries, strict=True):
        checkpoint = (root / f"seed_{seed}/multinet_3d.pth").resolve()
        recorded = Path(entry["checkpoint"]).resolve()
        if checkpoint != recorded:
            raise RuntimeError(
                f"seed {seed} checkpoint path differs from authoritative manifest: {recorded}"
            )
        actual_hash = sha256_file(checkpoint)
        if actual_hash != entry.get("checkpoint_sha256"):
            raise RuntimeError(f"seed {seed} checkpoint SHA-256 differs from frozen manifest")
        if entry.get("training_configuration", {}).get("training_datasets") != ["KLT3"]:
            raise RuntimeError(f"seed {seed} is not declared as KLT3-only training")
        if entry.get("training_configuration", {}).get("ibiza_used") is not False:
            raise RuntimeError(f"seed {seed} does not prove Ibiza exclusion from training")
        if len(entry.get("scaler_mean", [])) != len(FEATURE_NAMES):
            raise RuntimeError(f"seed {seed} frozen scaler has the wrong dimensionality")
        if len(entry.get("scaler_std", [])) != len(FEATURE_NAMES):
            raise RuntimeError(f"seed {seed} frozen scaler has the wrong dimensionality")
        verified.append(
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "checkpoint_sha256": actual_hash,
                "scaler_mean": entry["scaler_mean"],
                "scaler_std": entry["scaler_std"],
            }
        )
    return {
        "manifest_path": path,
        "manifest_sha256": sha256_file(path),
        "seed_count": len(verified),
        "seeds": verified,
    }


def assert_checkpoint_inventory_unchanged(inventory: Mapping[str, Any]) -> None:
    if sha256_file(Path(inventory["manifest_path"])) != inventory["manifest_sha256"]:
        raise RuntimeError("authoritative frozen checkpoint manifest changed during evaluation")
    for entry in inventory["seeds"]:
        checkpoint = Path(entry["checkpoint"])
        if sha256_file(checkpoint) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"frozen checkpoint changed during evaluation: {checkpoint}")


def load_frozen_model(entry: Mapping[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Construct an inference-only model using the scaler embedded in its checkpoint."""

    checkpoint = Path(entry["checkpoint"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mean = state["seq.0.mean"].detach().cpu().numpy().astype(np.float64, copy=True)
    std = state["seq.0.std"].detach().cpu().numpy().astype(np.float64, copy=True)
    if mean.shape != (9,) or std.shape != (9,) or np.any(std == 0):
        raise RuntimeError("checkpoint-embedded scaler is invalid")
    np.testing.assert_array_equal(mean, np.asarray(entry["scaler_mean"], dtype=np.float64))
    np.testing.assert_array_equal(std, np.asarray(entry["scaler_std"], dtype=np.float64))
    model = make_model(mean, std, device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("frozen model is not fully in evaluation mode")
    controls = {
        "model_class": type(model).__name__,
        "model_training": model.training,
        "all_modules_evaluation_mode": all(not module.training for module in model.modules()),
        "scaler_source": "checkpoint state_dict seq.0.mean and seq.0.std",
        "scaler_refit_on_ibiza": False,
        "scaler_mean_sha256": sha256_array(mean),
        "scaler_std_sha256": sha256_array(std),
    }
    return model, controls


def historical_common_epoch_ids(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if sha256_file(path) != IBIZA_NPZ_SHA256:
        raise RuntimeError("frozen historical Ibiza dataset SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as data:
        epoch_ids = data["epoch_split_index"].astype(np.int64)
        timestamps = data["epoch_time_gpst_like_s"].astype(np.float64)
    if epoch_ids.shape != (EXPECTED_HISTORICAL_COMMON_EPOCHS,):
        raise RuntimeError("historical-common Ibiza support is not the frozen 2,856 epochs")
    if np.unique(epoch_ids).size != epoch_ids.size or np.any(np.diff(epoch_ids) <= 0):
        raise RuntimeError("historical-common epoch IDs are not unique and ordered")
    return epoch_ids, timestamps


def _time_of_epoch(epoch: Any) -> float:
    value = epoch.data[0].time
    return float(value.time + value.sec)


def _preprocess_cache_paths(output_dir: Path, smoke_epochs: int | None) -> tuple[Path, Path]:
    suffix = "full" if smoke_epochs is None else f"first_{smoke_epochs}"
    directory = output_dir / "preprocessed"
    return directory / f"ibiza_current_{suffix}.pkl", directory / f"ibiza_current_{suffix}.json"


def _validate_neutral_audit(
    audit_path: Path,
    epoch_times: np.ndarray,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    with np.load(audit_path, allow_pickle=False) as audit:
        expected_times = audit["epoch_time_gpst_like_s"][: epoch_times.size]
        expected_solved = audit["solved"][: epoch_times.size].astype(bool)
        expected_positions = audit["position_ecef_m"][: epoch_times.size]
    if not np.array_equal(epoch_times, expected_times):
        raise RuntimeError("current preprocessing timestamps differ from the validated neutral audit")
    record_by_id = {int(record["candidate_epoch_id"]): record for record in records}
    actual_solved = np.asarray([index in record_by_id for index in range(epoch_times.size)])
    if not np.array_equal(actual_solved, expected_solved):
        raise RuntimeError("current preprocessing status differs from the validated neutral audit")
    differences = [
        np.max(
            np.abs(
                np.asarray(record_by_id[index]["gnss"]["pos"], dtype=np.float64)
                - expected_positions[index]
            )
        )
        for index in range(epoch_times.size)
        if actual_solved[index]
    ]
    maximum_difference = float(max(differences, default=0.0))
    if maximum_difference > 1.0e-9:
        raise RuntimeError("neutral positions differ from the validated current-stack audit")
    return {
        "path": str(audit_path.resolve()),
        "sha256": sha256_file(audit_path),
        "timestamps_identical": True,
        "solve_status_identical": True,
        "maximum_absolute_position_difference_m": maximum_difference,
        "tolerance_m": 1.0e-9,
    }


def preprocess_ibiza(
    observation: Path,
    navigation: Path,
    neutral_audit: Path,
    *,
    smoke_epochs: int | None = None,
    progress_every: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create ground-truth-free current-stack records from original RINEX."""

    tas, _core = import_current_stack()
    observations, nav, _station = tas.read_obs(str(observation), str(navigation), opt="")
    epochs = tas.split_obs(observations, ref_obs=False)
    source_candidate_count = len(epochs)
    if source_candidate_count != EXPECTED_CURRENT_EPOCHS:
        raise RuntimeError(
            f"Ibiza RINEX produced {source_candidate_count} epochs, expected {EXPECTED_CURRENT_EPOCHS}"
        )
    if smoke_epochs is not None:
        if smoke_epochs <= 0 or smoke_epochs > source_candidate_count:
            raise ValueError("smoke epoch count is outside the Ibiza candidate range")
        epochs = epochs[:smoke_epochs]
    epoch_times = np.asarray([_time_of_epoch(epoch) for epoch in epochs], dtype=np.float64)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, epoch in enumerate(epochs):
        try:
            result = tas.wls_pnt_pos(
                epoch,
                nav,
                use_cache=True,
                return_residual=True,
                enable_torch=False,
                w=1,
                b=None,
                device="cpu",
            )
        except Exception as error:
            failures.append(
                {
                    "candidate_epoch_id": index,
                    "epoch_time_gpst_like_s": float(epoch_times[index]),
                    "message": f"{type(error).__name__}: {error}",
                }
            )
            tas.cache_data.pop(id(epoch), None)
            continue
        if result.get("status", False):
            records.append(
                {
                    "candidate_epoch_id": index,
                    "epoch_time_gpst_like_s": float(epoch_times[index]),
                    "gnss": result,
                }
            )
        else:
            failures.append(
                {
                    "candidate_epoch_id": index,
                    "epoch_time_gpst_like_s": float(epoch_times[index]),
                    "message": str(result.get("msg", "missing status message")),
                }
            )
        tas.cache_data.pop(id(epoch), None)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"preprocessed {index + 1}/{len(epochs)} Ibiza epochs", flush=True)
    audit = _validate_neutral_audit(neutral_audit, epoch_times, records)
    manifest = {
        "schema_version": 1,
        "ground_truth_available_to_preprocessing": False,
        "source_candidate_epochs": source_candidate_count,
        "evaluated_candidate_epochs": len(epochs),
        "preprocessing_accepted_epochs": len(records),
        "preprocessing_failed_epochs": len(failures),
        "failures": failures,
        "first_epoch_gpst_like_s": float(epoch_times[0]),
        "last_epoch_gpst_like_s": float(epoch_times[-1]),
        "duration_seconds": time.monotonic() - started,
        "inputs": {
            "observation": {
                "path": str(observation.resolve()),
                "sha256": sha256_file(observation),
            },
            "navigation": {
                "path": str(navigation.resolve()),
                "sha256": sha256_file(navigation),
            },
        },
        "validated_neutral_audit": audit,
    }
    return records, manifest


def load_or_preprocess_ibiza(
    output_dir: Path,
    observation: Path,
    navigation: Path,
    neutral_audit: Path,
    *,
    smoke_epochs: int | None = None,
    progress_every: int = 200,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_path, manifest_path = _preprocess_cache_paths(output_dir, smoke_epochs)
    if cache_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_sha256") != sha256_file(cache_path):
            raise RuntimeError("Ibiza preprocessing cache SHA-256 mismatch")
        for name, path in (("observation", observation), ("navigation", navigation)):
            if manifest["inputs"][name]["sha256"] != sha256_file(path):
                raise RuntimeError(f"cached Ibiza {name} input SHA-256 mismatch")
        with cache_path.open("rb") as stream:
            records = pickle.load(stream)
        return records, manifest

    records, manifest = preprocess_ibiza(
        observation,
        navigation,
        neutral_audit,
        smoke_epochs=smoke_epochs,
        progress_every=progress_every,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache_path)
    manifest["cache_path"] = str(cache_path.resolve())
    manifest["cache_sha256"] = sha256_file(cache_path)
    manifest["cache_bytes"] = cache_path.stat().st_size
    write_json(manifest_path, manifest)
    return records, manifest


def _reset_solver_cache(tas: Any, record: Mapping[str, Any]) -> None:
    gnss = record["gnss"]
    tas.cache_data[id(record)] = [
        np.asarray(gnss["pos"]).copy(),
        np.asarray(gnss["cb"]).copy(),
        None,
        None,
        gnss["data"],
        gnss["solve_data"],
        gnss["raw_data"],
    ]


def _base_solver_arrays(record: Mapping[str, Any]) -> list[tuple[str, np.ndarray]]:
    gnss = record["gnss"]
    solve = gnss["solve_data"]
    return [
        ("initial_position", np.asarray(gnss["pos"], dtype=np.float64)),
        ("initial_clock", np.asarray(gnss["cb"], dtype=np.float64)),
        ("satpos", np.asarray(solve["satpos"], dtype=np.float64)),
        ("pseudorange", np.asarray(solve["pr"], dtype=np.float64)),
        ("satellite_clock", np.asarray(solve["sdt"], dtype=np.float64)),
        ("sagnac", np.asarray(solve["sagnac"], dtype=np.float64)),
        ("ionosphere", np.asarray(solve["I"], dtype=np.float64)),
        ("troposphere", np.asarray(solve["T"], dtype=np.float64)),
        ("systems", np.asarray(solve["sys"], dtype="U4")),
    ]


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def infer_seed(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_count: int,
    checkpoint_entry: Mapping[str, Any],
    output_path: Path | None = None,
    progress_every: int = 200,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run reference-free neutral/learned inference for one frozen seed."""

    seed = int(checkpoint_entry["seed"])
    model, model_controls = load_frozen_model(checkpoint_entry)
    tas, _core = import_current_stack()
    initialize_solver_cache(tas, records)
    record_by_id = {int(record["candidate_epoch_id"]): record for record in records}
    if len(record_by_id) != len(records):
        raise RuntimeError("preprocessed Ibiza epoch IDs are not unique")

    epoch_ids = np.arange(candidate_count, dtype=np.int64)
    epoch_times = np.full(candidate_count, np.nan, dtype=np.float64)
    preprocessing_accepted = np.zeros(candidate_count, dtype=np.uint8)
    feature_accepted = np.zeros(candidate_count, dtype=np.uint8)
    neutral_solved = np.zeros(candidate_count, dtype=np.uint8)
    learned_solved = np.zeros(candidate_count, dtype=np.uint8)
    neutral_position = np.full((candidate_count, 3), np.nan, dtype=np.float64)
    learned_position = np.full((candidate_count, 3), np.nan, dtype=np.float64)
    feature_hash = np.full(candidate_count, "", dtype="U64")
    neutral_input_hash = np.full(candidate_count, "", dtype="U64")
    learned_input_hash = np.full(candidate_count, "", dtype="U64")
    model_output_hash = np.full(candidate_count, "", dtype="U64")
    observation_offsets = np.zeros(candidate_count + 1, dtype=np.int64)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    neutral_messages: Counter[str] = Counter()
    learned_messages: Counter[str] = Counter()
    feature_rejected_ids: list[int] = []
    started = time.monotonic()

    with torch.inference_mode():
        for candidate_id in range(candidate_count):
            record = record_by_id.get(candidate_id)
            observation_offsets[candidate_id + 1] = observation_offsets[candidate_id]
            if record is None:
                continue
            preprocessing_accepted[candidate_id] = 1
            epoch_times[candidate_id] = float(record["epoch_time_gpst_like_s"])
            if "gt" in record or "ground_truth" in record or "reference" in record:
                raise RuntimeError("ground truth leaked into an Ibiza inference record")
            parts = record_feature_parts(record)
            if parts is None:
                feature_rejected_ids.append(candidate_id)
                continue
            inputs = feature_tensor(parts, device="cpu")
            if inputs.ndim != 2 or inputs.shape[1] != len(FEATURE_NAMES):
                raise RuntimeError("Ibiza model features are not the frozen nine-column contract")
            feature_accepted[candidate_id] = 1
            feature_hash[candidate_id] = sha256_array(inputs.numpy())
            weight, bias = model(inputs)
            if not torch.all((weight >= 0) & (weight <= 1)) or not torch.all(bias >= 0):
                raise RuntimeError("model output violates frozen weight/bias semantics")
            weight_np = weight.detach().cpu().numpy().astype(np.float64, copy=True)
            bias_np = bias.detach().cpu().numpy().astype(np.float64, copy=True)
            weights.append(weight_np)
            biases.append(bias_np)
            observation_offsets[candidate_id + 1] += weight_np.size
            model_output_hash[candidate_id] = _hash_named_arrays(
                (("weight", weight_np), ("bias", bias_np))
            )
            base_arrays = _base_solver_arrays(record)
            observation_count = weight_np.size
            neutral_input_hash[candidate_id] = _hash_named_arrays(
                [
                    *base_arrays,
                    ("weight", np.ones(observation_count, dtype=np.float64)),
                    ("bias", np.zeros(observation_count, dtype=np.float64)),
                ]
            )
            learned_input_hash[candidate_id] = _hash_named_arrays(
                [*base_arrays, ("weight", weight_np), ("bias", bias_np)]
            )

            _reset_solver_cache(tas, record)
            neutral = tas.wls_pnt_pos(
                record,
                None,
                use_cache=True,
                w=1,
                b=None,
                enable_torch=False,
                device="cpu",
            )
            neutral_messages[str(neutral.get("msg", "missing status message"))] += 1
            _reset_solver_cache(tas, record)
            learned = tas.wls_pnt_pos(
                record,
                None,
                use_cache=True,
                w=weight,
                b=bias,
                enable_torch=True,
                device="cpu",
            )
            learned_messages[str(learned.get("msg", "missing status message"))] += 1
            if neutral.get("status", False):
                neutral_solved[candidate_id] = 1
                neutral_position[candidate_id] = _as_numpy(neutral["pos"]).reshape(-1)[:3]
            if learned.get("status", False):
                learned_solved[candidate_id] = 1
                learned_position[candidate_id] = _as_numpy(learned["pos"]).reshape(-1)[:3]
            if progress_every and (candidate_id + 1) % progress_every == 0:
                print(
                    f"seed {seed}: inferred {candidate_id + 1}/{candidate_count} Ibiza epochs",
                    flush=True,
                )
    paired = (feature_accepted & neutral_solved & learned_solved).astype(np.uint8)
    arrays = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "seed": np.asarray([seed], dtype=np.int64),
        "candidate_epoch_id": epoch_ids,
        "epoch_time_gpst_like_s": epoch_times,
        "preprocessing_accepted": preprocessing_accepted,
        "feature_accepted": feature_accepted,
        "neutral_solved": neutral_solved,
        "learned_solved": learned_solved,
        "paired": paired,
        "neutral_position_ecef_m": neutral_position,
        "learned_position_ecef_m": learned_position,
        "feature_hash": feature_hash,
        "neutral_solver_input_hash": neutral_input_hash,
        "learned_solver_input_hash": learned_input_hash,
        "model_output_hash": model_output_hash,
        "model_output_offsets": observation_offsets,
        "model_weight": np.concatenate(weights) if weights else np.empty(0, dtype=np.float64),
        "model_bias_m": np.concatenate(biases) if biases else np.empty(0, dtype=np.float64),
    }
    if output_path is not None:
        _save_npz(output_path, arrays)
    report = {
        "seed": seed,
        "checkpoint_sha256": entry_hash(checkpoint_entry),
        "candidate_epochs": candidate_count,
        "preprocessing_accepted_epochs": int(np.sum(preprocessing_accepted)),
        "feature_accepted_epochs": int(np.sum(feature_accepted)),
        "feature_rejected_epochs": len(feature_rejected_ids),
        "feature_rejected_epoch_ids": feature_rejected_ids,
        "neutral_solved_epochs": int(np.sum(neutral_solved)),
        "learned_solved_epochs": int(np.sum(learned_solved)),
        "exact_paired_epoch_count": int(np.sum(paired)),
        "neutral_messages": dict(neutral_messages),
        "learned_messages": dict(learned_messages),
        "duration_seconds": time.monotonic() - started,
        "model_controls": model_controls,
        "reference_or_ground_truth_available": False,
    }
    return arrays, report


def entry_hash(entry: Mapping[str, Any]) -> str:
    return str(entry["checkpoint_sha256"])


def load_position_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def _solution_metrics(position_ecef: np.ndarray, reference_ecef: np.ndarray) -> dict[str, Any]:
    enu = enu_errors(position_ecef, reference_ecef)
    error_2d = np.linalg.norm(enu[:, :2], axis=1)
    error_3d = np.linalg.norm(enu, axis=1)
    return {
        "2d": percentile_metrics(error_2d),
        "3d": percentile_metrics(error_3d),
        "east_rms": float(np.sqrt(np.mean(enu[:, 0] ** 2))),
        "north_rms": float(np.sqrt(np.mean(enu[:, 1] ** 2))),
        "up_rms": float(np.sqrt(np.mean(enu[:, 2] ** 2))),
    }


def evaluate_seed_support(
    arrays: Mapping[str, np.ndarray],
    reference_ecef: np.ndarray,
    *,
    support_name: str,
    historical_epoch_ids: np.ndarray,
) -> dict[str, Any]:
    """Evaluate an already-frozen position artifact on a predeclared support."""

    epoch_ids = arrays["candidate_epoch_id"].astype(np.int64)
    if support_name == PRIMARY_SUPPORT:
        support_mask = np.ones(epoch_ids.size, dtype=bool)
    elif support_name == SECONDARY_SUPPORT:
        support_mask = np.isin(epoch_ids, historical_epoch_ids)
    else:
        raise ValueError(f"unknown support {support_name}")
    preprocessed = arrays["preprocessing_accepted"].astype(bool) & support_mask
    feature = arrays["feature_accepted"].astype(bool) & support_mask
    neutral = arrays["neutral_solved"].astype(bool) & feature
    learned = arrays["learned_solved"].astype(bool) & feature
    paired = neutral & learned
    paired_ids = epoch_ids[paired]
    if paired_ids.size == 0:
        raise RuntimeError(f"{support_name} has no exact paired epochs")
    neutral_position = arrays["neutral_position_ecef_m"][paired].astype(np.float64)
    learned_position = arrays["learned_position_ecef_m"][paired].astype(np.float64)
    neutral_metrics = _solution_metrics(neutral_position, reference_ecef)
    learned_metrics = _solution_metrics(learned_position, reference_ecef)
    neutral_enu = enu_errors(neutral_position, reference_ecef)
    learned_enu = enu_errors(learned_position, reference_ecef)
    neutral_2d = np.linalg.norm(neutral_enu[:, :2], axis=1)
    neutral_3d = np.linalg.norm(neutral_enu, axis=1)
    learned_2d = np.linalg.norm(learned_enu[:, :2], axis=1)
    learned_3d = np.linalg.norm(learned_enu, axis=1)
    paired_id_hash = sha256_array(paired_ids)
    candidate_epochs = int(np.sum(support_mask))
    preprocessing_accepted = int(np.sum(preprocessed))
    feature_accepted = int(np.sum(feature))
    result = {
        "support": support_name,
        "seed": int(arrays["seed"][0]),
        "learned": learned_metrics,
        "neutral_paired": neutral_metrics,
        "paired_delta": {
            "definition": "learned error - neutral error; negative is improvement",
            "2d": paired_delta_metrics(learned_2d, neutral_2d),
            "3d": paired_delta_metrics(learned_3d, neutral_3d),
        },
        "epoch_counts": {
            "candidate_epochs": candidate_epochs,
            "preprocessing_accepted_epochs": preprocessing_accepted,
            "preprocessing_failed_epochs": candidate_epochs - preprocessing_accepted,
            "feature_accepted_epochs": feature_accepted,
            "feature_rejected_epochs": preprocessing_accepted - feature_accepted,
            "neutral_solved_epochs": int(np.sum(neutral)),
            "neutral_failed_epochs": feature_accepted - int(np.sum(neutral)),
            "learned_solved_epochs": int(np.sum(learned)),
            "learned_failed_epochs": feature_accepted - int(np.sum(learned)),
            "exact_paired_epoch_count": int(np.sum(paired)),
        },
        "pairing": {
            "neutral_epoch_ids_sha256": paired_id_hash,
            "learned_epoch_ids_sha256": paired_id_hash,
            "identical_epoch_ids": True,
            "epoch_ids": paired_ids.tolist(),
        },
        "inclusion_policy": (
            "predeclared support membership, current preprocessing status, frozen feature "
            "validity, and neutral/learned solver status only; no ground-truth error threshold"
        ),
    }
    return result


def _flatten_seed_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for solution in ("learned", "neutral_paired"):
        for dimensions in ("2d", "3d"):
            for name, value in result[solution][dimensions].items():
                flattened[f"{solution}_{dimensions}_{name}"] = float(value)
        for axis in ("east_rms", "north_rms", "up_rms"):
            flattened[f"{solution}_{axis}"] = float(result[solution][axis])
    for dimensions in ("2d", "3d"):
        for name, value in result["paired_delta"][dimensions].items():
            flattened[f"delta_{dimensions}_{name}"] = float(value)
    for name, value in result["epoch_counts"].items():
        flattened[f"count_{name}"] = float(value)
    return flattened


def across_seed_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if [result["seed"] for result in results] != list(SEEDS):
        raise RuntimeError("Ibiza across-seed summary requires all ten ordered seed summaries")
    supports = {result["support"] for result in results}
    if len(supports) != 1:
        raise RuntimeError("across-seed input mixes support definitions")
    flattened = [_flatten_seed_metrics(result) for result in results]
    return {
        "support": results[0]["support"],
        "aggregation_unit": "ten per-seed summaries; seed x epoch observations are not pooled",
        "seed_count": len(results),
        "metrics": {
            name: distribution_metrics(row[name] for row in flattened)
            for name in flattened[0]
        },
        "epoch_counts_by_seed": [result["epoch_counts"] for result in results],
    }


def inference_fingerprints(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {name: sha256_array(np.asarray(arrays[name])) for name in POSITION_ARRAY_KEYS}


def reference_perturbation_audit(
    arrays: Mapping[str, np.ndarray],
    reference_ecef: np.ndarray,
    historical_epoch_ids: np.ndarray,
) -> dict[str, Any]:
    """Prove that changing GT can change metrics but not frozen inference arrays."""

    before = inference_fingerprints(arrays)
    original = evaluate_seed_support(
        arrays,
        reference_ecef,
        support_name=PRIMARY_SUPPORT,
        historical_epoch_ids=historical_epoch_ids,
    )
    perturbed_reference = np.asarray(reference_ecef, dtype=np.float64) + np.array(
        [37.0, -19.0, 11.0], dtype=np.float64
    )
    perturbed = evaluate_seed_support(
        arrays,
        perturbed_reference,
        support_name=PRIMARY_SUPPORT,
        historical_epoch_ids=historical_epoch_ids,
    )
    after = inference_fingerprints(arrays)
    unchanged = {name: before[name] == after[name] for name in before}
    metrics_changed = original["learned"] != perturbed["learned"]
    if not all(unchanged.values()) or not metrics_changed:
        raise RuntimeError("ground-truth perturbation leakage audit failed")
    return {
        "perturbation_ecef_m": [37.0, -19.0, 11.0],
        "feature_hashes_unchanged": unchanged["feature_hash"],
        "neutral_solver_input_hashes_unchanged": unchanged["neutral_solver_input_hash"],
        "learned_solver_input_hashes_unchanged": unchanged["learned_solver_input_hash"],
        "model_outputs_unchanged": unchanged["model_output_hash"],
        "neutral_estimated_ecef_unchanged": unchanged["neutral_position_ecef_m"],
        "learned_estimated_ecef_unchanged": unchanged["learned_position_ecef_m"],
        "evaluation_metrics_changed": metrics_changed,
        "only_evaluation_metrics_changed": all(unchanged.values()) and metrics_changed,
    }


def _ensure_output_policy(output_dir: Path, *, overwrite: bool) -> None:
    protected = (
        output_dir / "ibiza_evaluation_manifest.json",
        output_dir / "per_seed_metrics.json",
        output_dir / "across_seed_summary.json",
    )
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"evaluation outputs already exist; pass --overwrite explicitly: {existing}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if args.smoke_epochs is None and args.seed is not None:
        raise ValueError("a scientific full run cannot select one seed")
    if args.smoke_epochs is not None and args.seed is None:
        raise ValueError("a smoke run requires exactly one --seed")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        if torch.get_num_interop_threads() != args.threads:
            raise

    output_dir = args.output_dir.resolve()
    _ensure_output_policy(output_dir, overwrite=args.overwrite)
    provenance_before = verify_provenance()
    checkpoint_root = args.checkpoint_root.resolve()
    manifest_path = (
        args.checkpoint_manifest.resolve()
        if args.checkpoint_manifest is not None
        else checkpoint_root / "frozen_checkpoint_manifest.json"
    )
    inventory = load_checkpoint_inventory(checkpoint_root, manifest_path)
    historical_ids, historical_times = historical_common_epoch_ids(args.historical_dataset)
    input_hashes_before = {
        "observation": sha256_file(args.observation),
        "navigation": sha256_file(args.navigation),
        "historical_dataset": sha256_file(args.historical_dataset),
        "neutral_audit": sha256_file(args.neutral_audit),
        "checkpoint_manifest": inventory["manifest_sha256"],
    }
    records, preprocess_manifest = load_or_preprocess_ibiza(
        output_dir,
        args.observation,
        args.navigation,
        args.neutral_audit,
        smoke_epochs=args.smoke_epochs,
        progress_every=args.progress_every,
        force=args.force_preprocess,
    )
    candidate_count = int(preprocess_manifest["evaluated_candidate_epochs"])
    record_times = {
        int(record["candidate_epoch_id"]): float(record["epoch_time_gpst_like_s"])
        for record in records
    }
    historical_in_run = historical_ids[historical_ids < candidate_count]
    historical_time_lookup = dict(zip(historical_ids.tolist(), historical_times.tolist(), strict=True))
    for epoch_id in historical_in_run:
        if epoch_id in record_times and record_times[int(epoch_id)] != historical_time_lookup[int(epoch_id)]:
            raise RuntimeError("historical-common timestamp differs from current-stack timestamp")

    selected_seeds = list(SEEDS) if args.smoke_epochs is None else [args.seed]
    position_paths: dict[int, Path] = {}
    inference_reports: list[dict[str, Any]] = []
    for seed in selected_seeds:
        entry = inventory["seeds"][int(seed)]
        position_path = output_dir / "positions_pre_ground_truth" / f"seed_{seed}.npz"
        if position_path.exists() and not args.overwrite:
            raise FileExistsError(position_path)
        _arrays, report = infer_seed(
            records,
            candidate_count=candidate_count,
            checkpoint_entry=entry,
            output_path=position_path,
            progress_every=args.progress_every,
        )
        position_paths[int(seed)] = position_path
        inference_reports.append(
            {
                **report,
                "position_artifact": str(position_path),
                "position_artifact_sha256": sha256_file(position_path),
            }
        )

    # Leakage boundary: every requested position artifact exists before this
    # first read of the EPN reference document.
    reference_document = json.loads(args.reference.read_text(encoding="utf-8"))
    reference_ecef = trusted_reference_ecef(reference_document)
    per_support: dict[str, list[dict[str, Any]]] = {name: [] for name in SUPPORT_NAMES}
    arrays_by_seed: dict[int, dict[str, np.ndarray]] = {}
    for seed in selected_seeds:
        arrays = load_position_arrays(position_paths[int(seed)])
        arrays_by_seed[int(seed)] = arrays
        for support_name in SUPPORT_NAMES:
            per_support[support_name].append(
                evaluate_seed_support(
                    arrays,
                    reference_ecef,
                    support_name=support_name,
                    historical_epoch_ids=historical_ids,
                )
            )
    leakage = reference_perturbation_audit(
        arrays_by_seed[int(selected_seeds[0])], reference_ecef, historical_ids
    )
    per_seed_document = {
        "schema_version": 1,
        "supports": per_support,
    }
    write_json(output_dir / "per_seed_metrics.json", per_seed_document)
    if selected_seeds == list(SEEDS):
        across_document = {
            "schema_version": 1,
            "supports": {
                name: across_seed_summary(per_support[name]) for name in SUPPORT_NAMES
            },
        }
    else:
        across_document = {
            "schema_version": 1,
            "status": "not_applicable_to_single_seed_non_scientific_smoke",
            "supports": {},
        }
    write_json(output_dir / "across_seed_summary.json", across_document)

    assert_checkpoint_inventory_unchanged(inventory)
    input_hashes_after = {
        "observation": sha256_file(args.observation),
        "navigation": sha256_file(args.navigation),
        "historical_dataset": sha256_file(args.historical_dataset),
        "neutral_audit": sha256_file(args.neutral_audit),
        "checkpoint_manifest": sha256_file(manifest_path),
    }
    provenance_after = verify_provenance()
    external_unchanged = provenance_before["upstream"] == provenance_after["upstream"]
    full_run = args.smoke_epochs is None and selected_seeds == list(SEEDS)
    manifest = {
        "schema_version": 1,
        "status": "complete" if full_run else "non_scientific_smoke_complete",
        "scientific_question": (
            "zero-shot frozen KLT3 HybridShareSysNet plus TASGNSS versus neutral TASGNSS on Ibiza"
        ),
        "evaluated_seeds": selected_seeds,
        "all_ten_seeds_evaluated": full_run,
        "checkpoint_selection": False,
        "training_updates": False,
        "optimizer_created": False,
        "normalization_refit": False,
        "ibiza_assisted_calibration": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "model_output_semantics": {
            "weight": "per-observation w_i; TASGNSS forms W=diag(w)",
            "bias": "positive b_i in residual = corrected pseudorange - predicted pseudorange - b",
            "learned_solver_call": (
                "tasgnss.wls_pnt_pos(..., use_cache=True, w=weight, b=bias, "
                "enable_torch=True, device='cpu')"
            ),
            "neutral": "w=1 identity and b=None, which TASGNSS constructs as an all-zero bias vector",
        },
        "support_policy": {
            PRIMARY_SUPPORT: (
                "all 2,880 raw Ibiza epochs, followed only by current preprocessing status, frozen "
                "feature validity, and exact neutral/learned solved intersection"
            ),
            SECONDARY_SUPPORT: (
                "the previously frozen 2,856 historical-common epoch IDs, followed by the same "
                "current feature and exact-pairing rules"
            ),
            "ground_truth_error_threshold": None,
            "support_decided_before_ground_truth": True,
        },
        "preprocessing": preprocess_manifest,
        "checkpoint_inventory": {
            "manifest_path": str(inventory["manifest_path"]),
            "manifest_sha256": inventory["manifest_sha256"],
            "verified_checkpoint_count": inventory["seed_count"],
            "checkpoint_hashes": {
                str(entry["seed"]): entry["checkpoint_sha256"] for entry in inventory["seeds"]
            },
            "hashes_unchanged_after_evaluation": True,
        },
        "inference": inference_reports,
        "ground_truth_boundary": {
            "reference_path": str(args.reference.resolve()),
            "reference_sha256": sha256_file(args.reference),
            "reference_opened_after_all_positions_serialized": True,
            "reference_passed_to_preprocessing_model_or_solver": False,
            "perturbation_audit": leakage,
        },
        "controls": {
            "input_hashes_unchanged": input_hashes_after == input_hashes_before,
            "external_repositories_unchanged": external_unchanged,
            "paired_epoch_ids_required_identical": True,
            "across_seed_aggregation_unit": "ten per-seed summaries; no seed x epoch pooling",
            "strong_conventional_rtklib_spp_included": False,
        },
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "provenance_before": provenance_before,
        "provenance_after": provenance_after,
        "artifacts": {
            "per_seed_metrics": "per_seed_metrics.json",
            "across_seed_summary": "across_seed_summary.json",
            "positions_pre_ground_truth": {
                str(seed): {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for seed, path in position_paths.items()
            },
        },
    }
    write_json(output_dir / "ibiza_evaluation_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(output_dir),
                "evaluated_seeds": selected_seeds,
                "candidate_epochs": candidate_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
