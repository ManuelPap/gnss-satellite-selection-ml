"""Deterministic full-day batch export for the 30 frozen Ibiza models.

This is orchestration only.  It loads the already frozen NPZ once, delegates
every scientific calculation to :mod:`validation.ibiza_generalization.inference`,
and writes one JSON Lines result per architecture/seed.  An exception for one
epoch becomes an explicit result row; it never removes that epoch from the
output or changes the common epoch sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from validation.ibiza_generalization.checkpoints import (
    ARCHITECTURES,
    DEFAULT_MANIFEST as CHECKPOINT_MANIFEST,
    REPOSITORY_ROOT,
    CheckpointRecord,
    load_checkpoint_manifest,
    load_frozen_model,
    sha256_file,
    validate_checkpoint_inventory,
)
from validation.ibiza_generalization.inference import (
    DEFAULT_IBIZA_NPZ,
    IBIZA_NPZ_SHA256,
    EpochInference,
    evaluate_epoch,
    load_ibiza_dataset,
)
from validation.ibiza_generalization.runtime_cache import (
    PYRTKLIB_COMMIT,
    PYRTKLIB_VERSION,
    TDL_COMMIT,
)


BATCH_SPEC_PATH = Path(__file__).with_name("full_day_batch_manifest.json")
DEFAULT_RESULTS_DIR = (
    REPOSITORY_ROOT.parent
    / "external_data/ibiza_2025_01_01/results/frozen_tdl"
)
RESULT_SCHEMA_VERSION = 1
RUN_MANIFEST_FILENAME = "run_manifest.json"
LOW_WEIGHT_THRESHOLD = 0.01


def _display_path(path: Path) -> str:
    """Render a path relative to the repository without personal prefixes."""

    return Path(os.path.relpath(path.resolve(), REPOSITORY_ROOT)).as_posix()


def load_batch_spec(path: Path = BATCH_SPEC_PATH) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if "/home/" in text:
        raise RuntimeError("tracked full-day manifest contains an absolute /home path")
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError("unsupported tracked full-day batch manifest schema")
    dataset = value.get("source_dataset")
    checkpoints = value.get("checkpoint_inventory")
    if not isinstance(dataset, dict) or dataset.get("sha256") != IBIZA_NPZ_SHA256:
        raise RuntimeError("tracked full-day manifest has the wrong Ibiza NPZ hash")
    if not isinstance(checkpoints, dict):
        raise RuntimeError("tracked full-day manifest lacks checkpoint inventory")
    actual_checkpoint_manifest_hash = sha256_file(CHECKPOINT_MANIFEST)
    if checkpoints.get("manifest_sha256") != actual_checkpoint_manifest_hash:
        raise RuntimeError(
            "tracked full-day manifest checkpoint-manifest hash is stale"
        )
    return value


def discover_full_plan(
    manifest_path: Path = CHECKPOINT_MANIFEST,
) -> tuple[CheckpointRecord, ...]:
    """Return all 30 records exactly once in architecture/seed order."""

    records = load_checkpoint_manifest(manifest_path)
    by_identity: dict[tuple[str, int], CheckpointRecord] = {}
    for record in records:
        identity = (record.architecture, record.seed)
        if identity in by_identity:
            raise RuntimeError(f"duplicate checkpoint in full-day plan: {identity}")
        by_identity[identity] = record
    expected = {
        (architecture, seed) for architecture in ARCHITECTURES for seed in range(10)
    }
    if set(by_identity) != expected or len(by_identity) != 30:
        raise RuntimeError("full-day plan must contain exactly 30 architecture/seeds")
    return tuple(
        by_identity[(architecture, seed)]
        for architecture in ARCHITECTURES
        for seed in range(10)
    )


def reject_ground_truth_arrays(dataset: Mapping[str, np.ndarray]) -> None:
    prohibited_tokens = (
        "ground_truth",
        "reference_coordinate",
        "trusted_coordinate",
        "position_accuracy",
    )
    found = sorted(
        name
        for name in dataset
        if any(token in name.lower() for token in prohibited_tokens)
    )
    if found:
        raise RuntimeError(
            "full-day frozen inference rejects ground-truth/evaluation arrays: "
            + ", ".join(found)
        )


def _state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_model_unchanged(
    before: Mapping[str, torch.Tensor], model: nn.Module
) -> None:
    after = model.state_dict()
    if tuple(before) != tuple(after) or any(
        not torch.equal(before[name], after[name]) for name in before
    ):
        raise RuntimeError("frozen checkpoint parameters changed during batch inference")
    if model.training:
        raise RuntimeError("frozen model left eval mode during batch inference")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen model accumulated gradients during batch inference")


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _finite_vector_or_none(value: torch.Tensor) -> list[float] | None:
    flat = value.detach().cpu().reshape(-1)
    if not bool(torch.all(torch.isfinite(flat))):
        return None
    return [float(item) for item in flat]


def _output_statistics(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().cpu().to(torch.float64).reshape(-1)
    if flat.numel() == 0 or not bool(torch.all(torch.isfinite(flat))):
        raise RuntimeError("model-output diagnostics require finite non-empty values")
    return {
        "mean": float(torch.mean(flat)),
        "std": float(torch.std(flat, correction=0)),
        "min": float(torch.min(flat)),
        "max": float(torch.max(flat)),
    }


def _fraction(mask: torch.Tensor) -> float:
    flat = mask.detach().cpu().reshape(-1)
    if flat.numel() == 0:
        raise RuntimeError("cannot calculate a fraction over zero model outputs")
    return float(torch.count_nonzero(flat)) / flat.numel()


def _model_output_record(epoch: EpochInference) -> dict[str, object]:
    bias = epoch.neural_outputs.predicted_bias_m
    weight = epoch.neural_outputs.predicted_weight
    return {
        "bias_m": _output_statistics(bias) if bias is not None else None,
        "weight": _output_statistics(weight) if weight is not None else None,
        "weight_fraction_lt_0_01": (
            _fraction(weight < LOW_WEIGHT_THRESHOLD) if weight is not None else None
        ),
        "bias_fraction_eq_0": (
            _fraction(bias == 0.0) if epoch.architecture == "TDL-BW" else None
        ),
        "bias_fraction_gt_0": (
            _fraction(bias > 0.0) if epoch.architecture == "TDL-BW" else None
        ),
    }


def _record_prefix(
    record: CheckpointRecord,
    dataset: Mapping[str, np.ndarray],
    accepted_epoch_index: int,
) -> dict[str, object]:
    start = int(dataset["epoch_offsets"][accepted_epoch_index])
    stop = int(dataset["epoch_offsets"][accepted_epoch_index + 1])
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "architecture": record.architecture,
        "seed": record.seed,
        "checkpoint_sha256": record.sha256,
        "source_ibiza_npz_sha256": IBIZA_NPZ_SHA256,
        "accepted_epoch_index": accepted_epoch_index,
        "source_split_epoch_index": int(
            dataset["epoch_split_index"][accepted_epoch_index]
        ),
        "timestamp_gpst_like_s": float(
            dataset["epoch_time_gpst_like_s"][accepted_epoch_index]
        ),
        "row_start": start,
        "row_stop": stop,
        "satellite_count": stop - start,
        "active_state_dimension": int(
            dataset["epoch_active_state_count"][accepted_epoch_index]
        ),
    }


def successful_epoch_record(
    record: CheckpointRecord,
    dataset: Mapping[str, np.ndarray],
    epoch: EpochInference,
) -> dict[str, object]:
    result = _record_prefix(record, dataset, epoch.accepted_epoch_index)
    if epoch.row_start != result["row_start"] or epoch.row_stop != result["row_stop"]:
        raise RuntimeError("evaluator epoch row boundaries differ from the source NPZ")
    if epoch.wls.active_state_count != result["active_state_dimension"]:
        raise RuntimeError("WLS active-state dimension differs from the source NPZ")
    full_state = _finite_vector_or_none(epoch.receiver_state)
    solution_status = (
        epoch.wls.solution_status if full_state is not None else "diagnostic_failure"
    )
    result.update(
        {
            "estimated_ecef_m": full_state[:3] if full_state is not None else None,
            "estimated_receiver_state": full_state,
            "rank": epoch.wls.final_rank,
            "condition_number": _finite_or_none(
                epoch.wls.final_condition_number
            ),
            "iteration_count": len(epoch.wls.iterations),
            "solution_status": solution_status,
            "convergence_status": epoch.wls.convergence_status,
            "rank_status": epoch.wls.rank_status,
            "conditioning_status": epoch.wls.conditioning_status,
            "model_output": _model_output_record(epoch),
            "error": None,
        }
    )
    return result


def failed_epoch_record(
    record: CheckpointRecord,
    dataset: Mapping[str, np.ndarray],
    accepted_epoch_index: int,
    error: Exception,
) -> dict[str, object]:
    result = _record_prefix(record, dataset, accepted_epoch_index)
    result.update(
        {
            "estimated_ecef_m": None,
            "estimated_receiver_state": None,
            "rank": None,
            "condition_number": None,
            "iteration_count": None,
            "solution_status": "exception",
            "convergence_status": "not_available",
            "rank_status": "not_available",
            "conditioning_status": "not_available",
            "model_output": {
                "bias_m": None,
                "weight": None,
                "weight_fraction_lt_0_01": None,
                "bias_fraction_eq_0": None,
                "bias_fraction_gt_0": None,
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    )
    return result


def result_filename(record: CheckpointRecord) -> str:
    slug = {"TDL-B": "tdl_b", "TDL-W": "tdl_w", "TDL-BW": "tdl_bw"}[
        record.architecture
    ]
    return f"{slug}_seed_{record.seed}.jsonl"


def _json_line(value: Mapping[str, object]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"


def validate_result_file(
    path: Path,
    record: CheckpointRecord,
    dataset: Mapping[str, np.ndarray],
    accepted_epoch_indices: Sequence[int],
) -> dict[str, object]:
    """Independently check one result file's identities and row accounting."""

    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise RuntimeError(f"result line {line_number} lacks a newline")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"result line {line_number} is not an object")
            rows.append(value)
    expected_indices = tuple(int(value) for value in accepted_epoch_indices)
    actual_indices = tuple(int(row["accepted_epoch_index"]) for row in rows)
    if actual_indices != expected_indices:
        raise RuntimeError("result accepted-epoch sequence differs from the run policy")

    expected_satellite_rows = 0
    failed_count = 0
    status_counts: dict[str, int] = {}
    for row in rows:
        epoch_index = int(row["accepted_epoch_index"])
        start = int(dataset["epoch_offsets"][epoch_index])
        stop = int(dataset["epoch_offsets"][epoch_index + 1])
        satellite_count = int(row["satellite_count"])
        if row["architecture"] != record.architecture or int(row["seed"]) != record.seed:
            raise RuntimeError("result row architecture/seed differs from its file")
        if row["checkpoint_sha256"] != record.sha256:
            raise RuntimeError("result row checkpoint hash differs from its job")
        if row["source_ibiza_npz_sha256"] != IBIZA_NPZ_SHA256:
            raise RuntimeError("result row uses the wrong Ibiza NPZ hash")
        if int(row["row_start"]) != start or int(row["row_stop"]) != stop:
            raise RuntimeError("result row boundaries differ from the source NPZ")
        if satellite_count != stop - start:
            raise RuntimeError("result satellite count differs from row boundaries")
        if int(row["active_state_dimension"]) != int(
            dataset["epoch_active_state_count"][epoch_index]
        ):
            raise RuntimeError("result active-state dimension differs from the NPZ")
        expected_satellite_rows += satellite_count
        status = str(row["solution_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "solved":
            failed_count += 1
        if status == "exception":
            if row["error"] is None or row["estimated_receiver_state"] is not None:
                raise RuntimeError("exception result does not retain explicit failure data")
        elif row["error"] is not None:
            raise RuntimeError("non-exception result unexpectedly records an error")

    return {
        "record_count": len(rows),
        "satellite_row_count": expected_satellite_rows,
        "failed_epoch_count": failed_count,
        "solution_status_counts": dict(sorted(status_counts.items())),
    }


def _runtime_versions() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "pymap3d": importlib.metadata.version("pymap3d"),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "device": "cpu",
        "model_and_wls_dtype": "torch.float64",
        "tdl_gnss_reference_commit": TDL_COMMIT,
        "pyrtklib_reference_commit": PYRTKLIB_COMMIT,
        "pyrtklib_reference_version": PYRTKLIB_VERSION,
        "paper_runtime_imported_during_batch": False,
    }


def _epoch_policy(
    indices: Sequence[int], *, full_day: bool
) -> dict[str, object]:
    index_array = np.asarray(indices, dtype="<i8")
    return {
        "name": (
            "all_preprocessed_accepted_epochs_chronological"
            if full_day
            else "explicit_validation_subset"
        ),
        "accepted_epoch_count": len(indices),
        "first_accepted_epoch_index": int(indices[0]) if indices else None,
        "last_accepted_epoch_index": int(indices[-1]) if indices else None,
        "accepted_epoch_indices_sha256": hashlib.sha256(
            index_array.tobytes(order="C")
        ).hexdigest(),
        "failed_epoch_policy": "retain one explicit result row; never drop",
        "ground_truth_used": False,
    }


def _write_job(
    record: CheckpointRecord,
    dataset: Mapping[str, np.ndarray],
    accepted_epoch_indices: Sequence[int],
    output_path: Path,
    *,
    progress_every: int,
) -> dict[str, object]:
    frozen = load_frozen_model(record.architecture, record.seed)
    before = _state_snapshot(frozen.model)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    summary: dict[str, object] | None = None
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for position, epoch_index in enumerate(accepted_epoch_indices, start=1):
                try:
                    evaluated = evaluate_epoch(frozen, dataset, epoch_index)
                    result = successful_epoch_record(record, dataset, evaluated)
                except Exception as error:  # Retain scientific/numerical failures.
                    result = failed_epoch_record(record, dataset, epoch_index, error)
                stream.write(_json_line(result))
                if progress_every > 0 and position % progress_every == 0:
                    print(
                        f"{record.architecture} seed {record.seed}: "
                        f"{position}/{len(accepted_epoch_indices)} epochs",
                        file=sys.stderr,
                        flush=True,
                    )
        _assert_model_unchanged(before, frozen.model)
        summary = validate_result_file(
            temporary, record, dataset, accepted_epoch_indices
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    if summary is None:  # Defensive: validation or replacement would have raised.
        raise RuntimeError("result validation did not produce a job summary")
    summary.update(
        {
            "architecture": record.architecture,
            "seed": record.seed,
            "checkpoint_sha256": record.sha256,
            "result_file": output_path.name,
            "result_file_sha256": sha256_file(output_path),
            "checkpoint_parameters_unchanged": True,
        }
    )
    return summary


def run_batch(
    *,
    output_dir: Path,
    dataset_path: Path = DEFAULT_IBIZA_NPZ,
    jobs: Sequence[CheckpointRecord] | None = None,
    accepted_epoch_indices: Sequence[int] | None = None,
    require_complete_plan: bool = True,
    overwrite: bool = False,
    progress_every: int = 250,
) -> dict[str, object]:
    """Run a full plan or an explicit validation subset and export results."""

    batch_spec = load_batch_spec()
    full_plan = discover_full_plan()
    selected_jobs = tuple(full_plan if jobs is None else jobs)
    identities = [(record.architecture, record.seed) for record in selected_jobs]
    if len(identities) != len(set(identities)):
        raise RuntimeError("selected batch jobs contain a duplicate architecture/seed")
    full_identities = [(record.architecture, record.seed) for record in full_plan]
    if require_complete_plan and identities != full_identities:
        raise RuntimeError("full-day execution requires the exact ordered 30-job plan")
    known = {(record.architecture, record.seed): record for record in full_plan}
    for record in selected_jobs:
        if known.get((record.architecture, record.seed)) != record:
            raise RuntimeError("selected batch job differs from the frozen manifest")
    if require_complete_plan:
        validate_checkpoint_inventory()

    # One verified load is intentionally shared by every selected model.
    dataset = load_ibiza_dataset(dataset_path, expected_sha256=IBIZA_NPZ_SHA256)
    reject_ground_truth_arrays(dataset)
    epoch_count = int(dataset["epoch_offsets"].size - 1)
    if accepted_epoch_indices is None:
        indices = tuple(range(epoch_count))
        full_day = True
    else:
        indices = tuple(int(value) for value in accepted_epoch_indices)
        full_day = indices == tuple(range(epoch_count))
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("accepted epoch selection must be non-empty and unique")
    if any(value < 0 or value >= epoch_count for value in indices):
        raise IndexError("accepted epoch selection is outside the Ibiza NPZ")
    if require_complete_plan and not full_day:
        raise RuntimeError("full-day execution requires every accepted Ibiza epoch")

    output_dir = output_dir.resolve()
    targets = [output_dir / result_filename(record) for record in selected_jobs]
    targets.append(output_dir / RUN_MANIFEST_FILENAME)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "result targets already exist; pass --overwrite to replace this exact set: "
            + ", ".join(path.name for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    job_summaries = [
        _write_job(
            record,
            dataset,
            indices,
            output_dir / result_filename(record),
            progress_every=progress_every,
        )
        for record in selected_jobs
    ]
    failed_count = sum(int(item["failed_epoch_count"]) for item in job_summaries)
    total_records = sum(int(item["record_count"]) for item in job_summaries)
    total_satellite_rows = sum(
        int(item["satellite_row_count"]) for item in job_summaries
    )
    run_manifest: dict[str, object] = {
        "schema_version": 1,
        "status": (
            "completed" if failed_count == 0 else "completed_with_failed_epochs"
        ),
        "source_dataset": {
            "path": _display_path(dataset_path),
            "sha256": IBIZA_NPZ_SHA256,
            "loaded_once_and_shared_by_all_jobs": True,
        },
        "checkpoint_inventory": {
            "manifest_path": _display_path(CHECKPOINT_MANIFEST),
            "manifest_sha256": sha256_file(CHECKPOINT_MANIFEST),
            "selected_job_count": len(selected_jobs),
        },
        "tracked_batch_spec": {
            "path": _display_path(BATCH_SPEC_PATH),
            "sha256": sha256_file(BATCH_SPEC_PATH),
            "declared_status": batch_spec["status"],
        },
        "runtime_versions": _runtime_versions(),
        "common_epoch_policy": _epoch_policy(indices, full_day=full_day),
        "result_format": "one deterministic JSON object per line",
        "jobs": job_summaries,
        "totals": {
            "result_record_count": total_records,
            "satellite_row_count_across_models": total_satellite_rows,
            "failed_epoch_count": failed_count,
        },
        "scientific_controls": {
            "rinex_preprocessed_again": False,
            "training_or_fine_tuning": False,
            "optimizer_created": False,
            "backward_called": False,
            "ibiza_normalization_computed": False,
            "checkpoint_parameters_updated": False,
            "ground_truth_used": False,
            "positioning_accuracy_computed": False,
            "wls_implementation_modified": False,
        },
    }
    manifest_path = output_dir / RUN_MANIFEST_FILENAME
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        temporary_manifest.write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return run_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_IBIZA_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="Required safety acknowledgement for 30 checkpoints × all epochs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace the exact 30 result files and run manifest.",
    )
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.confirm_full_run:
        raise SystemExit(
            "refusing to start the full experiment without --confirm-full-run"
        )
    manifest = run_batch(
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        require_complete_plan=True,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BATCH_SPEC_PATH",
    "DEFAULT_RESULTS_DIR",
    "LOW_WEIGHT_THRESHOLD",
    "RESULT_SCHEMA_VERSION",
    "RUN_MANIFEST_FILENAME",
    "discover_full_plan",
    "failed_epoch_record",
    "load_batch_spec",
    "reject_ground_truth_arrays",
    "result_filename",
    "run_batch",
    "successful_epoch_record",
    "validate_result_file",
]
