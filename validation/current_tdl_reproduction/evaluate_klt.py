#!/usr/bin/env python3
"""Freeze ten KLT3 checkpoints and evaluate every seed on current KLT1/KLT2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .core import (
    DATASET_SPECS,
    DEFAULT_OUTPUT_ROOT,
    FEATURE_NAMES,
    SEEDS,
    distribution_metrics,
    feature_tensor,
    import_current_stack,
    initialize_solver_cache,
    load_or_preprocess_dataset,
    make_model,
    model_tensor_shapes,
    paired_delta_metrics,
    percentile_metrics,
    position_enu,
    record_feature_parts,
    sha256_file,
    verify_provenance,
    write_json,
)


LITERATURE_REFERENCE = {
    "classification": "external literature sanity reference; not a regression target",
    "citation": (
        "Yin et al., Residual-Guided Hybrid Stochastic Modeling: A Two-Stage "
        "Learning Framework for Urban GNSS Positioning Enhancement, Sensors 2026, 26, 5622"
    ),
    "KLT1": {
        "2d": {"mean": 2.75, "max": 6.71, "median": 2.51, "p95": 5.54},
        "3d": {"mean": 4.95, "max": 19.19, "median": 4.29, "p95": 11.37},
    },
    "KLT2": {
        "2d": {"mean": 3.05, "max": 8.52, "median": 2.88, "p95": 5.80},
        "3d": {"mean": 5.09, "max": 11.03, "median": 4.82, "p95": 8.64},
    },
    "restriction": (
        "Do not tune to these values or claim exact reproduction without identical checkpoint, "
        "software, preprocessing, solver, and evaluation semantics."
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--freeze-only", action="store_true")
    return parser.parse_args(argv)


def _state_shapes(state: Mapping[str, torch.Tensor]) -> dict[str, list[int]]:
    return {name: list(value.shape) for name, value in state.items()}


def freeze_checkpoints(output_root: Path) -> dict[str, object]:
    """Validate and hash every seed before any held-out data is opened."""

    provenance = verify_provenance()
    seeds: list[dict[str, object]] = []
    reference_shapes: dict[str, list[int]] | None = None
    for seed in SEEDS:
        directory = output_root / f"seed_{seed}"
        checkpoint = directory / "multinet_3d.pth"
        configuration_path = directory / "training_configuration.json"
        scaler_path = directory / "scaler.json"
        provenance_path = directory / "provenance.json"
        for path in (checkpoint, configuration_path, scaler_path, provenance_path):
            if not path.is_file():
                raise FileNotFoundError(f"seed {seed} freeze gate is missing {path}")
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
        seed_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if configuration.get("seed") != seed or configuration.get("epochs") != 120:
            raise RuntimeError(f"seed {seed} is not a completed 120-epoch run")
        if configuration.get("training_datasets") != ["KLT3"]:
            raise RuntimeError(f"seed {seed} training data boundary is invalid")
        if configuration.get("held_out_datasets") != ["KLT1", "KLT2"]:
            raise RuntimeError(f"seed {seed} held-out declaration is invalid")
        if configuration.get("ibiza_used") is not False:
            raise RuntimeError(f"seed {seed} does not prove Ibiza exclusion")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        shapes = _state_shapes(state)
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            raise RuntimeError(f"seed {seed} architecture tensor shapes differ")
        mean = state["seq.0.mean"].detach().cpu().numpy()
        std = state["seq.0.std"].detach().cpu().numpy()
        if mean.shape != (9,) or std.shape != (9,):
            raise RuntimeError(f"seed {seed} checkpoint scaler is not 9-dimensional")
        np.testing.assert_array_equal(
            mean, np.asarray(scaler["checkpoint_effective_mean_float32_then_float64"])
        )
        np.testing.assert_array_equal(
            std, np.asarray(scaler["checkpoint_effective_std_float32_then_float64"])
        )
        if seed_provenance["upstream"] != provenance["upstream"]:
            raise RuntimeError(f"seed {seed} upstream provenance differs from evaluation")
        seeds.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "tensor_shapes": shapes,
                "scaler_mean": mean.tolist(),
                "scaler_std": std.tolist(),
                "training_configuration": configuration,
                "upstream_revisions": {
                    name: record["head"] for name, record in provenance["upstream"].items()
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_held_out_evaluation",
        "selection_policy": "all ten seeds; no best-seed selection",
        "seed_count": len(seeds),
        "seeds": seeds,
        "provenance": provenance,
    }
    write_json(output_root / "frozen_checkpoint_manifest.json", manifest)
    return manifest


def _reset_one_solver_cache(tas: Any, record: Mapping[str, Any]) -> None:
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


def _write_epoch_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        raise RuntimeError("cannot write an empty epoch evaluation")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _position_errors(record: Mapping[str, Any], position: torch.Tensor) -> np.ndarray:
    return position_enu(record, position).detach().cpu().numpy().astype(np.float64)


def evaluate_seed_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    checkpoint: Path,
    seed: int,
    dataset: str,
    output_dir: Path,
) -> dict[str, object]:
    tas, _core = import_current_stack()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = make_model(np.zeros(9), np.ones(9))
    model.load_state_dict(state)
    model.eval()
    initialize_solver_cache(tas, records)
    rows: list[dict[str, object]] = []
    learned_enu: list[np.ndarray] = []
    neutral_enu: list[np.ndarray] = []
    failed = 0
    feature_rejected = 0

    with torch.no_grad():
        for record in records:
            parts = record_feature_parts(record)
            if parts is None:
                feature_rejected += 1
                continue
            inputs = feature_tensor(parts)
            weight, bias = model(inputs)

            # Both branches start from the same frozen neutral state and use
            # identical current TASGNSS preprocessing/support.
            _reset_one_solver_cache(tas, record)
            neutral = tas.wls_pnt_pos(
                record, None, use_cache=True, w=1, b=None, enable_torch=False, device="cpu"
            )
            _reset_one_solver_cache(tas, record)
            learned = tas.wls_pnt_pos(
                record,
                None,
                use_cache=True,
                w=weight,
                b=bias,
                enable_torch=True,
                device="cpu",
            )
            if not neutral.get("status", False) or not learned.get("status", False):
                failed += 1
                continue

            # Ground truth is first consulted here, after both positions exist.
            candidate_error = _position_errors(record, learned["pos"])
            baseline_position = torch.tensor(neutral["pos"], dtype=torch.float64)
            baseline_error = _position_errors(record, baseline_position)
            learned_enu.append(candidate_error)
            neutral_enu.append(baseline_error)
            candidate_2d = float(np.linalg.norm(candidate_error[:2]))
            candidate_3d = float(np.linalg.norm(candidate_error))
            baseline_2d = float(np.linalg.norm(baseline_error[:2]))
            baseline_3d = float(np.linalg.norm(baseline_error))
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "candidate_index": int(record["candidate_index"]),
                    "epoch_time_gpst_like": float(record["epoch_time_gpst_like"]),
                    "observation_count": int(inputs.shape[0]),
                    "learned_east_m": float(candidate_error[0]),
                    "learned_north_m": float(candidate_error[1]),
                    "learned_up_m": float(candidate_error[2]),
                    "learned_2d_m": candidate_2d,
                    "learned_3d_m": candidate_3d,
                    "neutral_east_m": float(baseline_error[0]),
                    "neutral_north_m": float(baseline_error[1]),
                    "neutral_up_m": float(baseline_error[2]),
                    "neutral_2d_m": baseline_2d,
                    "neutral_3d_m": baseline_3d,
                    "delta_2d_m": candidate_2d - baseline_2d,
                    "delta_3d_m": candidate_3d - baseline_3d,
                }
            )
    if not rows:
        raise RuntimeError(f"{dataset} seed {seed} has no paired solved epochs")
    learned_array = np.vstack(learned_enu)
    neutral_array = np.vstack(neutral_enu)
    learned_2d = np.linalg.norm(learned_array[:, :2], axis=1)
    learned_3d = np.linalg.norm(learned_array, axis=1)
    neutral_2d = np.linalg.norm(neutral_array[:, :2], axis=1)
    neutral_3d = np.linalg.norm(neutral_array, axis=1)
    total = len(records)
    matched = len(rows)
    if matched + failed + feature_rejected != total:
        raise RuntimeError("exact epoch reconciliation failed")
    result = {
        "dataset": dataset,
        "seed": seed,
        "checkpoint_sha256": sha256_file(checkpoint),
        "learned": {
            "2d": percentile_metrics(learned_2d),
            "3d": percentile_metrics(learned_3d),
            "east_rms": float(np.sqrt(np.mean(learned_array[:, 0] ** 2))),
            "north_rms": float(np.sqrt(np.mean(learned_array[:, 1] ** 2))),
            "up_rms": float(np.sqrt(np.mean(learned_array[:, 2] ** 2))),
        },
        "neutral_paired": {
            "2d": percentile_metrics(neutral_2d),
            "3d": percentile_metrics(neutral_3d),
            "east_rms": float(np.sqrt(np.mean(neutral_array[:, 0] ** 2))),
            "north_rms": float(np.sqrt(np.mean(neutral_array[:, 1] ** 2))),
            "up_rms": float(np.sqrt(np.mean(neutral_array[:, 2] ** 2))),
        },
        "paired_delta": {
            "definition": "trained TDL error - neutral TASGNSS error; negative is improvement",
            "2d": paired_delta_metrics(learned_2d, neutral_2d),
            "3d": paired_delta_metrics(learned_3d, neutral_3d),
        },
        "epoch_counts": {
            "preprocessed_epochs": total,
            "solved_epochs": matched,
            "failed_epochs": failed,
            "feature_rejected_epochs": feature_rejected,
            "exact_matched_epoch_count": matched,
            "reconciled": matched + failed + feature_rejected == total,
        },
    }
    seed_dir = output_dir / dataset.lower() / f"seed_{seed}"
    _write_epoch_csv(seed_dir / "paired_epoch_results.csv", rows)
    write_json(seed_dir / "metrics.json", result)
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
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[f"count_{name}"] = float(value)
    return flattened


def across_seed_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if [result["seed"] for result in results] != list(SEEDS):
        raise RuntimeError("across-seed summary requires all ten ordered seeds")
    flattened = [_flatten_seed_metrics(result) for result in results]
    names = flattened[0].keys()
    return {
        "aggregation_unit": "ten per-seed summaries; epoch×seed rows are not pooled",
        "seed_count": len(results),
        "metrics": {
            name: distribution_metrics(row[name] for row in flattened) for name in names
        },
        "epoch_counts_by_seed": [result["epoch_counts"] for result in results],
    }


def _assert_frozen_unchanged(freeze_manifest: Mapping[str, Any]) -> None:
    for item in freeze_manifest["seeds"]:
        path = Path(item["checkpoint"])
        actual = sha256_file(path)
        if actual != item["checkpoint_sha256"]:
            raise RuntimeError(f"evaluation changed frozen checkpoint {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        if torch.get_num_interop_threads() != args.threads:
            raise
    output_root = args.output_root.resolve()
    freeze_manifest = freeze_checkpoints(output_root)
    if args.freeze_only:
        print(json.dumps(freeze_manifest, indent=2, sort_keys=True))
        return 0

    evaluation_root = output_root / "evaluation"
    all_results: dict[str, list[dict[str, object]]] = {}
    preprocess_manifests: dict[str, object] = {}
    # Held-out data is opened only after all ten checkpoint hashes are frozen.
    for dataset in ("KLT1", "KLT2"):
        records, preprocess_manifest = load_or_preprocess_dataset(dataset, output_root)
        preprocess_manifests[dataset] = preprocess_manifest
        results = []
        for seed in SEEDS:
            checkpoint = output_root / f"seed_{seed}/multinet_3d.pth"
            results.append(
                evaluate_seed_dataset(
                    records,
                    checkpoint=checkpoint,
                    seed=seed,
                    dataset=dataset,
                    output_dir=evaluation_root,
                )
            )
        all_results[dataset] = results
        write_json(evaluation_root / dataset.lower() / "per_seed_metrics.json", results)
        write_json(
            evaluation_root / dataset.lower() / "across_seed_summary.json",
            across_seed_summary(results),
        )
    _assert_frozen_unchanged(freeze_manifest)
    final_manifest = {
        "status": "complete",
        "evaluated_seeds": list(SEEDS),
        "datasets": ["KLT1", "KLT2"],
        "training_updates": False,
        "normalization_refit": False,
        "checkpoint_selection": False,
        "all_ten_seeds_evaluated": all(len(all_results[name]) == 10 for name in all_results),
        "checkpoint_hashes_unchanged_after_evaluation": True,
        "preprocessing": preprocess_manifests,
        "literature_reference": LITERATURE_REFERENCE,
        "feature_names": FEATURE_NAMES,
        "current_dataset_cardinalities": {
            name: DATASET_SPECS[name].expected_epochs for name in ("KLT1", "KLT2")
        },
    }
    write_json(evaluation_root / "evaluation_manifest.json", final_manifest)
    print(json.dumps(final_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
