#!/usr/bin/env python3
"""Preprocess KLT3, run the scientific smoke test, or train frozen seeds."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .core import (
    DEFAULT_OUTPUT_ROOT,
    FEATURE_NAMES,
    FEATURE_UNITS,
    SEEDS,
    clone_state_dict,
    configure_determinism,
    environment_manifest,
    feature_tensor,
    initialize_solver_cache,
    import_current_stack,
    load_or_preprocess_dataset,
    make_model,
    model_tensor_shapes,
    paired_delta_metrics,
    position_loss,
    record_feature_parts,
    scaler_from_records,
    sha256_file,
    solver_input_fingerprint,
    state_dict_changed,
    state_dict_equal,
    training_source_contract,
    verify_provenance,
    write_json,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threads", type=int, default=1)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="prepare current KLT3 neutral cache")
    preprocess.add_argument("--force", action="store_true")

    smoke = subparsers.add_parser("smoke", help="run deterministic non-scientific smoke training")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--subset-size", type=int, default=8)
    smoke.add_argument("--epochs", type=int, default=2)

    seed = subparsers.add_parser("seed", help="train one independent KLT3 seed")
    seed.add_argument("--seed", type=int, required=True, choices=SEEDS)
    seed.add_argument("--epochs", type=int, default=120)

    all_seeds = subparsers.add_parser("all", help="train all ten project-defined seeds")
    all_seeds.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), choices=SEEDS)
    all_seeds.add_argument("--epochs", type=int, default=120)
    return parser.parse_args(argv)


def _eligible(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [record for record in records if record_feature_parts(record) is not None]


def _write_training_csv(path: Path, history: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "epoch",
        "average_loss_m",
        "summed_accepted_loss_m",
        "accepted_samples",
        "skipped_loss_samples",
        "missing_feature_samples",
        "optimizer_steps",
        "duration_seconds",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def _train(
    records: Sequence[Mapping[str, Any]],
    *,
    scaler_records: Sequence[Mapping[str, Any]],
    seed: int,
    epochs: int,
    output_dir: Path,
    threads: int,
    checkpoint_interval: int = 10,
) -> tuple[torch.nn.Module, dict[str, object]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    seed_record = configure_determinism(seed, threads=threads)
    tas, _tas_core = import_current_stack()
    mean, std = scaler_from_records(scaler_records)
    model = make_model(mean, std, device="cpu")
    initial_state = clone_state_dict(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _unused_mse = torch.nn.MSELoss(reduction="sum")
    del _unused_mse
    batch_size = 3000
    initialize_solver_cache(tas, records)
    working = list(records)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, object]] = []
    total_start = time.perf_counter()
    any_gradient = False
    for epoch_index in range(epochs):
        epoch_start = time.perf_counter()
        np.random.shuffle(working)
        accumulated_tensor: torch.Tensor | int = 0
        accumulated_value = 0.0
        total_loss_value = 0.0
        accepted = 0
        skipped_loss = 0
        missing_features = 0
        optimizer_steps = 0
        optimizer.zero_grad()

        for record in working:
            parts = record_feature_parts(record)
            if parts is None:
                missing_features += 1
                continue
            inputs = feature_tensor(parts)
            weight, bias = model(inputs)
            solution = tas.wls_pnt_pos(
                record,
                None,
                use_cache=True,
                w=weight,
                b=bias,
                enable_torch=True,
                device="cpu",
            )
            loss = position_loss(record, solution["pos"], dimensions=3, device="cpu")
            if bool(loss > 200):
                skipped_loss += 1
                continue
            accumulated_tensor = accumulated_tensor + loss
            accumulated_value += float(loss.item())
            accepted += 1
            if accepted % batch_size == 0:
                assert isinstance(accumulated_tensor, torch.Tensor)
                accumulated_tensor.backward()
                any_gradient = any(
                    parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                    for parameter in model.parameters()
                )
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steps += 1
                total_loss_value += accumulated_value
                accumulated_tensor = 0
                accumulated_value = 0.0

        if accumulated_value > 0:
            assert isinstance(accumulated_tensor, torch.Tensor)
            accumulated_tensor.backward()
            any_gradient = any_gradient or any(
                parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                for parameter in model.parameters()
            )
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1
            total_loss_value += accumulated_value

        # Preserve upstream behavior: the average divides by len(pres), not by
        # the accepted-sample count or its unused total_len variable.
        average_loss = total_loss_value / len(working) if working else 0.0
        row = {
            "epoch": epoch_index + 1,
            "average_loss_m": average_loss,
            "summed_accepted_loss_m": total_loss_value,
            "accepted_samples": accepted,
            "skipped_loss_samples": skipped_loss,
            "missing_feature_samples": missing_features,
            "optimizer_steps": optimizer_steps,
            "duration_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if checkpoint_interval and (epoch_index + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch_index + 1}.pth")

    final_checkpoint = output_dir / "multinet_3d.pth"
    torch.save(model.state_dict(), final_checkpoint)
    runtime = time.perf_counter() - total_start
    history_json = output_dir / "training_history.json"
    history_csv = output_dir / "training_history.csv"
    write_json(history_json, history)
    _write_training_csv(history_csv, history)
    scaler_path = output_dir / "scaler.json"
    effective_mean = model.state_dict()["seq.0.mean"].detach().cpu().numpy()
    effective_std = model.state_dict()["seq.0.std"].detach().cpu().numpy()
    write_json(
        scaler_path,
        {
            "feature_names": FEATURE_NAMES,
            "feature_units": FEATURE_UNITS,
            "numpy_source_mean_float64": mean.tolist(),
            "numpy_source_population_std_float64": std.tolist(),
            "checkpoint_effective_mean_float32_then_float64": effective_mean.tolist(),
            "checkpoint_effective_std_float32_then_float64": effective_std.tolist(),
        },
    )
    configuration_path = output_dir / "training_configuration.json"
    write_json(
        configuration_path,
        {
            **training_source_contract(),
            "seed": seed,
            "epochs": epochs,
            "records": len(working),
            "eligible_records": len(_eligible(working)),
            "training_datasets": ["KLT3"],
            "held_out_datasets": ["KLT1", "KLT2"],
            "ibiza_used": False,
        },
    )
    provenance_path = output_dir / "provenance.json"
    write_json(provenance_path, verify_provenance())
    environment_path = output_dir / "environment.json"
    write_json(environment_path, environment_manifest(seed_record))
    summary = {
        "seed": seed,
        "epochs": epochs,
        "runtime_seconds": runtime,
        "gradient_reached_network": any_gradient,
        "optimizer_changed_parameters": state_dict_changed(initial_state, model.state_dict()),
        "model_tensor_shapes": model_tensor_shapes(model),
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "history": history,
    }
    summary_path = output_dir / "training_summary.json"
    write_json(summary_path, summary)
    artifact_paths = sorted(
        (path for path in output_dir.iterdir() if path.name != "artifact_hashes.json"),
        key=lambda path: path.name,
    )
    write_json(
        output_dir / "artifact_hashes.json",
        {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        },
    )
    return model, summary


def train_seed(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    epochs: int,
    output_root: Path,
    threads: int,
) -> dict[str, object]:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    output_dir = output_root.resolve() / f"seed_{seed}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty scientific seed directory {output_dir}"
        )
    _model, summary = _train(
        records,
        scaler_records=records,
        seed=seed,
        epochs=epochs,
        output_dir=output_dir,
        threads=threads,
    )
    return summary


def _initialization_and_shuffle_checks(
    mean: np.ndarray, std: np.ndarray, *, seed: int, threads: int
) -> dict[str, object]:
    configure_determinism(seed, threads=threads)
    rng_before_tensor_construction = torch.random.get_rng_state().clone()
    torch.tensor(mean, dtype=torch.float32)
    torch.tensor(std, dtype=torch.float32)
    rng_after_tensor_construction = torch.random.get_rng_state().clone()
    first = make_model(mean, std)
    first_state = clone_state_dict(first)
    rng_after_model = torch.random.get_rng_state().clone()

    configure_determinism(seed, threads=threads)
    second = make_model(mean, std)
    configure_determinism(seed + 1, threads=threads)
    different = make_model(mean, std)

    def sequence(value: int) -> list[list[int]]:
        np.random.seed(value)
        result = []
        for _ in range(3):
            indices = np.arange(12)
            np.random.shuffle(indices)
            result.append(indices.tolist())
        return result

    same_sequence_a = sequence(seed)
    same_sequence_b = sequence(seed)
    return {
        "same_seed_initial_parameters_identical": state_dict_equal(first_state, second.state_dict()),
        "different_seed_initial_parameters_different": state_dict_changed(first_state, different.state_dict()),
        "same_seed_numpy_shuffle_identical": same_sequence_a == same_sequence_b,
        "tensor_construction_consumed_torch_rng": not torch.equal(
            rng_before_tensor_construction, rng_after_tensor_construction
        ),
        "model_construction_consumed_torch_rng": not torch.equal(
            rng_after_tensor_construction, rng_after_model
        ),
        "model_construction_is_first_torch_rng_consumer": torch.equal(
            rng_before_tensor_construction, rng_after_tensor_construction
        )
        and not torch.equal(rng_after_tensor_construction, rng_after_model),
        "shuffle_sequence": same_sequence_a,
    }


def run_smoke(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    subset_size: int,
    epochs: int,
    output_root: Path,
    threads: int,
) -> dict[str, object]:
    if subset_size <= 0:
        raise ValueError("subset-size must be positive")
    eligible = _eligible(records)
    if subset_size > len(eligible):
        raise ValueError("subset-size exceeds eligible current KLT3 records")
    subset = eligible[:subset_size]
    mean, std = scaler_from_records(records)
    seed_checks = _initialization_and_shuffle_checks(mean, std, seed=seed, threads=threads)
    if not all(
        seed_checks[name]
        for name in (
            "same_seed_initial_parameters_identical",
            "different_seed_initial_parameters_different",
            "same_seed_numpy_shuffle_identical",
            "model_construction_is_first_torch_rng_consumer",
        )
    ):
        raise RuntimeError(f"seed protocol failed: {seed_checks}")

    configure_determinism(seed, threads=threads)
    probe_model = make_model(mean, std)
    probe_parts = record_feature_parts(subset[0])
    assert probe_parts is not None
    probe_features = feature_tensor(probe_parts)
    probe_weight, probe_bias = probe_model(probe_features)
    if probe_features.shape[1] != 9:
        raise RuntimeError("smoke feature count is not nine")
    if not bool(torch.all((probe_weight >= 0) & (probe_weight <= 1))):
        raise RuntimeError("smoke weights are outside [0,1]")
    if not bool(torch.all(probe_bias >= 0)):
        raise RuntimeError("smoke biases are negative")

    original_fingerprint = solver_input_fingerprint(subset[0])
    changed_gt = copy.deepcopy(subset[0])
    changed_gt["gt"] = np.asarray(changed_gt["gt"]).copy()
    changed_gt["gt"][1] += 0.001
    changed_fingerprint = solver_input_fingerprint(changed_gt)
    if original_fingerprint != changed_fingerprint:
        raise RuntimeError("ground truth altered preprocessing or learned solver inputs")
    neutral_position = torch.tensor(subset[0]["gnss"]["pos"], dtype=torch.float64)
    original_gt_loss = float(position_loss(subset[0], neutral_position))
    changed_gt_loss = float(position_loss(changed_gt, neutral_position))
    if original_gt_loss == changed_gt_loss:
        raise RuntimeError("altered ground truth did not change the loss")

    smoke_root = output_root.resolve() / "smoke"
    first_dir = smoke_root / "replay_a"
    second_dir = smoke_root / "replay_b"
    first_model, first_summary = _train(
        subset,
        scaler_records=records,
        seed=seed,
        epochs=epochs,
        output_dir=first_dir,
        threads=threads,
        checkpoint_interval=0,
    )
    second_model, second_summary = _train(
        subset,
        scaler_records=records,
        seed=seed,
        epochs=epochs,
        output_dir=second_dir,
        threads=threads,
        checkpoint_interval=0,
    )
    # Wall-clock durations are intentionally excluded from the replay equality.
    first_history_no_time = [
        {key: value for key, value in row.items() if key != "duration_seconds"}
        for row in first_summary["history"]
    ]
    second_history_no_time = [
        {key: value for key, value in row.items() if key != "duration_seconds"}
        for row in second_summary["history"]
    ]
    deterministic_replay = state_dict_equal(first_model.state_dict(), second_model.state_dict()) and (
        first_history_no_time == second_history_no_time
    )
    if not deterministic_replay:
        raise RuntimeError("same-seed deterministic smoke replay failed")

    checkpoint = first_dir / "multinet_3d.pth"
    reloaded = make_model(mean, std)
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        expected_outputs = first_model(probe_features)
        reloaded_outputs = reloaded(probe_features)
    checkpoint_reload_equal = all(
        torch.equal(left, right) for left, right in zip(expected_outputs, reloaded_outputs, strict=True)
    )
    if not checkpoint_reload_equal:
        raise RuntimeError("checkpoint reload changed network outputs")

    # A direct real-epoch graph check separates gradient existence from the
    # training-loop summary and confirms TASGNSS differentiability itself.
    tas, _core = import_current_stack()
    initialize_solver_cache(tas, subset)
    graph_model = make_model(mean, std)
    graph_weight, graph_bias = graph_model(probe_features)
    graph_solution = tas.wls_pnt_pos(
        subset[0], None, use_cache=True, w=graph_weight, b=graph_bias,
        enable_torch=True, device="cpu"
    )
    graph_loss = position_loss(subset[0], graph_solution["pos"])
    graph_loss.backward()
    nonzero_gradient_parameters = sum(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in graph_model.parameters()
    )
    if nonzero_gradient_parameters == 0:
        raise RuntimeError("position loss did not reach network parameters")

    mean_epoch_seconds = float(
        np.mean([row["duration_seconds"] for row in first_summary["history"]])
    )
    eligible_full = len(eligible)
    estimated_one_seed_seconds = mean_epoch_seconds * (eligible_full / subset_size) * 120
    final_checkpoint_bytes = checkpoint.stat().st_size
    checkpoints_per_seed = 13  # epoch 10..120 plus final
    estimated_ten_seed_checkpoint_bytes = final_checkpoint_bytes * checkpoints_per_seed * 10
    preprocessing_cache = output_root.resolve() / "preprocessed/klt3_current.pkl"
    preprocessing_bytes = preprocessing_cache.stat().st_size
    report = {
        "label": "NON-SCIENTIFIC deterministic smoke test; accuracy is not a research result",
        "seed": seed,
        "subset_size": subset_size,
        "epochs": epochs,
        "full_klt3_records": len(records),
        "full_klt3_eligible_records": eligible_full,
        "feature_shape_first_epoch": list(probe_features.shape),
        "feature_names": FEATURE_NAMES,
        "feature_units": FEATURE_UNITS,
        "scaler_mean_shape": list(mean.shape),
        "scaler_std_shape": list(std.shape),
        "weight_shape": list(probe_weight.shape),
        "bias_shape": list(probe_bias.shape),
        "weight_min": float(probe_weight.detach().min()),
        "weight_max": float(probe_weight.detach().max()),
        "bias_min": float(probe_bias.detach().min()),
        "tasgnss_differentiable_solve_status": bool(graph_solution["status"]),
        "position_loss_has_autograd_graph": graph_loss.grad_fn is not None,
        "nonzero_gradient_parameter_tensors": nonzero_gradient_parameters,
        "gradient_reached_network": bool(first_summary["gradient_reached_network"]),
        "optimizer_changed_parameters": bool(first_summary["optimizer_changed_parameters"]),
        "checkpoint_saved": checkpoint.is_file(),
        "checkpoint_reload_identical_outputs": checkpoint_reload_equal,
        "same_seed_training_replay_identical": deterministic_replay,
        "seed_protocol": seed_checks,
        "gt_leakage": {
            "fingerprints_before": original_fingerprint,
            "fingerprints_after_gt_change": changed_fingerprint,
            "features_neutral_and_solver_inputs_unchanged": original_fingerprint == changed_fingerprint,
            "original_loss_m": original_gt_loss,
            "changed_gt_loss_m": changed_gt_loss,
            "only_loss_changed": original_fingerprint == changed_fingerprint and original_gt_loss != changed_gt_loss,
        },
        "timing_estimate": {
            "mean_smoke_epoch_seconds": mean_epoch_seconds,
            "linear_extrapolation_one_120_epoch_seed_seconds": estimated_one_seed_seconds,
            "linear_extrapolation_ten_seeds_seconds": estimated_one_seed_seconds * 10,
            "caveat": (
                "linear CPU estimate from one truncated full-KLT3 epoch; excludes "
                "the measured one-time preprocessing stage"
            ),
        },
        "disk_estimate": {
            "one_checkpoint_bytes": final_checkpoint_bytes,
            "ten_seeds_thirteen_checkpoints_each_bytes": estimated_ten_seed_checkpoint_bytes,
            "shared_klt3_preprocessing_cache_bytes": preprocessing_bytes,
            "training_total_before_small_manifests_bytes": (
                estimated_ten_seed_checkpoint_bytes + preprocessing_bytes
            ),
            "caveat": "excludes small per-seed JSON/CSV manifests and future KLT1/KLT2 evaluation artifacts",
        },
    }
    write_json(smoke_root / "smoke_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verify_provenance()
    output_root = args.output_root.resolve()
    if args.command == "preprocess":
        records, manifest = load_or_preprocess_dataset("KLT3", output_root, force=args.force)
        print(json.dumps({"records": len(records), "manifest": manifest}, indent=2, sort_keys=True))
        return 0
    records, _manifest = load_or_preprocess_dataset("KLT3", output_root)
    if args.command == "smoke":
        report = run_smoke(
            records,
            seed=args.seed,
            subset_size=args.subset_size,
            epochs=args.epochs,
            output_root=output_root,
            threads=args.threads,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "seed":
        summary = train_seed(
            records,
            seed=args.seed,
            epochs=args.epochs,
            output_root=output_root,
            threads=args.threads,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "all":
        if tuple(args.seeds) != SEEDS:
            raise RuntimeError(f"full experiment requires exactly ordered seeds {SEEDS}")
        if args.epochs != 120:
            raise RuntimeError("full experiment requires exactly 120 epochs")
        occupied = [
            output_root / f"seed_{seed}"
            for seed in args.seeds
            if (output_root / f"seed_{seed}").exists()
            and any((output_root / f"seed_{seed}").iterdir())
        ]
        if occupied:
            raise FileExistsError(
                "refusing to begin a partial all-seed run because these directories are non-empty: "
                + ", ".join(str(path) for path in occupied)
            )
        summaries = [
            train_seed(
                records,
                seed=seed,
                epochs=args.epochs,
                output_root=output_root,
                threads=args.threads,
            )
            for seed in args.seeds
        ]
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
