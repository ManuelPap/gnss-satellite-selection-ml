#!/usr/bin/env python3
"""Validate that initialization seed is the sweep's only scientific variable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from validation.paper_hybrid_seed_sensitivity.experiment import (  # noqa: E402
    PREDEFINED_SEEDS,
    build_initial_snapshot,
    load_dataset,
    normalization_identity,
    training_data_identity,
    validate_training_dataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    shared = REPOSITORY / "validation/paper_weightnet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PREDEFINED_SEEDS))
    parser.add_argument("--features", type=Path, default=shared / "klt3_features.npz")
    parser.add_argument(
        "--manifest", type=Path, default=shared / "klt3_feature_manifest.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY
        / "results/paper_hybrid_seed_sensitivity/control_validation.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_seed_list(seeds: list[int]) -> None:
    if len(seeds) < 2:
        raise ValueError("control validation needs at least two distinct seeds")
    if len(seeds) != len(set(seeds)):
        raise ValueError("control-validation seeds must be distinct")
    outside = sorted(set(seeds) - set(PREDEFINED_SEEDS))
    if outside:
        raise ValueError(f"control seeds are fixed to 0..9: {outside}")


def validate_controls(
    dataset: dict[str, object],
    manifest: dict[str, object],
    seeds: list[int],
    device: torch.device,
) -> dict[str, object]:
    """Construct untrained models only; no optimization step is performed."""

    validate_seed_list(seeds)
    validate_training_dataset(dataset)

    # These identities are deliberately computed before any seeded constructor.
    data = training_data_identity(dataset, manifest)
    normalization = normalization_identity(dataset)
    snapshots: list[dict[str, object]] = []
    for seed in seeds:
        _model, _optimizer, snapshot = build_initial_snapshot(dataset, seed, device)
        snapshot["training_data_identity_sha256"] = data["identity_sha256"]
        snapshots.append(snapshot)

    _repeat_model, _repeat_optimizer, repeated = build_initial_snapshot(
        dataset, seeds[0], device
    )
    repeated["training_data_identity_sha256"] = data["identity_sha256"]
    same_seed_parameter_match = (
        repeated["initial_parameter_sha256"]
        == snapshots[0]["initial_parameter_sha256"]
    )
    same_seed_output_match = (
        repeated["initial_output_sha256"] == snapshots[0]["initial_output_sha256"]
    )
    same_seed_tensor_match = (
        repeated["initial_parameter_tensor_sha256"]
        == snapshots[0]["initial_parameter_tensor_sha256"]
    )

    reference_tensors = snapshots[0]["initial_parameter_tensor_sha256"]
    differences: dict[str, list[str]] = {}
    for snapshot in snapshots[1:]:
        current = snapshot["initial_parameter_tensor_sha256"]
        differences[str(snapshot["seed"])] = [
            name for name in reference_tensors if reference_tensors[name] != current[name]
        ]
    pairwise_differences: dict[str, list[str]] = {}
    for left_index, left in enumerate(snapshots):
        left_tensors = left["initial_parameter_tensor_sha256"]
        for right in snapshots[left_index + 1 :]:
            right_tensors = right["initial_parameter_tensor_sha256"]
            pairwise_differences[f"{left['seed']}_vs_{right['seed']}"] = [
                name
                for name in left_tensors
                if left_tensors[name] != right_tensors[name]
            ]

    invariant_fields = (
        "configuration_sha256",
        "normalization_sha256",
        "training_data_identity_sha256",
        "architecture_sha256",
        "optimizer",
    )
    invariants_identical = {
        field: all(snapshot[field] == snapshots[0][field] for snapshot in snapshots)
        for field in invariant_fields
    }
    different_seed_parameters_differ = all(pairwise_differences.values())
    checks = {
        "data_rows_gt_features_hashed_before_initialization": True,
        "data_rows_gt_features_identical_across_seeds": invariants_identical[
            "training_data_identity_sha256"
        ],
        "same_seed_initial_parameter_hash_identical": same_seed_parameter_match,
        "same_seed_initial_parameter_tensors_identical": same_seed_tensor_match,
        "same_seed_initial_outputs_identical": same_seed_output_match,
        "different_seed_has_at_least_one_different_parameter_tensor": (
            different_seed_parameters_differ
        ),
        "normalization_identical_across_seeds": invariants_identical[
            "normalization_sha256"
        ],
        "architecture_identical_across_seeds": invariants_identical[
            "architecture_sha256"
        ],
        "optimizer_and_learning_rate_identical_across_seeds": invariants_identical[
            "optimizer"
        ],
        "epoch_solver_evaluation_configuration_identical_across_seeds": (
            invariants_identical["configuration_sha256"]
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "seeds": seeds,
        "intended_variable": "seed -> default-initialized trainable model parameters",
        "invariant_controls": {
            "training_data": data,
            "normalization": normalization,
            "configuration_sha256": snapshots[0]["configuration_sha256"],
            "architecture_sha256": snapshots[0]["architecture_sha256"],
            "optimizer": snapshots[0]["optimizer"],
            "includes": [
                "data",
                "row identity",
                "ground truth",
                "features before model initialization",
                "normalization",
                "architecture",
                "optimizer",
                "learning rate",
                "100-epoch count",
                "WLS solver settings",
                "KLT1/KLT2 evaluation procedure",
            ],
        },
        "per_seed_initialization": snapshots,
        "same_seed_repeat": repeated,
        "different_seed_parameter_tensors_vs_first_seed": differences,
        "different_seed_parameter_tensors_pairwise": pairwise_differences,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output}")
    device = torch.device(args.device)
    dataset, manifest = load_dataset(args.features, args.manifest, device)
    result = validate_controls(dataset, manifest, args.seeds, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"control validation: {result['status']}")
    print(f"result: {output}")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
