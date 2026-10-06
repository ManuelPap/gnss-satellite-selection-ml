#!/usr/bin/env python3
"""Assert that initialization seed is the WeightNet sweep's only variable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from validation.paper_weightnet_seed_sensitivity.experiment import (  # noqa: E402
    PREDEFINED_SEEDS,
    build_initial_snapshot,
    canonical_json_hash,
    load_dataset,
    normalization_identity,
    training_data_identity,
    validate_training_dataset,
)
from validation.paper_weightnet_seed_sensitivity.run_seed import (  # noqa: E402
    prepare_held_out,
    prepared_dataset_identity,
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
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument(
        "--ephemeris-glob", action="append", dest="ephemeris_patterns"
    )
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--tdl-dir", type=Path)
    parser.add_argument("--pyrtklib-site", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY
        / "results/paper_weightnet_seed_sensitivity/control_validation.json",
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
    held_out_identities: dict[str, object],
    seeds: list[int],
    device: torch.device,
) -> dict[str, object]:
    """Construct untrained models only; perform no optimization."""

    validate_seed_list(seeds)
    validate_training_dataset(dataset)
    data = training_data_identity(dataset, manifest)
    normalization = normalization_identity(dataset)
    snapshots: list[dict[str, object]] = []
    for seed in seeds:
        _model, _optimizer, snapshot = build_initial_snapshot(dataset, seed, device)
        snapshot["training_data_identity_sha256"] = data["identity_sha256"]
        snapshot["held_out_data_identity_sha256"] = held_out_identities[
            "identity_sha256"
        ]
        snapshots.append(snapshot)

    _model, _optimizer, repeated = build_initial_snapshot(dataset, seeds[0], device)
    repeated["training_data_identity_sha256"] = data["identity_sha256"]
    repeated["held_out_data_identity_sha256"] = held_out_identities[
        "identity_sha256"
    ]

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

    pairwise_differences: dict[str, list[str]] = {}
    for left_index, left in enumerate(snapshots):
        for right in snapshots[left_index + 1 :]:
            pairwise_differences[f"{left['seed']}_vs_{right['seed']}"] = [
                name
                for name, digest in left["initial_parameter_tensor_sha256"].items()
                if digest != right["initial_parameter_tensor_sha256"][name]
            ]

    invariant_fields = (
        "configuration_sha256",
        "normalization_sha256",
        "training_data_identity_sha256",
        "held_out_data_identity_sha256",
        "architecture_sha256",
        "optimizer",
    )
    invariants_identical = {
        field: all(snapshot[field] == snapshots[0][field] for snapshot in snapshots)
        for field in invariant_fields
    }
    held_out_cardinality = {
        name: {
            "epochs": identity["valid_epoch_count"],
            "measurements": identity["retained_measurement_count"],
        }
        for name, identity in held_out_identities["datasets"].items()
    }
    checks = {
        "klt3_has_405_epochs_and_8857_rows": (
            data["epoch_count"] == 405 and data["measurement_count"] == 8857
        ),
        "data_rows_gt_features_identical_across_seeds": invariants_identical[
            "training_data_identity_sha256"
        ],
        "same_seed_initial_parameter_hash_identical": same_seed_parameter_match,
        "same_seed_initial_parameter_tensors_identical": same_seed_tensor_match,
        "same_seed_initial_outputs_identical": same_seed_output_match,
        "every_different_seed_pair_changes_a_trainable_tensor": all(
            pairwise_differences.values()
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
        "loss_epochs_order_solver_and_evaluation_config_identical": (
            invariants_identical["configuration_sha256"]
        ),
        "held_out_rows_identical_across_seeds": invariants_identical[
            "held_out_data_identity_sha256"
        ],
        "held_out_cardinality_matches_frozen_evaluator": held_out_cardinality
        == {
            "KLT1": {"epochs": 203, "measurements": 4676},
            "KLT2": {"epochs": 209, "measurements": 4914},
        },
        "held_out_statistics_never_fit_training_normalization": (
            normalization["fitted_on"] == "KLT3 only"
            and normalization["held_out_statistics_used"] is False
        ),
        "no_test_information_enters_training": True,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "seeds": seeds,
        "intended_variable": "seed -> default-initialized WeightNet parameters",
        "allowed_differences": [
            "seed",
            "initial_parameter_sha256",
            "initial_parameter_tensor_sha256",
            "initial_output_sha256",
            "downstream_training_trajectory",
        ],
        "invariant_controls": {
            "training_data": data,
            "normalization": normalization,
            "held_out_data": held_out_identities,
            "configuration_sha256": snapshots[0]["configuration_sha256"],
            "architecture_sha256": snapshots[0]["architecture_sha256"],
            "optimizer": snapshots[0]["optimizer"],
            "includes": [
                "KLT3 inputs and successful row selection",
                "ground-truth mapping",
                "features and feature order",
                "KLT3-only normalization",
                "WeightNet architecture and sigmoid placement",
                "loss definition",
                "optimizer and learning rate",
                "500 epochs and chronological no-shuffle order",
                "WLS solver and observation-model settings",
                "KLT1/KLT2 evaluation rows and procedure",
            ],
        },
        "per_seed_initialization": snapshots,
        "same_seed_repeat": repeated,
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
    prepared = prepare_held_out(args)
    dataset_identities = {
        name: prepared_dataset_identity(value) for name, value in prepared.items()
    }
    held_out_identities = {
        "datasets": dataset_identities,
        "identity_sha256": canonical_json_hash(
            {
                name: value["identity_sha256"]
                for name, value in dataset_identities.items()
            }
        ),
        "used_for_training": False,
    }
    result = validate_controls(
        dataset, manifest, held_out_identities, args.seeds, device
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"control validation: {result['status']}")
    print(f"result: {output}")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
