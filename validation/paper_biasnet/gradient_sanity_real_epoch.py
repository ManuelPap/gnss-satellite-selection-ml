#!/usr/bin/env python3
"""Validate BiasNet gradients and finite differences on one real KLT3 epoch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .core import instantiate_released_biasnet, solve_paper_bias_position
    from .train_paper_biasnet import (
        DEFAULT_SEED,
        epoch_position_loss,
        gradient_audit,
        load_dataset,
        parameter_changes,
        parameter_snapshot,
        set_seed,
    )
except ImportError:
    from core import instantiate_released_biasnet, solve_paper_bias_position
    from train_paper_biasnet import (
        DEFAULT_SEED,
        epoch_position_loss,
        gradient_audit,
        load_dataset,
        parameter_changes,
        parameter_snapshot,
        set_seed,
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    shared = here.parent / "paper_weightnet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=shared / "klt3_features.npz")
    parser.add_argument(
        "--manifest", type=Path, default=shared / "klt3_feature_manifest.json"
    )
    parser.add_argument("--epoch-index", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=here / "gradient_sanity.json")
    return parser.parse_args()


def relative_error(autograd: float, finite_difference: float) -> float:
    scale = max(1.0, abs(autograd), abs(finite_difference))
    return abs(autograd - finite_difference) / scale


def main() -> int:
    args = parse_args()
    device = torch.device("cpu")
    dataset, _manifest = load_dataset(args.features, args.manifest, device)
    offsets = dataset["epoch_offsets"]
    if not 0 <= args.epoch_index < offsets.size - 1:
        raise IndexError(args.epoch_index)
    start = int(offsets[args.epoch_index])
    stop = int(offsets[args.epoch_index + 1])
    kwargs = {
        "satellite_positions_ecef_m": dataset["satellite_positions_ecef_m"][start:stop],
        "satellite_clock_bias_s": dataset["satellite_clock_bias_s"][start:stop],
        "corrected_pseudorange_m": dataset["corrected_pseudorange_m"][start:stop],
        "system_clock_indices": dataset["system_clock_indices"][start:stop],
        "initial_state": dataset["initial_states"][args.epoch_index],
        "return_trace": True,
    }
    gt = dataset["ground_truth_geodetic_deg_m"][args.epoch_index]

    set_seed(args.seed)
    model = instantiate_released_biasnet(
        dataset["features"].mean(axis=0), dataset["features"].std(axis=0)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    features = torch.as_tensor(dataset["features"][start:stop], dtype=torch.float32)
    predicted = model(features).squeeze(-1)
    predicted.retain_grad()
    solution = solve_paper_bias_position(predicted_bias_m=predicted, **kwargs)
    loss = epoch_position_loss(solution.state, gt)
    before = parameter_snapshot(model)
    loss.backward()
    audit = gradient_audit(model)
    predicted_gradient = predicted.grad.detach().clone()
    optimizer.step()
    changes = parameter_changes(before, model)

    direct_bias = predicted.detach().clone().requires_grad_(True)
    direct_solution = solve_paper_bias_position(predicted_bias_m=direct_bias, **kwargs)
    direct_loss = epoch_position_loss(direct_solution.state, gt)
    direct_loss.backward()
    assert direct_bias.grad is not None
    indices = sorted({0, direct_bias.numel() // 2, direct_bias.numel() - 1})
    checks: list[dict[str, float | int]] = []
    for index in indices:
        plus = direct_bias.detach().clone()
        minus = direct_bias.detach().clone()
        plus[index] += args.epsilon
        minus[index] -= args.epsilon
        plus_loss = epoch_position_loss(
            solve_paper_bias_position(predicted_bias_m=plus, **kwargs).state, gt
        )
        minus_loss = epoch_position_loss(
            solve_paper_bias_position(predicted_bias_m=minus, **kwargs).state, gt
        )
        finite_difference = float((plus_loss - minus_loss) / (2.0 * args.epsilon))
        automatic = float(direct_bias.grad[index])
        checks.append(
            {
                "bias_row": index,
                "autograd": automatic,
                "symmetric_finite_difference": finite_difference,
                "absolute_difference": abs(automatic - finite_difference),
                "scaled_relative_error": relative_error(automatic, finite_difference),
            }
        )

    output = {
        "status": "passed",
        "description": "gradient sanity check, not a formal proof of all derivatives",
        "seed": args.seed,
        "epoch_index": args.epoch_index,
        "measurement_rows": stop - start,
        "loss_3d_m": float(loss.detach()),
        "predicted_bias_all_finite": bool(torch.all(torch.isfinite(predicted.detach()))),
        "predicted_bias_gradient_present": predicted.grad is not None,
        "predicted_bias_gradient_all_finite": bool(
            torch.all(torch.isfinite(predicted_gradient))
        ),
        "predicted_bias_gradient_l2_norm": float(
            torch.linalg.vector_norm(predicted_gradient)
        ),
        "wls_state_all_finite": bool(torch.all(torch.isfinite(solution.state))),
        "wls_iterations": len(solution.iterations),
        "model_gradient_audit": audit,
        "parameter_max_abs_changes_after_one_adam_step": changes,
        "every_expected_trainable_tensor_changed": all(
            change > 0.0 for change in changes.values()
        ),
        "finite_difference_epsilon_m": args.epsilon,
        "finite_difference": checks,
        "maximum_scaled_relative_error": max(
            float(item["scaled_relative_error"]) for item in checks
        ),
    }
    if not audit["every_trainable_parameter_has_gradient"]:
        raise RuntimeError("a model tensor did not receive a gradient")
    if not audit["every_trainable_parameter_gradient_finite"]:
        raise RuntimeError("a model gradient is non-finite")
    if not output["every_expected_trainable_tensor_changed"]:
        raise RuntimeError("an expected trainable tensor did not change")
    if output["maximum_scaled_relative_error"] > 1.0e-4:
        raise RuntimeError("finite-difference discrepancy exceeds 1e-4")
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
