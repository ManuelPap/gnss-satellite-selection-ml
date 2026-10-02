#!/usr/bin/env python3
"""Validate both HybridShareNet gradient paths on one real KLT3 epoch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .core import instantiate_released_hybrid, solve_paper_hybrid_position
    from .train_paper_hybrid import (
        DEFAULT_SEED,
        RELEASED_LEARNING_RATE,
        epoch_position_loss,
        gradient_audit,
        parameter_changes,
        parameter_snapshot,
        set_seed,
    )
except ImportError:
    from core import instantiate_released_hybrid, solve_paper_hybrid_position
    from train_paper_hybrid import (
        DEFAULT_SEED,
        RELEASED_LEARNING_RATE,
        epoch_position_loss,
        gradient_audit,
        parameter_changes,
        parameter_snapshot,
        set_seed,
    )

from validation.paper_biasnet.train_paper_biasnet import load_dataset


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    shared = here.parent / "paper_weightnet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=shared / "klt3_features.npz")
    parser.add_argument(
        "--manifest", type=Path, default=shared / "klt3_feature_manifest.json"
    )
    parser.add_argument("--epoch-index", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=here / "gradient_sanity.json")
    return parser.parse_args()


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(1.0, abs(left), abs(right))


def tensor_global_norm(values: tuple[torch.Tensor | None, ...]) -> float:
    total = 0.0
    for value in values:
        if value is not None:
            total += float(torch.sum(value.detach() ** 2))
    return total**0.5


def maximum_gradient_sum_discrepancy(
    total: tuple[torch.Tensor | None, ...],
    bias: tuple[torch.Tensor | None, ...],
    weight: tuple[torch.Tensor | None, ...],
) -> tuple[float, float]:
    maximum = 0.0
    scale = 1.0
    for total_value, bias_value, weight_value in zip(total, bias, weight, strict=True):
        if total_value is None:
            continue
        combined = torch.zeros_like(total_value)
        if bias_value is not None:
            combined = combined + bias_value
        if weight_value is not None:
            combined = combined + weight_value
        maximum = max(maximum, float(torch.max(torch.abs(total_value - combined))))
        scale = max(scale, float(torch.max(torch.abs(total_value))))
    return maximum, maximum / scale


def main() -> int:
    args = parse_args()
    device = torch.device("cpu")
    dataset, _manifest = load_dataset(args.features, args.manifest, device)
    offsets = dataset["epoch_offsets"]
    start = int(offsets[args.epoch_index])
    stop = int(offsets[args.epoch_index + 1])
    features = torch.as_tensor(dataset["features"][start:stop], dtype=torch.float32)
    gt = dataset["ground_truth_geodetic_deg_m"][args.epoch_index]
    solver_inputs = {
        "satellite_positions_ecef_m": dataset["satellite_positions_ecef_m"][start:stop],
        "satellite_clock_bias_s": dataset["satellite_clock_bias_s"][start:stop],
        "corrected_pseudorange_m": dataset["corrected_pseudorange_m"][start:stop],
        "system_clock_indices": dataset["system_clock_indices"][start:stop],
        "initial_state": dataset["initial_states"][args.epoch_index],
        "return_trace": True,
    }

    set_seed(args.seed)
    model = instantiate_released_hybrid(
        dataset["features"].mean(axis=0), dataset["features"].std(axis=0)
    )
    with torch.no_grad():
        raw_before_control = model.raw_output(features)[:, 1]
        # Seed 20260929 makes every epoch-zero bias preactivation negative.  A
        # transparent +1 m final-bias offset keeps the exact architecture and
        # real inputs while moving this derivative audit away from ReLU's dead
        # region.  Full training does not apply this control.
        model.seq[7].bias[1] += 1.0
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    parameter_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )

    weight, bias = model(features)
    weight.retain_grad()
    bias.retain_grad()
    total_solution = solve_paper_hybrid_position(
        predicted_weight=weight, predicted_bias_m=bias, **solver_inputs
    )
    total_loss = epoch_position_loss(total_solution.state, gt)
    total_grads = torch.autograd.grad(
        total_loss, (*parameters, weight, bias), retain_graph=True, allow_unused=True
    )
    total_parameter_grads = total_grads[: len(parameters)]
    weight_output_gradient = total_grads[-2]
    bias_output_gradient = total_grads[-1]

    bias_solution = solve_paper_hybrid_position(
        predicted_weight=weight.detach(), predicted_bias_m=bias, **solver_inputs
    )
    bias_loss = epoch_position_loss(bias_solution.state, gt)
    bias_parameter_grads = torch.autograd.grad(
        bias_loss, parameters, retain_graph=True, allow_unused=True
    )
    weight_solution = solve_paper_hybrid_position(
        predicted_weight=weight, predicted_bias_m=bias.detach(), **solver_inputs
    )
    weight_loss = epoch_position_loss(weight_solution.state, gt)
    weight_parameter_grads = torch.autograd.grad(
        weight_loss, parameters, retain_graph=False, allow_unused=True
    )
    sum_absolute, sum_relative = maximum_gradient_sum_discrepancy(
        total_parameter_grads, bias_parameter_grads, weight_parameter_grads
    )

    # A fresh graph verifies an actual optimizer step and conventional .grad.
    optimizer = torch.optim.Adam(model.parameters(), lr=RELEASED_LEARNING_RATE)
    optimizer.zero_grad()
    update_weight, update_bias = model(features)
    update_solution = solve_paper_hybrid_position(
        predicted_weight=update_weight,
        predicted_bias_m=update_bias,
        **solver_inputs,
    )
    update_loss = epoch_position_loss(update_solution.state, gt)
    before = parameter_snapshot(model)
    update_loss.backward()
    conventional_audit = gradient_audit(model)
    optimizer.step()
    changes = parameter_changes(before, model)

    def loss_value() -> float:
        current_weight, current_bias = model(features)
        current = solve_paper_hybrid_position(
            predicted_weight=current_weight,
            predicted_bias_m=current_bias,
            **solver_inputs,
        )
        return float(epoch_position_loss(current.state, gt).detach())

    # Central differences at the largest-gradient entries of a shared layer
    # and each output row.  These are numerical sanity checks, not a proof.
    named_parameters = dict(model.named_parameters())
    checks: list[dict[str, object]] = []
    finite_difference_targets = []
    shared_gradient = total_parameter_grads[parameter_names.index("seq.1.weight")]
    assert shared_gradient is not None
    shared_flat = int(torch.argmax(torch.abs(shared_gradient)))
    finite_difference_targets.append(("seq.1.weight", np.unravel_index(shared_flat, shared_gradient.shape)))
    output_gradient = total_parameter_grads[parameter_names.index("seq.7.weight")]
    assert output_gradient is not None
    for row, branch in ((0, "weight"), (1, "bias")):
        column = int(torch.argmax(torch.abs(output_gradient[row])))
        finite_difference_targets.append(("seq.7.weight", (row, column, branch)))

    # The optimizer step changed the evaluation point, so obtain its exact
    # autograd gradient for comparison with finite differences.
    model.zero_grad()
    fd_weight, fd_bias = model(features)
    fd_solution = solve_paper_hybrid_position(
        predicted_weight=fd_weight, predicted_bias_m=fd_bias, **solver_inputs
    )
    fd_loss = epoch_position_loss(fd_solution.state, gt)
    fd_loss.backward()
    for name, raw_index in finite_difference_targets:
        branch = "shared" if len(raw_index) == 2 else str(raw_index[2])
        index = tuple(int(value) for value in raw_index[:2])
        parameter = named_parameters[name]
        automatic = float(parameter.grad[index])
        with torch.no_grad():
            original = float(parameter[index])
            parameter[index] = original + args.epsilon
        plus = loss_value()
        with torch.no_grad():
            parameter[index] = original - args.epsilon
        minus = loss_value()
        with torch.no_grad():
            parameter[index] = original
        numerical = (plus - minus) / (2.0 * args.epsilon)
        checks.append(
            {
                "parameter": name,
                "index": list(index),
                "branch": branch,
                "autograd": automatic,
                "symmetric_finite_difference": numerical,
                "absolute_difference": abs(automatic - numerical),
                "scaled_relative_error": relative_error(automatic, numerical),
            }
        )

    output = {
        "status": "passed",
        "description": "controlled dual-path and finite-difference numerical sanity checks, not formal derivative proofs",
        "seed": args.seed,
        "epoch_index": args.epoch_index,
        "measurement_rows": stop - start,
        "controlled_relu_activation": {
            "reason": "the training seed puts every bias output in the ReLU dead region on this epoch before training",
            "raw_bias_preactivation_min_before_offset": float(raw_before_control.min()),
            "raw_bias_preactivation_max_before_offset": float(raw_before_control.max()),
            "final_bias_parameter_offset_m_for_gradient_audit_only": 1.0,
            "applied_to_full_training": False,
        },
        "loss_3d_m": float(total_loss.detach()),
        "all_outputs_finite": bool(
            torch.all(torch.isfinite(weight)) and torch.all(torch.isfinite(bias))
        ),
        "all_wls_states_finite": all(
            bool(torch.all(torch.isfinite(item.state)))
            for item in (total_solution, bias_solution, weight_solution)
        ),
        "bias_output_gradient_l2_norm": float(torch.linalg.vector_norm(bias_output_gradient)),
        "weight_output_gradient_l2_norm": float(torch.linalg.vector_norm(weight_output_gradient)),
        "bias_path_shared_parameter_gradient_l2_norm": tensor_global_norm(bias_parameter_grads),
        "weight_path_shared_parameter_gradient_l2_norm": tensor_global_norm(weight_parameter_grads),
        "total_shared_parameter_gradient_l2_norm": tensor_global_norm(total_parameter_grads),
        "gradient_additivity": {
            "maximum_absolute_discrepancy": sum_absolute,
            "maximum_scaled_discrepancy": sum_relative,
            "relationship": "grad_total ~= grad_bias_path + grad_weight_path at identical output values",
        },
        "conventional_backward_gradient_audit": conventional_audit,
        "parameter_max_abs_changes_after_one_adam_step": changes,
        "every_expected_trainable_tensor_changed": all(
            value > 0.0 for value in changes.values()
        ),
        "finite_difference_epsilon": args.epsilon,
        "finite_difference": checks,
        "maximum_finite_difference_scaled_relative_error": max(
            float(item["scaled_relative_error"]) for item in checks
        ),
    }
    required_positive = (
        output["bias_output_gradient_l2_norm"],
        output["weight_output_gradient_l2_norm"],
        output["bias_path_shared_parameter_gradient_l2_norm"],
        output["weight_path_shared_parameter_gradient_l2_norm"],
    )
    if not all(float(value) > 0.0 for value in required_positive):
        raise RuntimeError("one hybrid branch has zero gradient")
    if not conventional_audit["every_trainable_parameter_has_gradient"]:
        raise RuntimeError("a trainable tensor lacks a gradient")
    if not conventional_audit["every_trainable_parameter_gradient_finite"]:
        raise RuntimeError("a trainable tensor gradient is non-finite")
    if not output["every_expected_trainable_tensor_changed"]:
        raise RuntimeError("an expected trainable tensor did not change")
    if sum_relative > 1.0e-10:
        raise RuntimeError("isolated gradient paths do not add to total")
    if output["maximum_finite_difference_scaled_relative_error"] > 5.0e-3:
        raise RuntimeError("finite-difference discrepancy exceeds tolerance")
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
