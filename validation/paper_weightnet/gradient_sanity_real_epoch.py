#!/usr/bin/env python3
"""Check real-KLT3 gradients through weights, WLS, and position loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pymap3d as p3d
import torch

from core import WeightNet, solve_paper_weighted_position


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=HERE / "klt3_features.npz")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "checkpoints/paper_weightnet/weightnet_3d.pth",
    )
    parser.add_argument("--epoch-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "gradient_sanity.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def position_loss(state: torch.Tensor, gt: np.ndarray) -> torch.Tensor:
    east, north, up = p3d.ecef2enu(
        state[0], state[1], state[2], float(gt[0]), float(gt[1]), float(gt[2])
    )
    return torch.linalg.vector_norm(torch.stack((east, north, up)))


def solve_loss(
    weights: torch.Tensor,
    cache: dict[str, np.ndarray],
    epoch_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = cache["epoch_offsets"]
    start = int(offsets[epoch_index])
    stop = int(offsets[epoch_index + 1])
    solution = solve_paper_weighted_position(
        cache["satellite_positions_ecef_m"][start:stop],
        cache["satellite_clock_bias_s"][start:stop],
        cache["corrected_pseudorange_m"][start:stop],
        cache["system_clock_indices"][start:stop],
        weights,
        cache["initial_states"][epoch_index],
    )
    loss = position_loss(
        solution.state, cache["ground_truth_geodetic_deg_m"][epoch_index]
    )
    return loss, solution.state


def main() -> int:
    args = parse_args()
    with np.load(args.features.resolve(), allow_pickle=False) as source:
        cache = {name: source[name].copy() for name in source.files}
    if not 0 <= args.epoch_index < cache["initial_states"].shape[0]:
        raise ValueError("--epoch-index is outside the cached KLT3 range")
    offsets = cache["epoch_offsets"]
    start = int(offsets[args.epoch_index])
    stop = int(offsets[args.epoch_index + 1])

    model = WeightNet()
    model.double()
    model.load_state_dict(
        torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    )
    model.eval()
    features = torch.as_tensor(cache["features"][start:stop], dtype=torch.float32)
    with torch.no_grad():
        predicted = model(features).squeeze(-1)
    weights = predicted.detach().clone().requires_grad_(True)
    loss, state = solve_loss(weights, cache, args.epoch_index)
    loss.backward()
    gradient = weights.grad
    if gradient is None or not bool(torch.all(torch.isfinite(gradient))):
        raise RuntimeError("loss-to-weight gradient is absent or non-finite")
    if not bool(torch.any(gradient != 0.0)):
        raise RuntimeError("loss-to-weight gradient is identically zero")
    if not bool(torch.all(torch.isfinite(state))):
        raise RuntimeError("real-epoch WLS state is non-finite")

    # Use the three largest learned weights. Their positive margin permits a
    # symmetric difference without leaving the released valid weight domain.
    checked_indices = torch.argsort(weights.detach(), descending=True)[:3].tolist()
    finite_differences: list[dict[str, float | int]] = []
    for index in checked_indices:
        value = float(weights[index].detach())
        epsilon = 1.0e-3 * max(1.0, abs(value))
        if value - epsilon <= 0.0:
            epsilon = value * 0.25
        plus = weights.detach().clone()
        minus = weights.detach().clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_loss, _ = solve_loss(plus, cache, args.epoch_index)
        minus_loss, _ = solve_loss(minus, cache, args.epoch_index)
        numerical = float((plus_loss - minus_loss) / (2.0 * epsilon))
        automatic = float(gradient[index])
        absolute = abs(automatic - numerical)
        symmetric_relative = absolute / max(
            1.0e-12, abs(automatic) + abs(numerical)
        )
        finite_differences.append(
            {
                "weight_index": index,
                "weight": value,
                "epsilon": epsilon,
                "autograd": automatic,
                "central_difference": numerical,
                "absolute_error": absolute,
                "symmetric_relative_error": symmetric_relative,
            }
        )
    maximum_relative = max(
        item["symmetric_relative_error"] for item in finite_differences
    )
    if maximum_relative > 1.0e-4:
        raise RuntimeError(
            f"weight-gradient finite-difference check failed: {maximum_relative}"
        )

    result = {
        "status": "passed",
        "epoch_index": args.epoch_index,
        "epoch_time_gpst_like": float(cache["epoch_times_gpst_like"][args.epoch_index]),
        "satellite_count": stop - start,
        "loss_3d_m": float(loss.detach()),
        "wls_state_all_finite": True,
        "all_weight_gradients_finite": True,
        "at_least_one_weight_gradient_nonzero": True,
        "weight_gradient_l2_norm": float(torch.linalg.vector_norm(gradient)),
        "finite_difference": finite_differences,
        "maximum_symmetric_relative_error": maximum_relative,
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "ground_truth_usage": "3D position loss only; absent from features",
    }
    args.output.resolve().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"loss: {result['loss_3d_m']:.12f}")
    print(f"weight-gradient L2 norm: {result['weight_gradient_l2_norm']:.12e}")
    print(f"finite-difference maximum relative error: {maximum_relative:.12e}")
    print(f"result: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
