#!/usr/bin/env python3
"""Refresh post-training KLT3 BiasNet diagnostics without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .held_out import load_frozen_biasnet
    from .train_paper_biasnet import load_dataset, training_bias_diagnostics
except ImportError:
    from held_out import load_frozen_biasnet
    from train_paper_biasnet import load_dataset, training_bias_diagnostics


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    shared = root / "validation/paper_weightnet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=shared / "klt3_features.npz")
    parser.add_argument("--manifest", type=Path, default=shared / "klt3_feature_manifest.json")
    parser.add_argument("--metrics", type=Path, default=here / "training_metrics.json")
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "checkpoints/paper_biasnet/biasnet_3d.pth"
    )
    args = parser.parse_args()
    dataset, _manifest = load_dataset(args.features, args.manifest, torch.device("cpu"))
    model = load_frozen_biasnet(args.checkpoint, args.metrics)
    record = json.loads(args.metrics.resolve().read_text())
    record["bias_output_diagnostics_klt3"] = training_bias_diagnostics(
        model, dataset, torch.device("cpu")
    )
    args.metrics.resolve().write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["bias_output_diagnostics_klt3"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
