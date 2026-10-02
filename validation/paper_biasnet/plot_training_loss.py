#!/usr/bin/env python3
"""Plot the reproduced BiasNet loss history without paper-curve digitization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    here = Path(__file__).resolve().parent
    repository = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=here / "training_metrics.json")
    parser.add_argument(
        "--output", type=Path, default=repository / "results/paper_biasnet/training_loss.png"
    )
    parser.add_argument(
        "--csv", type=Path, default=repository / "results/paper_biasnet/training_loss.csv"
    )
    args = parser.parse_args()
    metrics = json.loads(args.metrics.resolve().read_text())
    rows = metrics["epochs"]
    epochs = np.asarray([item["epoch"] for item in rows], dtype=np.int64)
    sums = np.asarray([item["loss_sum_3d_m"] for item in rows], dtype=np.float64)
    means = np.asarray(
        [item["loss_divided_by_405_like_released_print"] for item in rows],
        dtype=np.float64,
    )

    args.csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.csv.resolve().open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "released_saved_sum_3d_m", "released_printed_mean_like_3d_m"))
        writer.writerows(zip(epochs.tolist(), sums.tolist(), means.tolist(), strict=True))

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(epochs, means, color="#1769aa", linewidth=1.4)
    axis.set_xlabel("Training epoch")
    axis.set_ylabel("Position loss / 405 [m]")
    axis.set_title("Released-code BiasNet training-loss reproduction")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output.resolve(), dpi=180)
    plt.close(figure)

    minimum_index = int(np.argmin(means))
    result = {
        "epochs": int(epochs.size),
        "historical_saved_curve_definition": "sum of per-KLT3-epoch 3D ENU Euclidean norms",
        "historical_printed_quantity": "saved sum divided by len(obss)=405",
        "paper_caption_definition": "mean position loss",
        "initial_mean_like_loss_m": float(means[0]),
        "final_mean_like_loss_m": float(means[-1]),
        "minimum_mean_like_loss_m": float(means[minimum_index]),
        "minimum_epoch": int(epochs[minimum_index]),
        "plot": str(args.output.resolve()),
        "csv": str(args.csv.resolve()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
