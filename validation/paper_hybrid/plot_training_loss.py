#!/usr/bin/env python3
"""Plot the recorded released-code TDL-BW sum/405 training curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=here / "training_metrics.json")
    parser.add_argument(
        "--output", type=Path, default=root / "results/paper_hybrid/training_loss.png"
    )
    parser.add_argument(
        "--csv", type=Path, default=root / "results/paper_hybrid/training_loss.csv"
    )
    args = parser.parse_args()
    metrics = json.loads(args.metrics.resolve().read_text())
    rows = metrics["epochs"]
    args.csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.csv.resolve().open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "loss_sum_3d_m", "mean_like_loss_m", "gradient_l2_norm", "duration_seconds"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "epoch": row["epoch"],
                "loss_sum_3d_m": row["loss_sum_3d_m"],
                "mean_like_loss_m": row["loss_divided_by_405_like_released_print"],
                "gradient_l2_norm": row["gradient_l2_norm"],
                "duration_seconds": row["duration_seconds"],
            })
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(
        [row["epoch"] for row in rows],
        [row["loss_divided_by_405_like_released_print"] for row in rows],
        label="TDL-BW released objective / 405",
    )
    plt.xlabel("Training epoch")
    plt.ylabel("Mean-like 3D position loss (m)")
    plt.title("Paper-era TDL-BW reproduction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output.resolve(), dpi=180)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
