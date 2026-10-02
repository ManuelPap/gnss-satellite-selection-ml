#!/usr/bin/env python3
"""Plot historical-defective and corrected-GT BiasNet training curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def curve_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    losses = np.asarray(
        [float(row["loss_divided_by_405_like_released_print"]) for row in rows]
    )
    minimum = int(np.argmin(losses))
    return {
        "epoch_1_loss_m": float(losses[0]),
        "epoch_500_loss_m": float(losses[-1]),
        "minimum_loss_m": float(losses[minimum]),
        "minimum_loss_epoch": int(epochs[minimum]),
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-metrics",
        type=Path,
        default=root / "validation/paper_biasnet/training_metrics.json",
    )
    parser.add_argument(
        "--corrected-metrics", type=Path, default=here / "training_metrics.json"
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=root / "results/paper_biasnet_corrected_gt/training_loss_ab.png",
    )
    parser.add_argument("--output", type=Path, default=here / "curve_comparison.json")
    args = parser.parse_args()
    historical = json.loads(args.historical_metrics.resolve().read_text())
    corrected = json.loads(args.corrected_metrics.resolve().read_text())
    historical_rows = historical["epochs"]
    corrected_rows = corrected["epochs"]
    if len(historical_rows) != 500 or len(corrected_rows) != 500:
        raise RuntimeError("A/B curve comparison requires two 500-epoch runs")
    epochs = np.arange(1, 501)
    historical_loss = np.asarray(
        [row["loss_divided_by_405_like_released_print"] for row in historical_rows]
    )
    corrected_loss = np.asarray(
        [row["loss_divided_by_405_like_released_print"] for row in corrected_rows]
    )
    args.plot.resolve().parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(epochs, historical_loss, label="Historical duplicated GT", linewidth=1.3)
    axis.plot(epochs, corrected_loss, label="Corrected one-to-one GT", linewidth=1.3)
    axis.set_xlabel("Training epoch")
    axis.set_ylabel("Sum of epoch 3D norms / 405 [m]")
    axis.set_title("BiasNet controlled GT-mapping A/B")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.plot.resolve(), dpi=180)
    plt.close(figure)
    output = {
        "status": "passed",
        "curve_definition": "sum of 405 per-epoch 3D ENU Euclidean norms divided by 405",
        "historical_defective": curve_summary(historical_rows),
        "corrected_one_to_one": curve_summary(corrected_rows),
        "paper_figure_2_qualitative_reference": (
            "published TDL-B curve is approximately 10 m initially and 2 m late; "
            "not digitized and not used for fitting"
        ),
        "plot": str(args.plot.resolve()),
    }
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
