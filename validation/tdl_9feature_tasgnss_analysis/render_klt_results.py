"""Validate frozen KLT results and render the scientific summary.

This module only reads the completed experiment's JSON artifacts.  It does not
load model weights, preprocess data, train, select a seed, or write files.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT.parent / "external_data/current_tdl_reproduction"
DEFAULT_DOCUMENT = Path(__file__).with_name("KLT_HELDOUT_RESULTS.md")
EXPECTED_SEEDS = tuple(range(10))
EXPECTED_EPOCHS = {"KLT1": 203, "KLT2": 209}
AGGREGATION_UNIT = "ten per-seed summaries; epoch×seed rows are not pooled"
SOURCE_FILES = (
    "frozen_checkpoint_manifest.json",
    "evaluation/evaluation_manifest.json",
    "evaluation/klt1/across_seed_summary.json",
    "evaluation/klt1/per_seed_metrics.json",
    "evaluation/klt2/across_seed_summary.json",
    "evaluation/klt2/per_seed_metrics.json",
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _nested(record: Mapping[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        value = value[key]
    return value


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"{label} is {actual!r}; recomputed value is {expected!r}")


def _validate_summary(
    dataset: str,
    summary: Mapping[str, Any],
    per_seed: Sequence[Mapping[str, Any]],
) -> None:
    expected_epochs = EXPECTED_EPOCHS[dataset]
    if summary["seed_count"] != len(EXPECTED_SEEDS):
        raise RuntimeError(f"{dataset}: across-seed summary does not contain ten seeds")
    if summary["aggregation_unit"] != AGGREGATION_UNIT:
        raise RuntimeError(f"{dataset}: aggregation unit is not the ten per-seed summaries")

    expected_counts = {
        "preprocessed_epochs": expected_epochs,
        "solved_epochs": expected_epochs,
        "failed_epochs": 0,
        "feature_rejected_epochs": 0,
        "exact_matched_epoch_count": expected_epochs,
        "reconciled": True,
    }
    for index, (row, counts) in enumerate(
        zip(per_seed, summary["epoch_counts_by_seed"], strict=True)
    ):
        if row["epoch_counts"] != expected_counts or counts != expected_counts:
            raise RuntimeError(f"{dataset} seed {index}: epoch reconciliation failed")

    metric_paths = {
        **{
            f"learned_{dimension}_{stat}": ("learned", dimension, stat)
            for dimension in ("2d", "3d")
            for stat in ("mean", "median", "p95")
        },
        **{
            f"neutral_paired_{dimension}_{stat}": ("neutral_paired", dimension, stat)
            for dimension in ("2d", "3d")
            for stat in ("mean", "median", "p95")
        },
        **{
            f"delta_{dimension}_{stat}": ("paired_delta", dimension, stat)
            for dimension in ("2d", "3d")
            for stat in ("fraction_improved", "mean_delta")
        },
        "learned_up_rms": ("learned", "up_rms"),
        "neutral_paired_up_rms": ("neutral_paired", "up_rms"),
    }
    for metric_name, path in metric_paths.items():
        values = [float(_nested(row, *path)) for row in per_seed]
        aggregate = summary["metrics"][metric_name]
        _require_close(
            float(aggregate["mean"]),
            statistics.fmean(values),
            f"{dataset} {metric_name} mean",
        )
        _require_close(
            float(aggregate["population_sd"]),
            statistics.pstdev(values),
            f"{dataset} {metric_name} population SD",
        )


def load_and_validate(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Any]:
    """Read and cross-check the completed experiment's six JSON artifacts."""
    artifact_root = artifact_root.resolve()
    missing = [relative for relative in SOURCE_FILES if not (artifact_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing result artifacts under {artifact_root}: {missing}")

    frozen = _read_json(artifact_root / SOURCE_FILES[0])
    evaluation = _read_json(artifact_root / SOURCE_FILES[1])
    summaries = {
        dataset: _read_json(
            artifact_root / f"evaluation/{dataset.lower()}/across_seed_summary.json"
        )
        for dataset in EXPECTED_EPOCHS
    }
    per_seed = {
        dataset: _read_json(
            artifact_root / f"evaluation/{dataset.lower()}/per_seed_metrics.json"
        )
        for dataset in EXPECTED_EPOCHS
    }

    frozen_seeds = [int(row["seed"]) for row in frozen["seeds"]]
    if frozen["seed_count"] != len(EXPECTED_SEEDS) or frozen_seeds != list(EXPECTED_SEEDS):
        raise RuntimeError("frozen manifest does not contain exactly seeds 0-9")
    if frozen["status"] != "frozen_before_held_out_evaluation":
        raise RuntimeError("checkpoint manifest was not frozen before held-out evaluation")
    if frozen["selection_policy"] != "all ten seeds; no best-seed selection":
        raise RuntimeError("checkpoint manifest permits seed selection")
    if evaluation["evaluated_seeds"] != list(EXPECTED_SEEDS):
        raise RuntimeError("evaluation manifest does not contain exactly seeds 0-9")
    for field in (
        "all_ten_seeds_evaluated",
        "checkpoint_hashes_unchanged_after_evaluation",
    ):
        if evaluation[field] is not True:
            raise RuntimeError(f"evaluation manifest has false {field!r}")
    for field in ("checkpoint_selection", "normalization_refit", "training_updates"):
        if evaluation[field] is not False:
            raise RuntimeError(f"evaluation manifest has true {field!r}")
    if evaluation["status"] != "complete":
        raise RuntimeError("evaluation manifest is not complete")
    if evaluation["current_dataset_cardinalities"] != EXPECTED_EPOCHS:
        raise RuntimeError("held-out dataset cardinalities differ from 203/209")

    frozen_hashes = {int(row["seed"]): row["checkpoint_sha256"] for row in frozen["seeds"]}
    for dataset in EXPECTED_EPOCHS:
        rows = per_seed[dataset]
        if [int(row["seed"]) for row in rows] != list(EXPECTED_SEEDS):
            raise RuntimeError(f"{dataset}: per-seed metrics do not contain exactly seeds 0-9")
        for row in rows:
            seed = int(row["seed"])
            if row["dataset"] != dataset:
                raise RuntimeError(f"{dataset} seed {seed}: dataset label mismatch")
            if row["checkpoint_sha256"] != frozen_hashes[seed]:
                raise RuntimeError(f"{dataset} seed {seed}: checkpoint hash mismatch")
            for dimension in ("2d", "3d"):
                if float(row["paired_delta"][dimension]["mean_delta"]) >= 0.0:
                    raise RuntimeError(
                        f"{dataset} seed {seed}: {dimension} mean did not improve"
                    )
        _validate_summary(dataset, summaries[dataset], rows)

    configurations = [row["training_configuration"] for row in frozen["seeds"]]
    for seed, configuration in enumerate(configurations):
        expected = {
            "records": 404,
            "eligible_records": 404,
            "epochs": 120,
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "scheduler": None,
            "device": "cpu",
            "training_datasets": ["KLT3"],
            "held_out_datasets": ["KLT1", "KLT2"],
            "ibiza_used": False,
        }
        for key, value in expected.items():
            if configuration[key] != value:
                raise RuntimeError(f"seed {seed}: unexpected training configuration {key!r}")

    if evaluation["preprocessing"]["KLT1"]["provenance"] != frozen["provenance"]:
        raise RuntimeError("KLT1 and frozen source provenance differ")
    if evaluation["preprocessing"]["KLT2"]["provenance"] != frozen["provenance"]:
        raise RuntimeError("KLT2 and frozen source provenance differ")
    if provenance_status := frozen["provenance"]["project"]["status_short"]:
        raise RuntimeError(f"project worktree was not clean: {provenance_status}")
    for name, record in frozen["provenance"]["upstream"].items():
        if record["status_short"]:
            raise RuntimeError(f"{name} worktree was not clean")

    return {
        "artifact_root": artifact_root,
        "frozen": frozen,
        "evaluation": evaluation,
        "summaries": summaries,
        "per_seed": per_seed,
    }


def _f2(value: float) -> str:
    return f"{float(value):.2f}"


def _summary_mean(bundle: Mapping[str, Any], dataset: str, metric: str) -> float:
    return float(bundle["summaries"][dataset]["metrics"][metric]["mean"])


def _result_row(
    bundle: Mapping[str, Any], dataset: str, method: str, prefix: str
) -> str:
    values = [
        _summary_mean(bundle, dataset, f"{prefix}_{dimension}_{stat}")
        for dimension in ("2d", "3d")
        for stat in ("mean", "median", "p95")
    ]
    return f"| {dataset} | {method} | " + " | ".join(_f2(value) for value in values) + " |"


def _literature_row(bundle: Mapping[str, Any], dataset: str) -> str:
    source = bundle["evaluation"]["literature_reference"][dataset]
    values = [source[dimension][stat] for dimension in ("2d", "3d") for stat in ("mean", "median", "p95")]
    return f"| {dataset} | Yin et al. literature reference | " + " | ".join(
        _f2(value) for value in values
    ) + " |"


def render_markdown(bundle: Mapping[str, Any]) -> str:
    """Render the complete result note, with all result values at two decimals."""
    frozen = bundle["frozen"]
    evaluation = bundle["evaluation"]
    provenance = frozen["provenance"]
    project = provenance["project"]
    upstream = provenance["upstream"]

    primary_rows: list[str] = []
    for dataset in EXPECTED_EPOCHS:
        primary_rows.extend(
            (
                _result_row(bundle, dataset, "Neutral TASGNSS", "neutral_paired"),
                _result_row(
                    bundle,
                    dataset,
                    "Current TDL-GNSS + TASGNSS (mean across 10 seeds)",
                    "learned",
                ),
                _literature_row(bundle, dataset),
            )
        )

    paired_rows = []
    robustness_rows = []
    vertical_rows = []
    for dataset in EXPECTED_EPOCHS:
        fraction_2d = 100.0 * _summary_mean(bundle, dataset, "delta_2d_fraction_improved")
        fraction_3d = 100.0 * _summary_mean(bundle, dataset, "delta_3d_fraction_improved")
        delta_2d = _summary_mean(bundle, dataset, "delta_2d_mean_delta")
        delta_3d = _summary_mean(bundle, dataset, "delta_3d_mean_delta")
        paired_rows.append(
            f"| {dataset} | {_f2(fraction_2d)} | {_f2(delta_2d)} | "
            f"{_f2(fraction_3d)} | {_f2(delta_3d)} |"
        )
        learned_3d = _summary_mean(bundle, dataset, "learned_3d_mean")
        learned_3d_sd = float(
            bundle["summaries"][dataset]["metrics"]["learned_3d_mean"]["population_sd"]
        )
        robustness_rows.append(
            f"| {dataset} | {_f2(learned_3d)} | {_f2(learned_3d_sd)} |"
        )
        neutral_up = _summary_mean(bundle, dataset, "neutral_paired_up_rms")
        learned_up = _summary_mean(bundle, dataset, "learned_up_rms")
        vertical_rows.append(
            f"| {dataset} | {_f2(neutral_up)} | {_f2(learned_up)} |"
        )

    features = ", ".join(evaluation["feature_names"])
    literature = evaluation["literature_reference"]
    return f"""# Current TDL-GNSS + TASGNSS held-out KLT results

## Scientific question

Do independently trained current TDL-GNSS `HybridShareSysNet` models, trained only on KLT3, learn pseudorange bias corrections and measurement weights that improve held-out GNSS positioning on KLT1 and KLT2 relative to the same TASGNSS solver operated without learned influence?

The controlled baseline is neutral TASGNSS, with `w_i = 1` and `b_i = 0`. The learned system is:

```text
9 operational features [{features}]
    -> HybridShareSysNet
    -> per-observation weight w_i and non-negative bias b_i
    -> TASGNSS differentiable WLS
    -> position
```

Ground truth is used only for the training loss and for evaluation after position estimates exist. It is not an inference feature or solver input.

## Experimental protocol

- Training used KLT3 only, with 404 eligible epoch records per training pass.
- Ten models were trained independently with seeds 0-9 for 120 epochs each.
- Optimization used Adam, learning rate 0.01, no scheduler, and zero weight decay.
- The protocol was deterministic on CPU and used the released current TDL-GNSS/TASGNSS semantics.
- The final checkpoint from every seed was frozen before held-out evaluation. All ten were evaluated; no best-seed selection was performed.
- KLT1 and KLT2 were held out from training. Evaluation used 203 epochs for KLT1 and 209 epochs for KLT2, with zero failed and zero feature-rejected epochs for every seed.
- KLT3 normalization was retained. There was no held-out normalization refit and no training update during evaluation.
- Frozen checkpoint hashes agreed with the evaluation records and were verified unchanged after evaluation.
- Ibiza was not used in training, preprocessing, model selection, or evaluation for this experiment.

## Source provenance

The manifests capture clean source worktrees at experiment time:

| Source | Recorded repository | Branch/revision |
| --- | --- | --- |
| Project | `{project['remotes']['origin']['fetch']}` | `{project['branch']}` at `{project['head']}` |
| TDL-GNSS | `{upstream['TDL-GNSS']['remotes']['upstream']['fetch']}` | `{upstream['TDL-GNSS']['head']}` |
| TASGNSS | `{upstream['TASGNSS']['remotes']['upstream']['fetch']}` | `{upstream['TASGNSS']['head']}` |
| pyrtklib | `{upstream['pyrtklib']['remotes']['upstream']['fetch']}` | `{upstream['pyrtklib']['head']}` |

The numerical source of record is the frozen checkpoint manifest, evaluation manifest, and the KLT1/KLT2 per-seed and across-seed JSON summaries under `external_data/current_tdl_reproduction`. Checkpoints and external artifacts remain untracked.

## Primary results

All errors are metres. Every displayed result is rounded to two decimal places. Learned results are arithmetic means of the ten independently trained models' per-seed summary statistics.

| Dataset | Method | 2D mean | 2D median | 2D P95 | 3D mean | 3D median | 3D P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(primary_rows)}

The Yin et al. values are an **{literature['classification']}**. They are contextual values from *{literature['citation']}*. They are not tuning targets, and this experiment does not claim an exact reproduction of Yin et al.

## Paired epoch results

Paired delta is `trained TDL error - neutral TASGNSS error`, so a negative delta means improvement. Fractions are shown as percentages, also to two decimal places.

| Dataset | 2D epochs improved (%) | 2D mean delta (m) | 3D epochs improved (%) | 3D mean delta (m) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(paired_rows)}

This comparison is stronger than comparing aggregate means alone because each learned and neutral solution is evaluated on exactly the same epoch support.

## Robustness across seeds

| Dataset | Learned 3D mean (m) | Population SD across seed summaries (m) |
| --- | ---: | ---: |
{chr(10).join(robustness_rows)}

The learned mean error was lower than neutral TASGNSS for all ten independently initialized models on both held-out trajectories. The conclusion is therefore not based on one favorable seed. The aggregation unit is the ten per-seed summaries; seed×epoch observations are not pooled or treated as independent.

## Vertical component

| Dataset | Neutral Up RMS (m) | Learned mean Up RMS across seeds (m) |
| --- | ---: | ---: |
{chr(10).join(vertical_rows)}

The observed Up-component RMS reduction accounts for a substantial part of the 3D improvement. This is a result description, not a causal explanation of why the network reduces vertical error.

## Scientific interpretation

The independently trained current-stack `HybridShareSysNet` models learned from KLT3 transfer to the held-out KLT1 and KLT2 trajectories and substantially improve positioning relative to neutral TASGNSS on the same epoch support. Improvement occurs across all ten independently initialized models.

This establishes within-KLT held-out generalization. KLT1 and KLT2 are closely related KLT trajectories, so the evidence does not yet establish cross-dataset or cross-domain generalization. Ibiza is the necessary stronger external-domain test.

The independently trained current TDL-GNSS + TASGNSS models achieved held-out KLT1/KLT2 performance in the same general range as, and for several reported metrics numerically better than, the TDL-GNSS baseline reported by Yin et al. Differences in checkpoints, software, preprocessing, solver, and evaluation semantics prevent a claim of exact reproduction or comparative superiority.

## What this experiment answers

- Current `HybridShareSysNet` trains successfully on KLT3 under this protocol.
- The learned models improve held-out KLT1/KLT2 positioning relative to neutral TASGNSS.
- The conclusion is robust to seeds 0-9 within this experiment.
- A particularly strong improvement in the vertical component is observed.

## What this experiment does not answer

- Whether the model generalizes to an independent GNSS environment, platform, or domain.
- Whether the model generalizes to Ibiza.
- Whether it beats a strong conventional RTKLIB SPP baseline.
- Which learned feature or output is causally responsible for the improvement.
- Whether the model is suitable for exact-k satellite selection.
- Whether Yin et al.'s published results are exactly reproduced.

## Next scientific question

**Does the frozen KLT3-trained model retain its learned positioning benefit under zero-shot transfer to the independent Ibiza dataset?**

That evaluation must use all ten frozen checkpoints and preserve their hashes; perform no retraining, checkpoint selection, or normalization refit; use no Ibiza ground truth in features or solver inputs; use Ibiza ground truth only after position estimates are frozen, for evaluation; and compare against neutral TASGNSS on exactly paired epoch support.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--check",
        type=Path,
        metavar="DOCUMENT",
        help="compare rendered output with DOCUMENT instead of printing it",
    )
    arguments = parser.parse_args(argv)
    rendered = render_markdown(load_and_validate(arguments.artifact_root))
    if arguments.check is None:
        print(rendered, end="")
        return 0

    existing = arguments.check.read_text(encoding="utf-8")
    if existing == rendered:
        return 0
    difference = difflib.unified_diff(
        existing.splitlines(keepends=True),
        rendered.splitlines(keepends=True),
        fromfile=str(arguments.check),
        tofile="rendered",
    )
    print("".join(difference), end="")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
