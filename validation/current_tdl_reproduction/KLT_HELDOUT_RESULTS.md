# Current TDL-GNSS + TASGNSS held-out KLT results

## Scientific question

Do independently trained current TDL-GNSS `HybridShareSysNet` models, trained only on KLT3, learn pseudorange bias corrections and measurement weights that improve held-out GNSS positioning on KLT1 and KLT2 relative to the same TASGNSS solver operated without learned influence?

The controlled baseline is neutral TASGNSS, with `w_i = 1` and `b_i = 0`. The learned system is:

```text
9 operational features [SNR, elevation, azimuth, neutral_residual, G, R, E, C, J]
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
| Project | `git@github.com:ManuelPap/gnss-satellite-selection-ml.git` | `research/current-tdl-tasgnss-reproduction` at `420a0c31cb913d1ae028426fa164f651cbae15a0` |
| TDL-GNSS | `https://github.com/ebhrz/TDL-GNSS.git` | `a640b2832c90daeb1ce25644a1da91c8edeb13fc` |
| TASGNSS | `https://github.com/PolyU-TASLAB/TASGNSS.git` | `fdd7e8ebc0019ad9b7c73f31363de066290d057a` |
| pyrtklib | `https://github.com/IPNL-POLYU/pyrtklib.git` | `1c468dbe14074f1b7b3276ce265fe0fa5d6bef8b` |

The numerical source of record is the frozen checkpoint manifest, evaluation manifest, and the KLT1/KLT2 per-seed and across-seed JSON summaries under `external_data/current_tdl_reproduction`. Checkpoints and external artifacts remain untracked.

## Primary results

All errors are metres. Every displayed result is rounded to two decimal places. Learned results are arithmetic means of the ten independently trained models' per-seed summary statistics.

| Dataset | Method | 2D mean | 2D median | 2D P95 | 3D mean | 3D median | 3D P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KLT1 | Neutral TASGNSS | 2.83 | 2.51 | 6.13 | 10.93 | 9.35 | 23.22 |
| KLT1 | Current TDL-GNSS + TASGNSS (mean across 10 seeds) | 2.03 | 1.91 | 3.83 | 4.57 | 3.81 | 9.41 |
| KLT1 | Yin et al. literature reference | 2.75 | 2.51 | 5.54 | 4.95 | 4.29 | 11.37 |
| KLT2 | Neutral TASGNSS | 5.29 | 4.37 | 10.83 | 12.01 | 9.00 | 36.29 |
| KLT2 | Current TDL-GNSS + TASGNSS (mean across 10 seeds) | 2.12 | 1.82 | 4.64 | 3.87 | 3.19 | 8.79 |
| KLT2 | Yin et al. literature reference | 3.05 | 2.88 | 5.80 | 5.09 | 4.82 | 8.64 |

The Yin et al. values are an **external literature sanity reference; not a regression target**. They are contextual values from *Yin et al., Residual-Guided Hybrid Stochastic Modeling: A Two-Stage Learning Framework for Urban GNSS Positioning Enhancement, Sensors 2026, 26, 5622*. They are not tuning targets, and this experiment does not claim an exact reproduction of Yin et al.

## Paired epoch results

Paired delta is `trained TDL error - neutral TASGNSS error`, so a negative delta means improvement. Fractions are shown as percentages, also to two decimal places.

| Dataset | 2D epochs improved (%) | 2D mean delta (m) | 3D epochs improved (%) | 3D mean delta (m) |
| --- | ---: | ---: | ---: | ---: |
| KLT1 | 66.85 | -0.80 | 82.02 | -6.36 |
| KLT2 | 88.85 | -3.17 | 89.14 | -8.14 |

This comparison is stronger than comparing aggregate means alone because each learned and neutral solution is evaluated on exactly the same epoch support.

## Robustness across seeds

| Dataset | Learned 3D mean (m) | Population SD across seed summaries (m) |
| --- | ---: | ---: |
| KLT1 | 4.57 | 0.25 |
| KLT2 | 3.87 | 0.31 |

The learned mean error was lower than neutral TASGNSS for all ten independently initialized models on both held-out trajectories. The conclusion is therefore not based on one favorable seed. The aggregation unit is the ten per-seed summaries; seed×epoch observations are not pooled or treated as independent.

## Vertical component

| Dataset | Neutral Up RMS (m) | Learned mean Up RMS across seeds (m) |
| --- | ---: | ---: |
| KLT1 | 12.74 | 5.05 |
| KLT2 | 14.03 | 3.95 |

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
