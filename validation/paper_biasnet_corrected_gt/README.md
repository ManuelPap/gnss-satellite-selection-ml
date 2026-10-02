# Corrected-GT BiasNet controlled experiment

## Scope

This is a controlled follow-up to the preserved historical released-code
BiasNet reproduction in `validation/paper_biasnet`. It asks whether correcting
only the demonstrated duplicated-ground-truth indexing defect explains the
gap between released BiasNet and published TDL-B behavior.

Historical released behavior:

```text
GNSS training epoch i -> matched KLT3 GT for epoch floor(i/2)
```

Corrected one-to-one GT behavior:

```text
GNSS training epoch i -> matched KLT3 GT for epoch i
```

All other choices remain those of the historical reproduction: released
`BiasNetTest` dimensions `3 -> 64 -> 128 -> 1`, hidden ReLU activations,
linear unbounded metre output, `P_corrected = P_RTKLIB - b`, Adam at `0.01`,
500 epochs, seed `20260929`, chronological full-dataset updates without
shuffle, 405 KLT3 epochs / 8,857 measurements, frozen KLT3 normalization, the
sum of per-epoch 3D ENU Euclidean norms, and the same seven-state solver with
10 iterations, `1e-4` tolerance, and historical observation behavior.

## Released defect and correction

At TDL-GNSS commit `dd5eac669676ba0a922102047e58c2dfc9be9267`,
`bias_network_train.py` is module-level code. Lines 57–58 append the nearest GT
before checking the OLS result; lines 74–76 append the same row again after a
successful solve. Training then uses `gt_row = gts[i]` at line 107. Every one
of the 405 KLT3 solves succeeds, producing 810 GT entries whose source indices
begin:

```text
0, 0, 1, 1, 2, 2, 3, 3, ...
```

Consequently GNSS indices begin `0->0, 1->0, 2->1, 3->1, ...`; the target
advances at half the GNSS rate. The corrected loss in `experiment.py` changes
only the target expression to `ground_truth[epoch_index]` and asserts 405
epochs, 405 targets, identity order, and unique target indices.

`gt_alignment_audit.json` contains the raw-timestamp proof, first/middle/last
10 epochs, and selected historical-versus-corrected A/B rows. Each original GT
UTC timestamp receives the historically validated `+18 s` adjustment before
nearest-row matching.

## Commands

Run from the repository root with `PYTHONPATH=src:.`.

Inspect the exact historical duplicate appends:

```bash
git -C /home/manuelpap/PhD/external_references/TDL-GNSS \
  show dd5eac669676ba0a922102047e58c2dfc9be9267:bias_network_train.py | \
  nl -ba | sed -n '49,110p'
```

Inspect the corrected mapping and compare selected timestamps:

```bash
.venv/bin/python validation/paper_biasnet_corrected_gt/audit_gt_alignment.py \
  --ground-truth /tmp/gnss-weightnet-repro/extracted/data/0610_KLT/20210610_100.txt
```

Run the gradient smoke and full training:

```bash
.venv/bin/python validation/paper_biasnet_corrected_gt/train_corrected_biasnet.py --smoke
.venv/bin/python validation/paper_biasnet_corrected_gt/train_corrected_biasnet.py
```

Inspect A/B curves and the corrected checkpoint hash:

```bash
.venv/bin/python validation/paper_biasnet_corrected_gt/compare_training_curves.py
sha256sum checkpoints/paper_biasnet_corrected_gt/biasnet_3d.pth
```

Evaluate the untouched KLT1 and KLT2 paths:

```bash
.venv/bin/python validation/paper_biasnet/evaluate_test_dataset.py --dataset KLT1 \
  --checkpoint checkpoints/paper_biasnet_corrected_gt/biasnet_3d.pth \
  --training-metrics validation/paper_biasnet_corrected_gt/training_metrics.json \
  --output-dir results/paper_biasnet_corrected_gt
.venv/bin/python validation/paper_biasnet/evaluate_test_dataset.py --dataset KLT2 \
  --checkpoint checkpoints/paper_biasnet_corrected_gt/biasnet_3d.pth \
  --training-metrics validation/paper_biasnet_corrected_gt/training_metrics.json \
  --output-dir results/paper_biasnet_corrected_gt
```

Independently verify each CSV and consolidate comparisons:

```bash
.venv/bin/python validation/paper_biasnet/check_exported_metrics.py \
  results/paper_biasnet_corrected_gt/klt1_per_epoch.csv \
  --summary results/paper_biasnet_corrected_gt/klt1_summary.json
.venv/bin/python validation/paper_biasnet/check_exported_metrics.py \
  results/paper_biasnet_corrected_gt/klt2_per_epoch.csv \
  --summary results/paper_biasnet_corrected_gt/klt2_summary.json
.venv/bin/python validation/paper_biasnet_corrected_gt/consolidate_results.py
```

Generated checkpoints, plots, and per-epoch CSVs are under already ignored
`checkpoints/` and `results/` paths. The historical checkpoint and metrics are
never overwritten.

## Results

### Alignment audit and controlled A/B

The corrected mapping has 405 training epochs, 405 GT targets, and 405 unique
identity-ordered indices. Raw GT reconstruction agrees bit-for-bit with the
target values in the shared KLT3 cache. The timestamp audit reports:

| Mapping | First absolute offset | Last absolute offset | fitted absolute-offset growth |
| --- | ---: | ---: | ---: |
| Historical duplicated GT | 0.00399995 s | 201.99600005 s | 0.49999971 s/epoch |
| Corrected one-to-one GT | 0.00399995 s | 0.00399995 s | effectively 0 s/epoch |

For the corrected mapping, maximum, median, and P95 absolute mismatch are all
`0.0039999485 s`, below the declared `0.0041 s` tolerance. First, middle, and
last ten-row audits including original UTC timestamps, `+18 s` timestamps,
and released dataframe row indices are in `gt_alignment_audit.json`.

| Absolute timestamp mismatch | Historical duplicated GT | Corrected one-to-one GT |
| --- | ---: | ---: |
| Maximum | 201.99600005 s | 0.00399995 s |
| Median | 100.99600005 s | 0.00399995 s |
| P95 | 191.99600005 s | 0.00399995 s |

Selected A/B timestamps make the growing historical lag explicit:

| Epoch | GNSS GPST-like | Historical GT +18 s | Corrected GT +18 s | Historical offset | Corrected offset |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1623297151.006 | 1623297151.010 | 1623297151.010 | +0.004 s | +0.004 s |
| 1 | 1623297152.006 | 1623297151.010 | 1623297152.010 | -0.996 s | +0.004 s |
| 2 | 1623297153.006 | 1623297152.010 | 1623297153.010 | -0.996 s | +0.004 s |
| 10 | 1623297161.006 | 1623297156.010 | 1623297161.010 | -4.996 s | +0.004 s |
| 100 | 1623297251.006 | 1623297201.010 | 1623297251.010 | -49.996 s | +0.004 s |
| 200 | 1623297351.006 | 1623297251.010 | 1623297351.010 | -99.996 s | +0.004 s |
| 400 | 1623297551.006 | 1623297351.010 | 1623297551.010 | -199.996 s | +0.004 s |

The A/B initialization audit reset seed `20260929` before constructing each
model. Both initial state dictionaries hash to
`ee991c755bbc6083fdf66d082c7eaa99b0e34f12021e424fdf98a5ef73e2ede2`.
Optimizer signatures, the feature-array hash, normalization, solver inputs,
and training order are identical. The historical and corrected target tensor
hashes differ; the audit records no other experimental difference. KLT1 and
KLT2 appear only in the held-out list and were not used during training.

### Training curve and checkpoint

The CPU run completed 500 epochs in `409.7639 s`. Every epoch's summed loss,
sum/405 quantity, gradient norm, and duration is recorded in
`training_metrics.json`.

| Curve (sum of 405 3D norms / 405) | Epoch 1 | Epoch 500 | Minimum |
| --- | ---: | ---: | ---: |
| Historical duplicated GT | 245.8711 m | 165.9938 m | 165.6172 m at epoch 498 |
| Corrected one-to-one GT | 24.3751 m | 2.7474 m | 2.7474 m at epoch 500 |

The ignored A/B plot is
`results/paper_biasnet_corrected_gt/training_loss_ab.png`. The corrected curve
is far closer in scale and convergence trend to the published Figure 2 TDL-B
curve (roughly 10 m initially and 2 m late) than the defective curve. It is
not an exact Figure 2 reproduction: the paper curve was not digitized, the
initial scale still differs, and the paper/code loss-definition ambiguity
remains.

The distinct ignored checkpoint is
`checkpoints/paper_biasnet_corrected_gt/biasnet_3d.pth` (73,271 bytes), with
SHA-256:

```text
2f7bcbc65f8cfd6eaae649f36043b510cd6fae9ce026446fc66396968bc8b43c
```

### Corrected KLT3 bias outputs

No clipping was applied.

| KLT3 statistic | Historical defective | Corrected one-to-one |
| --- | ---: | ---: |
| Minimum | -948.66 m | -29.8261 m |
| Maximum | 462.35 m | 51.8806 m |
| Mean | 68.60 m | -1.8769 m |
| Median | 86.66 m | -1.7802 m |
| Population std | 160.78 m | 6.3013 m |
| P5 | -205.33 m | -11.6577 m |
| P95 | 290.83 m | 8.0848 m |

Correcting the GT alignment therefore causes this controlled run to stop
learning the several-hundred-metre corrections seen in the defective run.

### Frozen held-out evaluation

The epoch-500 model was frozen with `model.eval()` and `torch.no_grad()`.
KLT1/KLT2 normalization was not recalculated, and neither dataset was used for
tuning or model selection. Both datasets had zero WLS failures. An independent
CSV-only checker reproduced all means within `1e-12`.

| Dataset | Defective 2D | Corrected 2D | Paper 2D | Corrected - paper (relative) | Defective 3D | Corrected 3D | Paper 3D | Corrected - paper (relative) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KLT1 (203 / 4,676) | 102.0353 m | 2.1032 m | 2.24 m | -0.1368 m (-6.11%) | 184.7434 m | 5.5924 m | 5.30 m | +0.2924 m (+5.52%) |
| KLT2 (209 / 4,914) | 92.7961 m | 2.4328 m | 2.35 m | +0.0828 m (+3.52%) | 166.7303 m | 5.7879 m | 5.89 m | -0.1021 m (-1.73%) |

Relative to the defective run, corrected error falls by 97.94% (2D) and
96.97% (3D) on KLT1, and by 97.38% (2D) and 96.53% (3D) on KLT2.

### Interpretation and remaining limitations

This experiment matches outcome A from the predeclared interpretation rules:
the corrected curve becomes sensible and the frozen held-out metrics become
close to the published values. The duplicated-GT defect therefore explains a
large part of the released-code/paper discrepancy in this controlled setup.
It does not establish that it is the sole historical cause or recover the
authors' exact run. Residual differences remain, including up to 6.11% in the
four reported held-out metrics, the paper/code loss-definition discrepancy,
the unpublished runnable `biasnet_3d.pth` and random seed, the unpinned
historical `pyrtklib` version, and software/hardware differences.
