# Paper-era standalone BiasNet (TDL-B) reproduction

## Scope and graph

This directory reproduces the released standalone pseudorange-bias path in
TDL-GNSS commit `dd5eac669676ba0a922102047e58c2dfc9be9267`:

```text
KLT observations
  -> equal-weight OLS
  -> per-satellite [SNR[0]/1000, elevation rad, OLS residual m]
  -> frozen KLT3 population standardization
  -> BiasNet
  -> one unbounded correction b_i [m] per retained satellite row
  -> RTKLIB-corrected pseudorange P_i - b_i
  -> equal-weight differentiable GNSS solve
  -> receiver ECEF position
  -> 3D ENU position loss against GT (training only)
```

BiasNet does **not** predict the receiver position. It changes each physical
pseudorange measurement, and the GNSS solver estimates the position from those
corrected measurements. It does not learn measurement weights and is not
TDL-BW.

For contrast:

```text
WeightNet: features -> positive weight -> diagonal W -> weighted solve
BiasNet:   features -> correction [m] -> P - b -> identity-W solve
```

The already validated WeightNet and the frozen source files
`differentiable_wls.py` and `paper_observation_model.py` are unchanged.

## Historical audit

The primary target is released code, not a silent reinterpretation of the
paper. The full machine-readable audit is in `biasnet_provenance.json`.

### Exact released network

Both `bias_network_train.py` and `bias_network_predict.py` instantiate the
class named `BiasNetTest`:

```text
fixed StandardizeLayer
  -> Linear(3, 64)   -> ReLU
  -> Linear(64, 128) -> ReLU
  -> Linear(128, 1)
```

There is no output activation, scaling, de-standardization, or clamp. Outputs
may be positive or negative. Their unit is metres because the solver directly
subtracts them from metre-valued pseudorange residuals.

The different class named `BiasNet` contains BatchNorm and output
de-standardization, but neither standalone bias script instantiates it. The
only bias checkpoint tracked at `dd5eac6`, `model/bias/biasnet.pth` (SHA-256
`0c752b6e20f5bb7d0c2bdd0f31d102e3c333017767c161b305630ec849928b9f`),
belongs to that different class and is incompatible with `BiasNetTest`.
Training and prediction instead name `biasnet_3d.pth`, which is absent from
the released tree. Therefore the released runnable checkpoint cannot be
reused and a controlled from-scratch run is required.

### Bias sign and solver

The archived solve computes

```text
v = P - predicted_observation
delta = inv(H^T W H) H^T W (v - b)
```

Thus positive `b_i` means **subtract `b_i` metres from pseudorange**:

```text
P_bias_corrected_i = P_RTKLIB_corrected_i - b_i
```

Standalone TDL-B passes no learned weight, so `W` is identity. It initializes
from the equal-weight OLS state, uses at most 10 iterations, and stops when
`norm(delta) <= 1e-4`. The seven-slot state is
`[x,y,z,b_GPS,b_BDS,b_Galileo,b_GLONASS]`, with only present constellation
clocks active.

`forward_equivalence.json` records a deterministic real KLT1 check with the
same supplied biases in the archived CPU-patched path and this controlled
adapter. It compares `P-b`, residual, `H`, identity `W`, `H^T W H`,
`H^T W v`, delta, updated state, and final state. The tiny unit test in
`test_paper_biasnet.py` independently proves the sign with two pseudoranges.

The archived Torch observation path also passes an allocated but unpopulated
line-of-sight vector to `satazel`; the resulting atmospheric terms are zero.
That verified defect is preserved by the validated shared solver.

### Features and normalization

BiasNet and WeightNet use equivalent feature-gathering loops on KLT3. An
independent recomputation over the shared 8,857-row cache gives:

```text
mean = [29.084904595235407,
        0.8471899979658503,
       -1.7873233713548635e-07]

population std = [5.89184205821295,
                  0.28798909935383876,
                  4.3018036109624145]
```

The released program converts these NumPy float64 values to float32 tensors
before calling `net.double()`. The checkpoint stores the float32-rounded
values as float64 parameters. KLT1 and KLT2 always use those frozen checkpoint
values; their statistics are never fitted or consulted. Ground truth has no
route into feature construction.

The strict KLT3 interval `(1623297151, 1623297556)` yields 405 epochs and
8,857 measurements in released code, while the paper reports 404 epochs and
8,857 measurements. Dropping either endpoint loses measurements, so no
undocumented epoch deletion is applied.

### Released training behavior

- Epochs are chronological after `sortobs()` and `split_obs()`; no shuffle.
- PyTorch default `Linear` initialization is used in float32, then the model
  is converted to float64. Upstream sets no seed. Local seed `20260929`
  selects one reproducible initialization from that released procedure.
- Adam uses learning rate `0.01` and 500 training epochs.
- The configured `batch=128` is unused. One optimizer update follows
  accumulation over all KLT3 epochs.
- Imported `MSELoss(reduction='sum')` is unused. The executed loss is the sum
  of per-epoch 3D ENU Euclidean position-error norms.
- The console prints the sum divided by `len(obss)=405`; `loss.csv` and
  `loss.png` save/plot the raw sum.
- The state dictionary is named `biasnet_3d.pth` and includes normalization.

### Released GT alignment defect

`bias_network_train.py` appends a nearest GT row before its OLS validity check
and appends the same row again after every successful OLS solve. It then uses
`gts[i]` for `i=0..404`. All KLT3 OLS solves succeed, so training epoch `i`
uses KLT3 GT row `floor(i/2)`, and the last half of the trajectory is
supervised against the first half. Prediction appends once and aligns GT
normally.

This material released-code defect is preserved and labeled. Repairing it
would be a separate ablation, not this computational reproduction.

## Manuscript versus released code

| Topic | Accepted manuscript | Released `dd5eac6` implementation |
| --- | --- | --- |
| Bias layers | `3 -> 64 -> 128 -> 1` | same dimensions via `BiasNetTest` |
| Activation | says ReLU “throughout”; output treatment not separately resolved | ReLU after hidden layers; linear output |
| Bias operation | measurement minus predicted bias | `(P - predicted) - b`, same sign |
| Loss | MSE / half squared error | sum of 3D Euclidean ENU norms |
| Learning rate | inspected accepted manuscript says `0.001` (task brief reports `0.0016`) | `0.01` |
| Epochs / optimizer | 500 / Adam | 500 / Adam |
| Loss figure | caption calls bias curve mean position loss | script plots raw sum; console prints sum/405 |
| GT alignment | implicitly per epoch | duplicate-append defect above |

No released script combines histories into the paper's multi-curve loss
figure (Figure 2 in the arXiv/task nomenclature and Figure 3 in the inspected
accepted manuscript), so
the exact figure-production procedure cannot be established.
`plot_training_loss.py` plots the independently reproduced sum/405 quantity
called “mean position loss” by the caption and exports it alongside the raw
saved sum. It does not digitize or tune against the paper curve, and no
pixel-identical claim is made.

## Recorded deterministic result

The controlled run used seed `20260929` on CPU with Python `3.12.3`, NumPy
`2.5.1`, PyTorch `2.14.0+cpu`, and pymap3d `3.2.0`. Training took
`575.6143094440049 s`. The ignored checkpoint is
`checkpoints/paper_biasnet/biasnet_3d.pth` (73,271 bytes), SHA-256:

```text
7eb8e7b0265ec3dc16519ab8151948ae3b94335a2cd643276e2b22ac93b9724d
```

The 500 values are pre-update losses, matching when upstream appends to
`vis_loss`:

| Curve quantity (sum / 405) | Value |
| --- | ---: |
| epoch 1 | 245.8710875496921 m |
| epoch 500 | 165.9937795828507 m |
| minimum | 165.617236071611 m at epoch 498 |

The curve falls quickly and then oscillates while declining more slowly. Its
downward direction is qualitatively like the paper curve, but its scale is
not: the accepted figure's blue bias curve is roughly 10 m initially and 2 m
late, whereas this released-code/defective-GT reproduction is roughly 246 m
to 166 m. The accepted figure also labels the ordinate `RMSE(m)`, its caption
says mean position loss, and the released script saves a sum of Euclidean
norms. The available code cannot reconcile those definitions.

### Frozen KLT results

| Dataset | Released epochs / measurements | Our 2D | Paper 2D | Difference (relative) | Our 3D | Paper 3D | Difference (relative) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KLT1 | 203 / 4,676 | 102.0353 m | 2.24 m | +99.7953 m (+4455.15%) | 184.7434 m | 5.30 m | +179.4434 m (+3385.73%) |
| KLT2 | 209 / 4,914 | 92.7961 m | 2.35 m | +90.4461 m (+3848.77%) | 166.7303 m | 5.89 m | +160.8403 m (+2730.74%) |

Both datasets have zero WLS failures. The CSV-only checker independently
recomputes the arithmetic means and agrees within `1e-12`. Closeness was not
used as a success criterion, and KLT1/KLT2 were not used for training,
normalization, model selection, early stopping, or hyperparameter tuning.

The large paper gap is consistent with direct evidence: the released GT
duplication defect, the missing authors' runnable checkpoint and random seed,
the paper/code loss and learning-rate discrepancies, unpinned pyrtklib, and
software/hardware differences. The first factor is especially strong because
the reproduced training objective explicitly supervises the latter KLT3
epochs against positions from the first half of the trajectory.

### Physical bias diagnostics

No clipping was applied because released `BiasNetTest` has none.

| Dataset | Min | Max | Mean | Median | Std | P5 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KLT3 training | -948.6646 m | 462.3506 m | 68.5952 m | 86.6606 m | 160.7800 m | -205.3265 m | 290.8262 m |
| KLT1 test | -708.2690 m | 407.2366 m | 88.4823 m | 98.5406 m | 130.2412 m | -147.7867 m | 297.4381 m |
| KLT2 test | -664.1732 m | 425.7987 m | 99.4101 m | 108.9577 m | 136.3138 m | -141.3884 m | 305.5501 m |

All outputs are finite. The largest KLT3 magnitude is G07 at
`[37, 0.436468, -11.877261]`, or maximum absolute feature z-score 2.761; only
one of the top ten KLT3 magnitudes exceeds 3 standard deviations. None of the
top-ten KLT1 or KLT2 corrections has any feature beyond 3 standard deviations.
Therefore the extreme corrections do not generally correspond to extreme
feature vectors. They are an unbounded learned response consistent with the
defective training target, not merely input outliers.

The real KLT3 gradient sanity check has worst scaled symmetric-finite-
difference discrepancy `2.3259179617873826e-06`. All trainable tensors receive
finite gradients and change after an Adam step. Archived versus controlled
forward comparison is exactly equal for every requested array and final state
in float64 on the frozen KLT1 epoch.

One structural detail is visible in the audit: the final Linear layer's scalar
bias has a first full-dataset gradient of only `4.53e-16` and changes by
`4.53e-10`. Adding the same correction to every row is almost exactly absorbed
by the per-constellation receiver-clock states, so this common offset is not
position-identifiable. The tensor still has a present finite gradient and the
record reports its near-zero magnitude rather than hiding it.

## Reproduction commands

Run from the repository root.

Historical provenance audit (read-only):

```bash
git -C /home/manuelpap/PhD/external_references/TDL-GNSS \
  show dd5eac669676ba0a922102047e58c2dfc9be9267:bias_network_train.py
git -C /home/manuelpap/PhD/external_references/TDL-GNSS \
  show dd5eac669676ba0a922102047e58c2dfc9be9267:bias_network_predict.py
git -C /home/manuelpap/PhD/external_references/TDL-GNSS \
  show dd5eac669676ba0a922102047e58c2dfc9be9267:model.py
git -C /home/manuelpap/PhD/external_references/TDL-GNSS \
  show dd5eac669676ba0a922102047e58c2dfc9be9267:rtk_util.py
```

Prepare/verify the scientifically identical shared KLT3 feature cache:

```bash
.venv/bin/python -m validation.ibiza_generalization.prepare_runtime

.venv/bin/python validation/paper_weightnet/prepare_klt3_features.py \
  --runtime-dir ../external_data/.paper_runtime \
  --observation /tmp/gnss-weightnet-repro/extracted/data/0610_KLT/COM38_210610_025603.obs \
  --ephemeris-glob '/tmp/gnss-weightnet-repro/extracted/data/0610_KLT/sta/hksc161d.21*' \
  --ground-truth /tmp/gnss-weightnet-repro/extracted/data/0610_KLT/20210610_100.txt \
  --dataset-archive /tmp/gnss-weightnet-repro/download/data.zip
```

Gradient smoke, finite difference, and archived forward equivalence:

```bash
PYTHONPATH=src .venv/bin/python validation/paper_biasnet/gradient_sanity_real_epoch.py
PYTHONPATH=src .venv/bin/python validation/paper_biasnet/train_paper_biasnet.py --smoke
PYTHONPATH=src .venv/bin/python validation/paper_biasnet/compare_bias_path_real_epoch.py
```

Full training, checkpoint hash, and loss curve:

```bash
PYTHONPATH=src .venv/bin/python validation/paper_biasnet/train_paper_biasnet.py
sha256sum checkpoints/paper_biasnet/biasnet_3d.pth
PYTHONPATH=src .venv/bin/python validation/paper_biasnet/plot_training_loss.py
```

Manual one-epoch physical trace:

```bash
PYTHONPATH=src .venv/bin/python validation/paper_biasnet/manual_test_epoch.py \
  --dataset KLT1 --epoch-index 0
```

All KLT1/KLT2 epochs and independent CSV-only arithmetic checks:

```bash
PYTHONPATH=src .venv/bin/python validation/paper_biasnet/evaluate_test_dataset.py --dataset KLT1
PYTHONPATH=src .venv/bin/python validation/paper_biasnet/evaluate_test_dataset.py --dataset KLT2
PYTHONPATH=src .venv/bin/python validation/paper_biasnet/check_exported_metrics.py \
  results/paper_biasnet/klt1_per_epoch.csv \
  --summary results/paper_biasnet/klt1_summary.json
PYTHONPATH=src .venv/bin/python validation/paper_biasnet/check_exported_metrics.py \
  results/paper_biasnet/klt2_per_epoch.csv \
  --summary results/paper_biasnet/klt2_summary.json
```

Generated checkpoints, per-epoch CSVs, plots, and binary traces are ignored by
Git. Small JSON audit/metrics records remain reviewable. Whampoa is not
evaluated because the exact released 2021-07-14 inputs are not public; no
different public run is substituted.

## Limitations

- The authors did not pin pyrtklib. Version 0.2.6 / commit `916d3cc8` is the
  best-supported historical hypothesis, not a proven publication dependency.
- Upstream did not publish runnable `biasnet_3d.pth` weights or a seed. This
  deterministic checkpoint is one reproducible initialization, not recovery
  of the authors' exact weights.
- The released GT defect makes its training target scientifically dubious but
  is retained for fidelity.
- The paper figure's composite pipeline is absent; only the released raw-sum
  and printed sum/405 definitions can be reproduced.
