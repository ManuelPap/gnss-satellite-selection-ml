# Paper-era shared hybrid TDL-BW reproduction

## Scope and result

This directory reproduces the executable shared bias-and-weight path in
TDL-GNSS commit `dd5eac669676ba0a922102047e58c2dfc9be9267`. The external
TDL-GNSS, pyrtklib, and TASGNSS repositories remain read-only. The validated
`src/gnss_satellite_selection_ml/differentiable_wls.py` and
`src/gnss_satellite_selection_ml/paper_observation_model.py` files are
unchanged.

The three learned paths are physically distinct:

```text
TDL-B:  features -> b [m] -> P-b -> identity-W solver
TDL-W:  features -> w     -> W   -> uncorrected-P solver

TDL-BW: features -> shared HybridShareNet -> (weight, bias)
                                               |       |
                                               v       v
                                           W=diag(w)  P-b
                                               \       /
                                                v     v
                                                  WLS
```

The deterministic seed-`20260929` run completed 100 from-scratch epochs and
froze `checkpoints/paper_hybrid/hybrid_share_3d.pth` with SHA-256
`9c39b90b1fbff7f1e066cd6f9515dc474222881e87512f54d3980a0d839739cb`.
It must be interpreted with an important negative result: this initialization
puts every released ReLU bias head output in its dead region. Bias stayed
exactly zero throughout all 100 epochs, so this particular faithful run
learned through the weight path only. No seed search, restart, checkpoint
selection, or held-out tuning was used to hide that behavior.

## Historical audit

`hybrid_provenance.json` is the machine-readable audit. The historical
training and inference entry points are `hybrid_network_train.py` and
`hybrid_network_predict.py`. Both instantiate `HybridShareNet`, not the
separate `HybridNet(WeightNet, BiasNet)` class.

### Exact released network and output semantics

```text
f_i = [SNR[0]/1000, elevation radians, equal-weight OLS residual metres]
  -> frozen StandardizeLayer
  -> Linear(3,64)   -> ReLU
  -> Linear(64,128) -> ReLU
  -> Linear(128,64) -> ReLU
  -> Linear(64,2)   -> raw[:,0], raw[:,1]

weight = clamp(sigmoid(raw[:,0]), 0, 1)
bias   = ReLU(raw[:,1])
return weight, bias
```

The executable output order is therefore **`(weight, bias)`**. This is proven
from use, not names: the trainer assigns `predict[0]` to `weight`, puts it on
the diagonal of `W`, assigns `predict[1]` to `bias`, and passes it to the
subtractive WLS argument. The implemented weight bound is `[0,1]`; sigmoid's
mathematical range is `(0,1)`, with endpoint rounding possible in finite
precision, and the clamp is normally inactive. Bias has units of metres and range `[0,+infinity)`. It is
not the linear/unbounded output suggested by a literal reading of the paper.

Positive bias has the physical sign

```text
P_bias_corrected_i = P_RTKLIB_corrected_i - b_i
W = diag(w_i)
```

and `w_i` is a dimensionless relative WLS coefficient. A deterministic unit
test and the real-epoch archived comparison both verify the sign.

### Manuscript versus released implementation

The [arXiv v1 manuscript](https://arxiv.org/abs/2409.12996) describes a shared
network, ReLU in the initial two layers, a two-output layer, sigmoid
specifically on weight, `P-b`, and the sum of bias-path and weight-path
gradients. The released computational target differs:

| Topic | Manuscript | Released `dd5eac6` |
| --- | --- | --- |
| Shared hidden layers | initial two ReLU layers | three: `64 -> 128 -> 64`, ReLU after each |
| Output order | written `(b, W)` | returned `(weight, bias)` |
| Weight | sigmoid, `(0,1)` | sigmoid then clamp `[0,1]` |
| Bias output | sigmoid only said to apply to weight, implying an untransformed bias | ReLU, hence nonnegative |
| Loss | MSE / half-squared error | sum of unsquared per-epoch 3D ENU norms |
| Learning rate | `0.0016` in arXiv v1 Table IV | `0.01` |
| State | four entries in equations | seven slots with one metre-valued clock per constellation |
| TDL-BW epochs | 100 | 100 from config |

The manuscript text says Figure 2 contains the networks' loss curves, while
its caption names only the blue TDL-B and green TDL-W curves. The released
hybrid trainer has no multi-model Figure-2 combiner. It saves its raw summed
loss to `result/hybrid_share/klt3_train/loss_100.csv` and plots the same sum;
the console alone prints sum/405. `plot_training_loss.py` creates our own
clearly labeled TDL-BW sum/405 curve without digitizing or fitting the paper.

The archived tree contains `model/hybrid_share/hybrid_share.pth`, but the
corresponding inference load is commented. The active line expects the
configured `hybrid_share_3d.pth`, which is absent. The released trainer loads
no pretrained standalone or hybrid weights, so the reproduction starts from
PyTorch's default `Linear` initialization.

## Data, features, normalization, and GT

The hybrid feature-gathering loop is scientifically identical to the
validated KLT3 path:

```text
RINEX -> satellite states/clocks -> RTKLIB-corrected pseudorange
      -> equal-weight OLS -> [C/N0, elevation, OLS residual]
```

All 8,857 KLT3 measurement rows independently reproduce:

```text
mean = [29.084904595235407,
        0.8471899979658503,
       -1.7873233713548635e-07]

population std = [5.89184205821295,
                  0.28798909935383876,
                  4.3018036109624145]
```

The program converts these NumPy float64 values to float32 tensors before
`net.double()`. KLT1/KLT2 use only the values frozen in the checkpoint; no
test normalization is fitted. Ground truth has no route into feature
construction.

The hybrid GT audit is independent of the standalone BiasNet audit. In
`hybrid_network_train.py`, lines 58–60 append exactly one target for every
epoch in the strict time window, and lines 103–109 consume `gts[i]`. There is
no second append. The released interval retains 405 GNSS epochs, builds 405
GT entries, and all 405 OLS solutions succeed. Therefore:

```text
hybrid GNSS epoch i -> hybrid GT list index i
```

For indices `0,1,2,10,100,200,400`, every GT timestamp is `+0.0039999485 s`
from its GNSS timestamp. Raw dataframe row IDs and timestamps are exported in
`gt_alignment_audit.json`. No hybrid GT defect exists, so no corrected-target
mode or second training experiment was invented.

## Solver and controlled forward equivalence

The released solver starts from the equal-weight OLS state, uses the seven
slots `[x,y,z,b_GPS,b_BDS,b_Galileo,b_GLONASS]`, stops when
`norm(delta) <= 1e-4`, and permits at most 10 iterations. It forms an explicit
inverse. The already verified archived Torch defect that supplies an
unpopulated line-of-sight vector to elevation/atmosphere calls—and therefore
produces zero atmospheric corrections—is preserved.

`compare_hybrid_path_real_epoch.py` supplies deterministic nonuniform raw
two-column outputs on real KLT1 epoch 0. It applies the transformations with
the archived class and controlled class, then compares `b`, `w`, `P-b`, raw
residual, effective residual, `H`, `W`, `H^T W H`, `H^T W v`, delta, updated
state, and final state. Every maximum absolute discrepancy, including final
state, is exactly `0.0` in `forward_equivalence.json`.

## Dual-gradient validation

`gradient_sanity_real_epoch.py` uses real KLT3 epoch 0. Seed `20260929`
initially gives bias preactivations from `-0.1295691` to `-0.0582723`, making
the released bias derivative exactly zero. For this gradient audit only, a
documented `+1 m` final-bias offset moves the exact network graph into ReLU's
active region; full training never receives this offset.

At identical output values:

| Check | Result |
| --- | ---: |
| Bias-output gradient L2 | `1.8513146656492856` |
| Weight-output gradient L2 | `20.693800547045505` |
| Bias-path shared-parameter gradient L2 | `2.393064527790362` |
| Weight-path shared-parameter gradient L2 | `7.4080888800071145` |
| Maximum scaled `grad_total - (grad_bias + grad_weight)` | `2.7755575615628914e-16` |
| Worst representative central-difference scaled error | `3.4069814540052534e-4` |

Every trainable tensor receives a finite nonzero gradient and changes after
one Adam step in this controlled audit; all outputs and solver states remain
finite. These are numerical sanity checks, not formal proofs.

## Training recipe and curve

- KLT3 only: 405 epochs / 8,857 measurement rows, chronological, no shuffle.
- One whole-dataset accumulated backward pass and Adam update per training
  epoch; `batch=128` is read but unused.
- Adam, learning rate `0.01`, default remaining Adam settings.
- Sum of 405 unsquared 3D ENU Euclidean position-error norms.
- Equal-weight OLS initialization; 10 WLS iterations maximum; `1e-4`
  tolerance.
- 100 training epochs, from scratch, local seed `20260929`. Upstream fixes no
  author seed.

Training took `85.2774 s` on CPU. Each epoch's loss sum, sum/405, gradient
norm, and duration is in `training_metrics.json`.

| Mean-like loss (sum / 405) | Value |
| --- | ---: |
| Epoch 1, pre-update | `24.565237356288375 m` |
| Epoch 100, pre-update | `5.01549282888056 m` |
| Minimum pre-update | `4.987085142407641 m` at epoch 99 |
| Final post-update | `5.15599285208663 m` |

The curve falls sharply, continues downward with oscillations, reaches its
minimum at epoch 99, and worsens on the final Adam update. Comparison with the
paper is qualitative only because the paper/code objective and learning rate
differ and the author seed/checkpoint is unavailable.

### Final KLT3 outputs

All 8,857 biases are exactly `0 m` (min, max, mean, median, standard deviation,
P5, and P95 are all zero). No additional clipping was imposed.

| Weight statistic | Value |
| --- | ---: |
| Min / max | `5.43315e-58` / `0.999829` |
| Mean / median / std | `0.0611301` / `0.00296093` / `0.138092` |
| P5 / P95 | `5.21902e-20` / `0.287426` |
| Fraction `<1e-5` / `<0.01` | `0.299650` / `0.575138` |
| Fraction `>0.5` / `>0.99` | `0.0237101` / `0.00270972` |

## Frozen held-out evaluation

The epoch-100 checkpoint is evaluated with `model.eval()` and
`torch.no_grad()`. KLT1/KLT2 do not affect training, normalization, model
selection, early stopping, or hyperparameters. No WLS failures occur. The
CSV-only checker reproduces all arithmetic means within `1e-12`.

| Dataset | Epochs / rows | Our 2D | Paper 2D | Difference | Our 3D | Paper 3D | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KLT1 | 203 / 4,676 | `2.3891` | `1.84` | `+0.5491` (`+29.84%`) | `7.9982` | `4.72` | `+3.2782` (`+69.45%`) |
| KLT2 | 209 / 4,914 | `3.1596` | `1.86` | `+1.2996` (`+69.87%`) | `10.7438` | `3.92` | `+6.8238` (`+174.08%`) |

Held-out bias is identically zero. KLT1 weights have min/max
`3.76876e-41 / 0.841574`, mean `0.0563002`, median `0.00310027`, standard
deviation `0.108125`, P5 `2.27336e-17`, and P95 `0.287229`. KLT2 weights have
min/max `2.93987e-46 / 0.992968`, mean `0.0607681`, median `0.00198606`,
standard deviation `0.133435`, P5 `8.66599e-20`, and P95 `0.303695`.

### Descriptive effective geometry

Thresholds below are diagnostics, not a definition of usable satellites.
Every final normal matrix has rank 7.

| Dataset | Mean total sats | Mean `w>0.01` | Mean `w>0.1` | Mean `w>0.5` | Median / max condition number |
| --- | ---: | ---: | ---: | ---: | ---: |
| KLT1 | `23.03` | `9.76` | `4.43` | `0.20` | `4.284e3 / 1.622e5` |
| KLT2 | `23.51` | `9.78` | `4.34` | `0.47` | `5.131e3 / 2.145e6` |

The per-epoch satellite counts, all three threshold counts, constellation
composition for `w>0.01`, rank, and condition number are in
`held_out_evaluation_summary.json` and the ignored per-epoch CSVs.

### Descriptive comparison with frozen learned baselines

No earlier model was retrained.

| Dataset | Method | 2D mean | 3D mean |
| --- | --- | ---: | ---: |
| KLT1 | Our TDL-W | `2.3730` | `9.5118` |
| KLT1 | Our corrected-GT TDL-B | `2.1032` | `5.5924` |
| KLT1 | Our TDL-BW | `2.3891` | `7.9982` |
| KLT1 | Paper TDL-W / TDL-B / TDL-BW | `2.57 / 2.24 / 1.84` | `9.92 / 5.30 / 4.72` |
| KLT2 | Our TDL-W | `2.6377` | `7.1108` |
| KLT2 | Our corrected-GT TDL-B | `2.4328` | `5.7879` |
| KLT2 | Our TDL-BW | `3.1596` | `10.7438` |
| KLT2 | Paper TDL-W / TDL-B / TDL-BW | `2.89 / 2.35 / 1.86` | `7.75 / 5.89 / 3.92` |

These comparisons are observational. The dead ReLU bias head in this seeded
run, unavailable author `hybrid_share_3d.pth` and random seed, manuscript/code
differences, unpinned historical pyrtklib version, and software/hardware
differences prevent strict model-parameter equivalence. Whampoa is not
attempted because the exact July 2021 dataset is not public.

## Commands

Run from the repository root with `PYTHONPATH=src:.`. Commands that need the
paper-era TDL-GNSS/pyrtklib runtime use the validated persistent cache at
`../external_data/.paper_runtime` by default; pass `--runtime-dir PATH` to
select another complete validated cache.

```bash
.venv/bin/python validation/paper_hybrid/audit_gt_alignment.py
.venv/bin/python validation/paper_hybrid/compare_hybrid_path_real_epoch.py
.venv/bin/python validation/paper_hybrid/gradient_sanity_real_epoch.py
.venv/bin/python validation/paper_hybrid/train_paper_hybrid.py --smoke
.venv/bin/python validation/paper_hybrid/train_paper_hybrid.py
.venv/bin/python validation/paper_hybrid/plot_training_loss.py

.venv/bin/python validation/paper_hybrid/evaluate_test_dataset.py --dataset KLT1
.venv/bin/python validation/paper_hybrid/evaluate_test_dataset.py --dataset KLT2
.venv/bin/python validation/paper_hybrid/check_exported_metrics.py \
  results/paper_hybrid/klt1_per_epoch.csv \
  --summary results/paper_hybrid/klt1_summary.json
.venv/bin/python validation/paper_hybrid/check_exported_metrics.py \
  results/paper_hybrid/klt2_per_epoch.csv \
  --summary results/paper_hybrid/klt2_summary.json
.venv/bin/python validation/paper_hybrid/consolidate_results.py
```

Manual single-epoch walkthrough:

```bash
PYTHONPATH=src:. .venv/bin/python \
  validation/paper_hybrid/manual_test_epoch.py \
  --dataset KLT1 \
  --epoch-index 0
```

It prints every PRN's RTKLIB-corrected pseudorange, raw and normalized
features, predicted bias, `P-b`, and weight; then every WLS iteration's state,
raw/effective residuals, `H`, `W`, normal matrix, right-hand side, delta, and
updated state; and finally ECEF, GT, ENU, 2D/3D error, rank, and condition
number.

Generated checkpoints, plots, binary traces, and per-epoch CSVs live under
ignored paths. They are not committed. The JSON audit/metrics summaries and
tests are reviewable text artifacts.
