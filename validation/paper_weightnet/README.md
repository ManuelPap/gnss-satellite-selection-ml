# Paper-era WeightNet reproduction on real KLT data

## Scope

This directory reproduces the released paper-era learned continuous
measurement-weight chain:

```text
public KLT observations
  -> dd5eac6 / pyrtklib preprocessing
  -> [SNR[0]/1000, elevation, equal-weight OLS residual]
  -> released WeightNet
  -> continuous diagonal measurement weights
  -> released seven-slot differentiable WLS
  -> receiver position
  -> 3D ground-truth position loss
```

The primary computational target is TDL-GNSS commit
`dd5eac669676ba0a922102047e58c2dfc9be9267`. The pyrtklib dependency was not
pinned by the authors. Commit
`916d3cc8eb202718a16097cea4a5729bd6b27ac5`, package version `0.2.6`, is the
best-supported pre-publication hypothesis, but it remains uncertain and must
not be described as the proven publication dependency.

**This stage validates learned continuous measurement weighting. It does not
yet implement or validate hard/exact-k satellite selection.** There is no
Top-k operation in this reproduction.

## Source audit

The released `WeightNet` is in `model.py`. Its path is:

```text
f in R^3
 -> fixed StandardizeLayer
 -> Linear(3, 64)   -> Sigmoid
 -> Linear(64, 128) -> Sigmoid
 -> Linear(128, 64) -> Sigmoid
 -> Linear(64, 1)   -> Sigmoid
 -> multiply by 10
 -> clamp to [0, 10]
```

Because sigmoid has an open range, finite released-code outputs are strictly
between 0 and 10; the clamp is normally inactive. Input mean and population
standard deviation (`numpy.std`, `ddof=0`) are calculated over every retained
KLT3 satellite row. The released program creates those statistics as float32
Torch tensors, constructs the model in float32, and then calls `net.double()`.
This reproduction preserves that conversion order.

The four linear layers use PyTorch's default `Linear.reset_parameters()`
initialization: Kaiming-uniform weights with `a=sqrt(5)` and the associated
fan-in uniform bias. The upstream program sets no Python, NumPy, Torch, CUDA,
or deterministic-algorithm seed. For an auditable from-scratch run, the local
trainer records and applies its explicit seed before model construction; this
selects a reproducible draw from the same default initialization without
changing the initializer.

The exact source-byte SHA-256 values at `dd5eac6` are:

| Historical file | SHA-256 |
| --- | --- |
| `model.py` | `12bab5692b899e07681c5b130aaf69b54334328d3689426db1187b8594e9a2d2` |
| `weight_network_train.py` | `ee7cf1f8d8ded71845a2d97c7f303bfa3f6231c8babb4f78af2447a0f9202cd5` |
| `rtk_util.py` | `25fa8e26ff5772960e0dd33763950868aaaf1d181ebae25b3b93051e7511692c` |
| `config/weight/klt3_train.json` | `66d205340529731d5ccce75375e391a582628b6f629442aa285f6bb60bca3ff3` |

### Released training semantics

- Features are `[SNR[0]/1000, elevation radians, equal-weight OLS residual
  metres]`. No azimuth, constellation one-hot, raw pseudorange, Doppler,
  oracle label, or ground-truth value is an input.
- `prl.sortobs()` precedes `split_obs()`. Epochs are processed chronologically
  with no shuffle.
- `get_ls_pnt_pos()` supplies the equal-weight OLS features and the WLS
  initializer for each epoch.
- The differentiable state has seven slots:
  `[x, y, z, b_GPS, b_BDS, b_Galileo, b_GLONASS]`; only clocks represented in
  an epoch are active.
- The loss actually executed is the sum, over valid epochs, of the 3D
  Euclidean ENU position-error norm. Imported `MSELoss` is unused.
- One `zero_grad()`, backward pass, and Adam update occur after accumulating
  all epochs. Thus each training epoch is one full-dataset optimization step.
  The read `batch=128` configuration value is unused.
- Adam uses learning rate `0.01` and PyTorch defaults for other parameters.
- TDL-W trains for 500 epochs.
- Ground truth is the nearest public 100 Hz row after the released `+18`
  second time conversion. It enters only the final position-loss path.
- The released checkpoint is a state dictionary named `weightnet_3d.pth`.
  The frozen standardization mean/std are parameters in that state dictionary,
  so prediction constructs a default model and loading restores them.
- The upstream training and Torch WLS paths assume CUDA. The reproduction may
  run the identical architecture and equations on CPU and records the device.

## Manuscript-versus-code discrepancies

The accepted manuscript and arXiv v1 describe the same high-level three
features, OLS initialization, differentiable WLS chain, Adam optimizer, and
500 TDL-W epochs. The released implementation nevertheless differs in material
details:

| Topic | Accepted paper / arXiv description | Released `dd5eac6` code |
| --- | --- | --- |
| WeightNet layers | `3 -> 64 -> 128 -> 1` figure | `3 -> 64 -> 128 -> 64 -> 1` |
| Loss | MSE, with a half-squared-error equation | sum of per-epoch 3D Euclidean norms |
| Learning rate | `0.001` | `0.01` |
| State | four entries, receiver clock expressed in time with Jacobian `c` | seven entries, one metre-valued clock per supported constellation, Jacobian clock columns `1` |
| Weight range | final sigmoid, visually implying `(0,1)` | final sigmoid multiplied by 10, then clamped to `[0,10]` |

These differences are recorded, not reconciled. The released code is the
primary target for this milestone.

## Public KLT3 provenance and the 404/405 discrepancy

Original archive:

```text
https://www.dropbox.com/scl/fi/d3urwaquf5ema5j0unmt4/data.zip?rlkey=tuwpx9pdzqtdvoeoqwhcc5gi8&st=wh5qhg6e&dl=1
```

| Input | SHA-256 |
| --- | --- |
| `data.zip` | `2afd7b1e395f8494e6992d1e109f9e8d9d83d992bf6d446d83d91aaf46cfc721` |
| `data/0610_KLT/COM38_210610_025603.obs` | `f722557326d1d32c42e023d4e78515e885d21c8ae824e79460bef61c67b9b5c4` |
| `data/0610_KLT/20210610_100.txt` | `9f7ae89cfe4db0470e78ad1cc6b0a3209a740f5ed2f0500b6dac532cc4f58b42` |
| `sta/hksc161d.21f` | `64e8e3ec2f4a9eeb17379a499e5779d378978b834a1a1441240e487e7ce23768` |
| `sta/hksc161d.21g` | `0bddf6292d39f00845e9038f7f87dcecc403ec9944ca144b7345bcb233d8e660` |
| `sta/hksc161d.21l` | `795de82407d394097628a7a2595e284d666df3dc62c4ad5a34896d6de05b84e3` |
| `sta/hksc161d.21m` | `9583d5e061d46f3de29b6f783332dda9e7ff61d62041f2c8a031f6b45628e376` |
| `sta/hksc161d.21n` | `a5bc8ab35fe0c80f91d0e57517b495be6563385d235835fd6aa063d73bb7072c` |
| `sta/hksc161d.21o` | `7335032796f9b46176c359e8a39cc3f8496dd1f3fd6003fcfb9a794aafe88dd6` |

The released KLT3 configuration applies the strict condition
`1623297151 < t < 1623297556`. With the original observation file, the first
retained timestamp is `1623297151.006` and the last is `1623297555.006`.
Therefore:

```text
Published KLT3 metadata:
    404 epochs / 8,857 satellite measurements

Released-code reproduction:
    405 epochs / 8,857 satellite measurements

Drop first retained epoch:
    404 epochs / 8,836 satellite measurements

Drop last retained epoch:
    404 epochs / 8,835 satellite measurements
```

The lower-bound epoch legitimately satisfies the released strict predicate.
There is no evidence-supported removal that produces both published numbers.
This is treated as a paper-versus-released-code cardinality discrepancy. **No
undocumented epoch removal is applied:** all 405 epochs and all 8,857 retained
measurements are used for training.

`klt3_feature_manifest.json` is the machine-readable validation record. The
binary feature cache is local and ignored by Git.

## Recorded reproduction result

The reviewed run used local seed `20260929`. Upstream sets no seed; the local
seed was applied to Python, NumPy, and Torch before the unchanged PyTorch
default `Linear` initialization. No shuffle or data-dependent seed operation
was added.

| Quantity | Recorded value |
| --- | ---: |
| KLT3 epochs | 405 |
| Retained measurement rows | 8,857 |
| Feature mean | `[29.084904595235407, 0.8471899979658503, -1.7873233713548635e-07]` |
| Feature population std | `[5.89184205821295, 0.28798909935383876, 4.3018036109624145]` |
| Initial released loss sum | `9916.699158335115 m` |
| Epoch-500 pre-update loss sum | `2276.506767118122 m` |
| Final post-update loss sum | `2275.891293343780 m` |
| First gradient L2 norm | `104.61940766307202` |
| Epoch-500 gradient L2 norm | `83.76593365127725` |
| Training duration | `471.5554717119958 s` |
| Device | CPU |
| Python | `3.12.3` |
| NumPy | `2.5.1` |
| PyTorch | `2.14.0+cpu` |
| pymap3d | `3.2.0` |
| Checkpoint | `checkpoints/paper_weightnet/weightnet_3d.pth` |
| Checkpoint SHA-256 | `2ccb3f7efc17755499f38a2ccff6205a8a54816f4dbe984a941bf6cc4d26644c` |

The checkpoint is a local ignored artifact. `training_metrics.json` records
all 500 loss and gradient values. The one-step prerequisite in
`smoke_metrics.json` confirms that every trainable weight and bias received a
present, finite, nonzero gradient, and that one Adam step changed every
trainable tensor.

For KLT3 epoch zero (`1623297151.006`, 21 satellites), the 3D position loss was
`2.7329920815207385 m`; the gradient with respect to all predicted weights was
finite and nonzero with L2 norm `26.670070726212735`. Central differences for
the three largest weights agreed with autograd to maximum symmetric relative
error `2.719244147401465e-05`.

### KLT1 GPS NN weights

For the frozen epoch `1623296154.005`, exact row order is:

| Satellite | NN-generated weight |
| --- | ---: |
| G01 | `0.07314680266624517` |
| G03 | `1.0131400513434018e-06` |
| G07 | `1.2320675939213364` |
| G14 | `2.1208441379198444` |
| G21 | `0.013848770274251644` |
| G22 | `8.616523730726238e-07` |
| G28 | `1.394661856158215e-06` |
| G30 | `0.00621803034444976` |

All are finite and inside the released `[0,10]` bound. Both WLS paths execute
two iterations. Maximum absolute paper-versus-controlled discrepancies are:

| Quantity | Maximum absolute discrepancy |
| --- | ---: |
| Predicted observation | `0` |
| `H` | `0` |
| Residual `v` | `0` |
| Weight vector / `W` | `0` / `0` |
| `H^T W H` | `4.440892098500626e-16` |
| `H^T W v` | `3.469446951953614e-18` |
| Delta state | `8.698819442543027e-12` |
| Updated state | `0` |
| Final state | `0` |

This is numerical agreement at floating-point roundoff, including exact final
state equality for the recorded run.

## Reproduction commands

The external repositories are read-only. Create disposable copies with
`git archive`, build the pinned pyrtklib hypothesis into a temporary target,
and replace only `.to('cuda')` with `.to('cpu')` in the disposable TDL copy as
documented in `../real_klt/README.md`.

Prepare KLT3:

```bash
.venv/bin/python validation/paper_weightnet/inspect_weightnet_provenance.py

PYTHONPATH="$PYRTKLIB_SITE:$TDL_COPY" .venv/bin/python \
  validation/paper_weightnet/prepare_klt3_features.py \
  --tdl-dir "$TDL_COPY" \
  --observation "$KLT_DIR/COM38_210610_025603.obs" \
  --ephemeris-glob "$KLT_DIR/sta/hksc161d.21*" \
  --ground-truth "$KLT_DIR/20210610_100.txt" \
  --dataset-archive "$ARCHIVE"
```

Run the mandatory one-step smoke test, then the full 500-epoch training:

```bash
.venv/bin/python validation/paper_weightnet/train_paper_weightnet.py --smoke
.venv/bin/python validation/paper_weightnet/train_paper_weightnet.py
.venv/bin/python validation/paper_weightnet/gradient_sanity_real_epoch.py
```

Generate the eight KLT1 GPS weights and compare the archived and controlled
WLS paths:

```bash
PYTHONPATH="$PYRTKLIB_SITE:$TDL_COPY" .venv/bin/python \
  validation/paper_weightnet/compare_nn_weights_real_epoch.py \
  --tdl-dir "$TDL_COPY" \
  --observation "$KLT_DIR/COM38_210610_025603.obs" \
  --ephemeris-glob "$KLT_DIR/sta/hksc161d.21*"
```

## Interpretation boundary

Agreement on the KLT1 epoch establishes that, given identical learned
continuous diagonal weights and exact satellite-row alignment, the controlled
implementation reproduces the released paper-era positioning computation. It
does not reproduce a missing historical checkpoint, establish equivalence of
randomly initialized network parameters, validate full-dataset paper accuracy,
repair preserved observation-model defects, or validate hard satellite
selection.
