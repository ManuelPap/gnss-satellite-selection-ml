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

## Historical held-out testing audit

The held-out target is the released computation at TDL-GNSS commit
`dd5eac669676ba0a922102047e58c2dfc9be9267`, not a reinterpretation designed
to approach the paper tables. The audit used `git show` for the exact
revisions of `weight_network_predict.py`, `rtk_util.py`, `model.py`,
`plot_result.py`, and the three `config/weight/*_predict.json` files.

### Manuscript description versus released evaluation code

The manuscript says that KLT1, KLT2, and Whampoa are held-out test datasets,
that the three features are C/N0, elevation, and OLS residual, and that the NN
weights form a diagonal WLS matrix. Its tables list dataset cardinalities and
TDL-W mean errors. It describes the positioning values as "MSE errors," but
does not define a separate held-out aggregation equation.

The released code is more specific and is the computational target here:

| Topic | Released `dd5eac6` behavior |
| --- | --- |
| Epoch bounds | Strict `t > start_time` and `t < end_time`; neither integer endpoint is inclusive. |
| Epoch formation | `prl.sortobs(obs)` followed by `rtk_util.split_obs(obs)`; observations within 0.05 s are one epoch. |
| GT parsing | pandas-style whitespace parsing after 30 header rows and before four footer rows. |
| GT clock alignment | Add exactly `18 s` to GT column 0, then select the row minimizing `abs(gt_time - observation_time)`. The first row wins an exact tie. |
| OLS rejection | Skip an epoch only when `get_ls_pnt_pos()` raises or reports false status. |
| Measurement validity | Remove missing ephemeris, zero L1 pseudorange, unsupported constellation, and below-horizon rows. Supported leading letters are GPS `G`, BeiDou `C`, Galileo `E`, and GLONASS `R`. |
| Pseudorange correction | `rtk_util.prange()` applies the released single-frequency constellation-specific code/TGD corrections. The NumPy OLS predicted observation also includes released broadcast ionosphere and Saastamoinen troposphere terms. |
| OLS | Equal-weight explicit-inverse Gauss-Newton, seven state slots, tolerance `1e-4`, at most 10 iterations, with residual-norm rejection above `1000 m`. |
| Features | `[SNR[0]/1000, final OLS elevation radians, final OLS residual metres]`, in retained satellite-row order. |
| Model load | `WeightNet()` with defaults, `double()`, `load_state_dict(weightnet_3d.pth)`, move to device, then `eval()`. Loading the state dictionary restores normalization parameters. |
| Test normalization | The checkpoint's KLT3 mean/std only. No test-set statistics are calculated. |
| Learned WLS | `W = diag(WeightNet(features))`; initialize from OLS. The archived Torch path preserves its zero-line-of-sight atmosphere behavior, so those atmospheric terms are zero in learned WLS. |
| WLS failure handling | `weight_network_predict.py` does not check the returned Torch-WLS status. A finite state returned at iteration/residual failure would still be aggregated. The evaluator reports this status but preserves inclusion. An exception would abort the historical program and is not silently filtered. |
| Evaluation frame | Estimated ECEF is converted to geodetic, then `pymap3d.geodetic2enu(estimate, ground_truth)` produces estimate-minus-GT East/North/Up. |
| 2D per epoch | `sqrt(E^2 + N^2)` metres. |
| 3D per epoch | `sqrt(E^2 + N^2 + U^2)` metres. |
| Dataset result | Arithmetic mean of the per-epoch Euclidean errors. It is not mean squared error and not root mean squared error. |
| Extra filtering | None after OLS validity. There is no outlier trimming, timestamp tuning, epoch selection, or paper-value matching. |

The exact historical inputs configured by source are:

| Dataset | Strict interval | Observation | Navigation | Ground truth |
| --- | --- | --- | --- | --- |
| KLT1 | `1623296154 < t < 1623296357` | `data/0610_KLT/COM38_210610_025603.obs` | `data/0610_KLT/sta/hksc161d.21*` | `data/0610_KLT/20210610_100.txt` |
| KLT2 | `1623296917 < t < 1623297126` | same KLT observation | same KLT navigation wildcard | same KLT GT |
| Whampoa | `1626238258 < t < 1626239511` | `data/whampoa/20210714.2.whampoa.ublox.f9p.obs` | `data/whampoa/hksc195e.21*` and `hksc195f.21*` | `data/whampoa/20210714_2_100hz.txt` |

The exact KLT archive is locally available and hash-verified. The released
KLT cardinalities reproduce the paper metadata without any adjustment:

| Dataset | Raw candidates in strict interval | First retained | Last retained | Valid epochs | Measurements | Published |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KLT1 | 203 | `1623296154.005` | `1623296356.005` | 203 | 4,676 | 203 / 4,676 |
| KLT2 | 209 | `1623296917.006` | `1623297125.006` | 209 | 4,914 | 209 / 4,914 |

There are no invalid OLS epochs in either KLT interval. The exact historical
Whampoa inputs are not present in the TDL-GNSS repository or its KLT download.
The currently linked UrbanNav Deep-Urban GNSS archive contains a different
run whose RINEX header spans 2021-05-21; it is not the configured 2021-07-14
observation and was not substituted. Consequently Whampoa cardinality and
accuracy remain unevaluated until the three exact historical input groups are
provided. This is an input-availability limitation, not an epoch filter.
The rejected public archive SHA-256 is
`2ce5ffdc984c6a814faa2c2be6c470289e869457d89f9d7a67eeedef3b171efd`;
its F9P observation SHA-256 is
`b357a8b9847d50d8cd10cd9afdc5d80a046c050c7ded1748710ce784d90ce40a`
and its header states first/last observations
`2021-05-21 06:29:06.004` / `2021-05-21 06:54:39.005` GPS.

## Manual held-out inference walkthrough

The evaluator is intentionally split into auditable pieces:

- `held_out.py` contains data preparation, frozen inference, positioning, and
  error-frame functions.
- `audit_test_dataset.py` stops after historical selection/OLS and reports
  cardinality.
- `manual_test_epoch.py` prints every numerical stage for one valid epoch.
- `evaluate_test_dataset.py` repeats the same operation and exports one CSV
  row per valid epoch plus a JSON summary.
- `check_exported_metrics.py` reads only CSV text and independently aggregates
  with `math.fsum`; it imports no GNSS, NN, or WLS implementation.

### One epoch, stage by stage

For measurement row `i`, the released learned-WLS observation model is

```text
rho_i = ||s_i - r|| + omega_e/c * (s_ix*r_y - s_iy*r_x)          [m]
z_hat_i = rho_i + b_system(i) - c*dt_satellite_i                 [m]
v_i = P_corrected_i - z_hat_i                                    [m]
```

With the active columns of the seven-slot state, one Gauss-Newton step is

```text
delta = (H^T W H)^-1 H^T W v
state_active <- state_active + delta
```

and `W = diag(w_1, ..., w_n)`. The position-column row of `H` is the released
`-(s_i-r)/rho_i`; the matching constellation clock column is one and the other
clock columns are zero.

Run the complete verbose trace once and retain it for stage-specific viewing:

```bash
PYTHONPATH=src .venv/bin/python \
  validation/paper_weightnet/manual_test_epoch.py \
  --dataset KLT1 --epoch-index 0 > /tmp/klt1_weightnet_epoch0.txt
```

`--epoch-index` counts valid epochs after the released OLS-status filter. The
following commands expose each requested stage; the named local function is
where the controlled implementation makes the operation explicit.

| Stage | Operation, units, and source function | Manual inspection command |
| --- | --- | --- |
| 1 | Select sorted/split RINEX epoch under the strict open interval; `prepare_dataset()` / historical `sortobs()` and `split_obs()`. | `sed -n '/^A\./,/^B\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 2 | Produce satellite ECEF `[X,Y,Z]` in metres and clock bias `dt` in seconds via historical `get_sat_pos()` -> `prl.satposs()`. | `sed -n '/^E\./,/^F\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 3 | Correct raw `P[0]` with constellation code/TGD terms via historical `prange()`; metres. | `sed -n '/^D\./,/^E\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 4 | Filter missing ephemeris, zero L1, unsupported constellation, and negative final OLS elevation in `get_ls_pnt_pos()` / `H_matrix_prl()`. | `sed -n '/^C\./,/^D\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 5 | Equal-weight OLS uses `delta=(H^T H)^-1 H^T v`; seven-state result is metres; historical `wls_solve()`. | `sed -n '/^F\./,/^G\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 6 | OLS residual feature is final `P_corrected - P_predicted`, metres; returned by `get_ls_pnt_pos()`. | `sed -n '/^G\./,/^H\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 7 | Assemble `[SNR[0]/1000, elevation rad, OLS residual m]` with `construct_features()`. | `sed -n '/^G\./,/^H\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 8 | Apply `(float32(x)-mean_KLT3)/std_KLT3`, dimensionless, in `normalize_with_frozen_klt3()`. | `sed -n '/^H\./,/^J\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 9 | Frozen `WeightNet` emits one dimensionless weight per row under `torch.no_grad()` in `infer_weights()`. | `sed -n '/^J\./,/^K\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 10 | Construct `W=diag(weights)` in `solve_paper_weighted_position()`. | `sed -n '/^K\./,/^L\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 11 | Iterate the equations above to `||delta|| <= 1e-4` or 10 iterations; every state, `v`, `H`, normal matrix, right-hand side, delta, and update is printed. | `sed -n '/^L\./,/^M\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 12 | Convert estimated ECEF to geodetic and then to the GT-centred ENU evaluation frame in `historical_position_error()`. | `sed -n '/^M\./,/^N\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 13 | Match the nearest GT row after `GT_time += 18 s`; first tie wins; `nearest_ground_truth()`. | `sed -n '/^B\./,/^C\./p; /^N\./,/^O\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 14 | Compute `e_2D=sqrt(E^2+N^2)` metres. | `sed -n '/^O\./,/^Q\./p' /tmp/klt1_weightnet_epoch0.txt` |
| 15 | Compute `e_3D=sqrt(E^2+N^2+U^2)` metres. | `sed -n '/^O\./,$p' /tmp/klt1_weightnet_epoch0.txt` |
| 16 | Repeat for every valid epoch and calculate `sum(e_i)/N`; `evaluate_test_dataset.py`, then independent `check_exported_metrics.py`. | `PYTHONPATH=src .venv/bin/python validation/paper_weightnet/check_exported_metrics.py results/paper_weightnet/klt1_per_epoch.csv --summary results/paper_weightnet/klt1_summary.json` |

### Concrete KLT1 epoch-zero path

The first released-code KLT1 epoch is `1623296154.005`. Its nearest GT row,
after the historical `+18 s`, is `1623296154.010`, a `GT-GNSS` difference of
approximately `+0.004999876 s`; GT is
`[22.33051288888889 deg, 114.18076400833334 deg, 19.522 m]`.

The raw observations, OLS features, corrected pseudoranges, and frozen NN
weights are:

| PRN | raw P[0] m | corrected P m | C/N0 | elevation rad | OLS residual m | NN weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G01 | 22596907.000 | 22596905.464381 | 33 | 0.833440170 | 0.744222473 | 0.057534868 |
| G03 | 23760072.656 | 23760072.097593 | 20 | 0.654060643 | 2.859705761 | 0.000000978 |
| G07 | 23622270.967 | 23622274.317442 | 36 | 0.588319886 | -1.740558967 | 0.690315969 |
| G14 | 22831645.581 | 22831647.954230 | 26 | 0.897269899 | -3.608088851 | 2.208093780 |
| G21 | 25328714.365 | 25328717.436238 | 27 | 0.441198293 | -7.776705712 | 0.043703426 |
| G22 | 24282547.775 | 24282553.079866 | 13 | 0.577241657 | -0.113327671 | 0.000001756 |
| G28 | 24261265.817 | 24261269.167442 | 27 | 0.614893948 | 6.895672865 | 0.000001388 |
| G30 | 22924670.463 | 22924669.346186 | 22 | 0.878052383 | 2.738969628 | 0.000109378 |
| R09 | 21877454.823 | 21877454.823000 | 31 | 0.857424776 | -0.000011008 | 0.174664714 |
| E13 | 25210682.564 | 25210682.564000 | 23 | 1.110563663 | -0.097241338 | 9.970740882 |
| E15 | 28514403.352 | 28514403.352000 | 20 | 0.285868686 | 4.828844603 | 0.000002258 |
| E21 | 25555152.045 | 25555152.045000 | 28 | 1.103504048 | 2.892015874 | 9.913858897 |
| E26 | 25636560.222 | 25636560.222000 | 21 | 0.832779724 | -6.300792120 | 0.370613554 |
| E27 | 25625189.064 | 25625189.064000 | 28 | 1.001825701 | -1.322896171 | 9.832538867 |
| C07 | 37618120.084 | 37618115.617092 | 24 | 1.215719575 | 3.141104072 | 9.921342488 |
| C11 | 24323156.313 | 24323155.053872 | 27 | 0.848902318 | -5.524903052 | 2.041749953 |
| C13 | 38486352.451 | 38486355.209091 | 21 | 0.931644955 | 2.383768678 | 0.000953422 |

The OLS seven-state initializer is

```text
[-2417851.904967737, 5384797.795507054, 2408320.922378853,
  1581584.695039450, 1581603.296534870, 1581587.614464301,
  1581593.439032623] m
```

The KLT3 source statistics are mean
`[29.084904595235407, 0.8471899979658503, -1.7873233713548635e-07]` and
population std
`[5.89184205821295, 0.28798909935383876, 4.3018036109624145]`. Preserving the
released float32-then-double construction makes the checkpoint values
`[29.084903717041016, 0.8471900224685669, -1.7873233559839719e-07]` and
`[5.891841888427734, 0.2879891097545624, 4.3018035888671875]`.

After three learned-WLS iterations, the final ECEF position is
`[-2417845.713594851, 5384767.019720683, 2408313.351380184] m`. The matched GT
ECEF is `[-2417843.166647466, 5384779.404821256, 2408314.139888462] m`.
The estimate-minus-GT ENU error is
`[7.396617393, 3.167048847, -9.785665804] m`, giving `8.046126227 m` 2D and
`12.668835878 m` 3D error. The verbose command prints all satellite ECEF and
clock values, normalized rows, and matrices without abbreviation.

### Full manual command set

From the repository root:

1. Verify the immutable checkpoint:

   ```bash
   sha256sum checkpoints/paper_weightnet/weightnet_3d.pth
   ```

2. Inspect both source and checkpoint-stored KLT3 normalization values:

   ```bash
   PYTHONPATH=src .venv/bin/python -c "from validation.paper_weightnet.held_out import KLT3_FEATURE_MEAN,KLT3_FEATURE_POPULATION_STD,FROZEN_MODEL_MEAN,FROZEN_MODEL_STD; print(KLT3_FEATURE_MEAN); print(KLT3_FEATURE_POPULATION_STD); print(FROZEN_MODEL_MEAN); print(FROZEN_MODEL_STD)"
   ```

3. Inspect released-code cardinality without running NN WLS:

   ```bash
   PYTHONPATH=src .venv/bin/python validation/paper_weightnet/audit_test_dataset.py --dataset KLT1
   PYTHONPATH=src .venv/bin/python validation/paper_weightnet/audit_test_dataset.py --dataset KLT2
   PYTHONPATH=src .venv/bin/python validation/paper_weightnet/audit_test_dataset.py --dataset Whampoa --data-root /path/to/exact/historical/data
   ```

4. Run one KLT1 epoch verbosely:

   ```bash
   PYTHONPATH=src .venv/bin/python validation/paper_weightnet/manual_test_epoch.py --dataset KLT1 --epoch-index 0
   ```

5. Evaluate KLT1:

   ```bash
   PYTHONPATH=src .venv/bin/python validation/paper_weightnet/evaluate_test_dataset.py --dataset KLT1
   ```

6. Evaluate KLT2:

   ```bash
   PYTHONPATH=src .venv/bin/python validation/paper_weightnet/evaluate_test_dataset.py --dataset KLT2
   ```

7. Evaluate Whampoa once the exact 2021-07-14 files are available:

   ```bash
   PYTHONPATH=src .venv/bin/python validation/paper_weightnet/evaluate_test_dataset.py --dataset Whampoa --data-root /path/to/exact/historical/data
   ```

8. Inspect generated ignored CSV/JSON outputs:

   ```bash
   sed -n '1,6p' results/paper_weightnet/klt1_per_epoch.csv
   .venv/bin/python -m json.tool results/paper_weightnet/klt1_summary.json
   ```

9. Independently recompute metrics from CSV only and require `1e-12`
   agreement with the evaluator summary:

   ```bash
   PYTHONPATH=src .venv/bin/python validation/paper_weightnet/check_exported_metrics.py \
     results/paper_weightnet/klt1_per_epoch.csv \
     --summary results/paper_weightnet/klt1_summary.json
   ```

Generated per-epoch files remain under the ignored `results/` directory. The
small reviewed cross-dataset record is `held_out_evaluation_summary.json`.

## Held-out results available from exact inputs

| Dataset | Released-code epochs | Reproduction 2D mean m | Paper TDL-W 2D m | Reproduction - paper m | Absolute relative difference | Reproduction 3D mean m | Paper TDL-W 3D m | Reproduction - paper m | Absolute relative difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KLT1 | 203 | 2.373042477 | 2.57 | -0.196957523 | 7.6637% | 9.511807439 | 9.92 | -0.408192561 | 4.1148% |
| KLT2 | 209 | 2.637716454 | 2.89 | -0.252283546 | 8.7295% | 7.110819014 | 7.75 | -0.639180986 | 8.2475% |
| Whampoa | not available | not evaluated | 16.11 | not evaluated | not evaluated | not evaluated | 42.49 | not evaluated | not evaluated |

Equivalently, paper minus reproduction is `+0.196957523 m` 2D and
`+0.408192561 m` 3D on KLT1, and `+0.252283546 m` 2D and `+0.639180986 m` 3D
on KLT2. These are objective numerical comparisons; no success label is
assigned. The independent CSV checker agrees within `1e-12` for both datasets.
There are zero failed learned-WLS epochs and zero excluded OLS epochs in both
KLT sets.

The same-epoch equal-weight OLS sanity results are `2.798307553 m` 2D /
`11.257174950 m` 3D for KLT1 and `5.218077033 m` 2D / `12.184638961 m` 3D for
KLT2. They are the released OLS initializer, not reproductions of the paper's
RTKLIB or goGPS baselines. Those baselines involve separate weighting and
configuration, so their unverified settings are not guessed here.

## Interpretation boundary

Agreement on the KLT1 epoch establishes that, given identical learned
continuous diagonal weights and exact satellite-row alignment, the controlled
implementation reproduces the released paper-era positioning computation. It
does not reproduce a missing historical checkpoint, establish equivalence of
randomly initialized network parameters, validate full-dataset paper accuracy,
repair preserved observation-model defects, or validate hard satellite
selection.
