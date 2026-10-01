# Paper-era TDL-GNSS WLS reproduction on one KLT1 epoch

## Scientific purpose and non-goals

This validation makes the previously audited, paper-era real-data experiment
repeatable without inline Python. It takes the original public KLT1 rover
observation, runs the paper-era `pyrtklib` preprocessing, recomputes the OLS
initializer from the same GPS-only measurements, and records every algebraic
quantity in the archived Torch WLS iterations. A separate NumPy-only program
then checks

```text
N = H.T @ W @ H
g = H.T @ W @ v
delta = numpy.linalg.solve(N, g)
```

against the reference quantities exported from the legacy explicit-inverse
implementation.

This experiment does **not** train or load a neural network. It does not assess
learned weights, positioning accuracy against ground truth, or equivalence to
`src/gnss_satellite_selection_ml/differentiable_wls.py`. It also does not repair any
paper-era observation-model behavior.

## Frozen revisions and provenance limitation

The inputs to the reproduction are:

- TDL-GNSS commit
  `dd5eac669676ba0a922102047e58c2dfc9be9267`.
- pyrtklib commit
  `916d3cc8eb202718a16097cea4a5729bd6b27ac5`, reporting package version
  `0.2.6`.
- This repository's frozen synthetic baseline at
  `7edcb9266953e48c1f21076fc0d9a0263d95399b`.

**pyrtklib 0.2.6 is a provenance hypothesis, not a proven publication
dependency.** TDL-GNSS did not pin the exact pyrtklib revision/version used for
the publication. Commit `916d3cc8...` is the current best match and produced
the validated result, but that uncertainty must remain attached to any result.

## Historical KLT1 data and hashes

The original public archive is:

```text
https://www.dropbox.com/scl/fi/d3urwaquf5ema5j0unmt4/data.zip?rlkey=tuwpx9pdzqtdvoeoqwhcc5gi8&st=wh5qhg6e&dl=1
```

Archive SHA-256:

```text
2afd7b1e395f8494e6992d1e109f9e8d9d83d992bf6d446d83d91aaf46cfc721  data.zip
```

Files used under `data/0610_KLT/`:

| File | Role | SHA-256 |
| --- | --- | --- |
| `COM38_210610_025603.obs` | Original KLT1 rover observations | `f722557326d1d32c42e023d4e78515e885d21c8ae824e79460bef61c67b9b5c4` |
| `sta/hksc161d.21f` | BeiDou navigation | `64e8e3ec2f4a9eeb17379a499e5779d378978b834a1a1441240e487e7ce23768` |
| `sta/hksc161d.21g` | GLONASS navigation | `0bddf6292d39f00845e9038f7f87dcecc403ec9944ca144b7345bcb233d8e660` |
| `sta/hksc161d.21l` | Galileo navigation | `795de82407d394097628a7a2595e284d666df3dc62c4ad5a34896d6de05b84e3` |
| `sta/hksc161d.21m` | Meteorological data in the historical wildcard | `9583d5e061d46f3de29b6f783332dda9e7ff61d62041f2c8a031f6b45628e376` |
| `sta/hksc161d.21n` | GPS navigation | `a5bc8ab35fe0c80f91d0e57517b495be6563385d235835fd6aa063d73bb7072c` |
| `sta/hksc161d.21o` | HKSC observation file in the historical wildcard | `7335032796f9b46176c359e8a39cc3f8496dd1f3fd6003fcfb9a794aafe88dd6` |

The script deliberately passes the complete
`sta/hksc161d.21*` wildcard through the paper-era reader, even though the
selected solution is GPS-only. The locally re-exported 2025 `test.obs` is not
accepted as a substitute: the reproducer verifies the original rover file's
hash.

## Manual reproduction

Run these commands from the root of this repository. They use only
repository-relative reference paths and a configurable temporary directory.
The project virtual environment must already contain the runtime packages used
by the repository, notably NumPy and CPU Torch.

### 1. Set paths and download the archive

```bash
REPO_DIR="$PWD"
WORK_DIR="/tmp/gnss-klt-paper-wls"
ARCHIVE="$WORK_DIR/data.zip"
EXTRACT_DIR="$WORK_DIR/extracted"
TDL_COPY="$WORK_DIR/TDL-GNSS-dd5eac6"
PYRTKLIB_SRC="$WORK_DIR/pyrtklib-0.2.6-src"
PYRTKLIB_SITE="$WORK_DIR/pyrtklib-0.2.6-site"

mkdir -p "$WORK_DIR" "$EXTRACT_DIR" "$TDL_COPY" "$PYRTKLIB_SRC" "$PYRTKLIB_SITE"

curl -L \
  'https://www.dropbox.com/scl/fi/d3urwaquf5ema5j0unmt4/data.zip?rlkey=tuwpx9pdzqtdvoeoqwhcc5gi8&st=wh5qhg6e&dl=1' \
  -o "$ARCHIVE"
```

### 2. Verify and extract the historical data

```bash
echo "2afd7b1e395f8494e6992d1e109f9e8d9d83d992bf6d446d83d91aaf46cfc721  $ARCHIVE" | sha256sum -c -
unzip -q "$ARCHIVE" -d "$EXTRACT_DIR"

KLT_DIR="$EXTRACT_DIR/data/0610_KLT"

echo "f722557326d1d32c42e023d4e78515e885d21c8ae824e79460bef61c67b9b5c4  $KLT_DIR/COM38_210610_025603.obs" | sha256sum -c -
echo "64e8e3ec2f4a9eeb17379a499e5779d378978b834a1a1441240e487e7ce23768  $KLT_DIR/sta/hksc161d.21f" | sha256sum -c -
echo "0bddf6292d39f00845e9038f7f87dcecc403ec9944ca144b7345bcb233d8e660  $KLT_DIR/sta/hksc161d.21g" | sha256sum -c -
echo "795de82407d394097628a7a2595e284d666df3dc62c4ad5a34896d6de05b84e3  $KLT_DIR/sta/hksc161d.21l" | sha256sum -c -
echo "9583d5e061d46f3de29b6f783332dda9e7ff61d62041f2c8a031f6b45628e376  $KLT_DIR/sta/hksc161d.21m" | sha256sum -c -
echo "a5bc8ab35fe0c80f91d0e57517b495be6563385d235835fd6aa063d73bb7072c  $KLT_DIR/sta/hksc161d.21n" | sha256sum -c -
echo "7335032796f9b46176c359e8a39cc3f8496dd1f3fd6003fcfb9a794aafe88dd6  $KLT_DIR/sta/hksc161d.21o" | sha256sum -c -
```

### 3. Create disposable source copies at the pinned commits

```bash
git -C ../external_references/TDL-GNSS \
  archive --format=tar dd5eac669676ba0a922102047e58c2dfc9be9267 |
  tar -xf - -C "$TDL_COPY"

git -C ../external_references/pyrtklib \
  archive --format=tar 916d3cc8eb202718a16097cea4a5729bd6b27ac5 |
  tar -xf - -C "$PYRTKLIB_SRC"
```

`git archive` is important here: it selects the exact revision without
checking out, modifying, or otherwise disturbing either external reference
repository. Those repositories are read-only evidence. The copies under
`/tmp` are disposable work products.

### 4. Build pyrtklib 0.2.6 into the temporary tree

```bash
.venv/bin/python -m pip install \
  --no-deps \
  --no-build-isolation \
  --target "$PYRTKLIB_SITE" \
  "$PYRTKLIB_SRC"

PYTHONPATH="$PYRTKLIB_SITE" \
  .venv/bin/python -c \
  'import importlib.metadata; print(importlib.metadata.version("pyrtklib"))'
```

The version command must print `0.2.6`. A compiler and the build requirements
already present in the project environment are needed; no pyrtklib files are
installed into the repository or global Python environment.

### 5. Apply only the CPU device substitution to the disposable TDL copy

```bash
sed -i "s/\.to('cuda')/.to('cpu')/g" "$TDL_COPY/rtk_util.py"
```

This substitution changes device placement only. It does not change equations,
constants, data flow, convergence logic, or the explicit inverse. Never apply
it to `../external_references/TDL-GNSS`.

### 6. Generate the trace and validate its algebra

Start from clean generated outputs:

```bash
rm -f \
  validation/real_klt/paper_epoch_trace.npz \
  validation/real_klt/paper_epoch_manifest.json \
  validation/real_klt/algebra_validation.json
```

Run the paper-era trace:

```bash
PYTHONPATH="$PYRTKLIB_SITE:$TDL_COPY" \
  .venv/bin/python validation/real_klt/reproduce_paper_wls_epoch.py \
  --tdl-dir "$TDL_COPY" \
  --observation "$KLT_DIR/COM38_210610_025603.obs" \
  --ephemeris-glob "$KLT_DIR/sta/hksc161d.21*" \
  --dataset-archive "$ARCHIVE" \
  --output-dir validation/real_klt
```

Run the independent NumPy validation:

```bash
.venv/bin/python validation/real_klt/validate_paper_wls_algebra.py \
  --trace validation/real_klt/paper_epoch_trace.npz \
  --output validation/real_klt/algebra_validation.json
```

The validator never imports TDL-GNSS, pyrtklib, or Torch and never calls
`wls_solve_torch()`. It exits non-zero if any configured tolerance, rank,
conditioning, residual identity, or row-alignment check fails.

## Selected epoch, state, and deterministic weights

The selection is the first valid GPS-only epoch satisfying the historical
strict condition `t > 1623296154`:

- GPST-like timestamp: `1623296154.005`.
- Human rendering of that GPST-like scalar:
  `2021-06-10 03:35:54.005`.
- Index after `split_obs()`: `2382`.
- Satellite order: `G01 G03 G07 G14 G21 G22 G28 G30`.
- Source rows: `0 1 2 3 4 5 6 7`.

The fractional `.005` is why the selected epoch is 03:35:54.005 even though
the initial search target was described approximately as 03:35:55.

The legacy state allocates seven entries:

```text
[x, y, z, b_GPS, b_BDS, b_Galileo, b_GLONASS]
```

Because this epoch is GPS-only, the active state indices at both iterations are
`[0, 1, 2, 3]`. The three non-GPS clock entries stay zero. The GPS-only OLS
initializer is:

```text
[-2417850.6248949184,
  5384790.762720588,
  2408319.1245824904,
  1581580.6444318756,
  0.0, 0.0, 0.0]
```

The supplied positive, non-uniform measurement weights are mapped by satellite:

| Satellite | Weight |
| --- | ---: |
| G01 | 0.5 |
| G03 | 0.7 |
| G07 | 0.9 |
| G14 | 1.1 |
| G21 | 1.3 |
| G22 | 1.5 |
| G28 | 1.7 |
| G30 | 1.9 |

They are deliberately not neural-network predictions. A fixed mapping isolates
the WLS algebra, makes the exact same `W` available to both implementations,
and avoids introducing a checkpoint or learned-model provenance question. The
script verifies satellite-row/weight alignment before solving.

## Observation and normal equations

For GPS satellite `i`, the archived Torch path predicts the corrected
pseudorange as

```text
P_hat_i = rho_i + sagnac_i + b_GPS - c * dt_sat_i + I_i + T_i
v_i     = P_corrected_i - P_hat_i
```

`P_corrected` is the raw L1 pseudorange after paper-era `prange()` corrections,
including the GPS broadcast group-delay/DCB path. `rho` is geometric range;
the archived Sagnac sign/convention is retained; `b_GPS` is in metres; and the
satellite clock bias is converted to metres as `-c * dt_sat`. No learned bias
is supplied, so the effective residual equals the raw residual.

At each iteration, the archived solver computes the update using an explicit
matrix inverse:

```text
delta_reference = inverse(H.T @ W @ H) @ H.T @ W @ v
```

The independent validator instead uses:

```text
N = H.T @ W @ H
g = H.T @ W @ v
delta_independent = numpy.linalg.solve(N, g)
```

and applies `delta` to the exported active state indices.

## Preserved legacy behavior

The reproducer intentionally preserves all of the following:

- The Torch path passes an allocated but unpopulated line-of-sight vector to
  `satazel()`, so Torch azimuth/elevation is zero for every row.
- The resulting ionosphere delay `I_i` is zero.
- The resulting troposphere delay `T_i` is zero.
- Values returned as `vion` and `vtrp` are variances, not delays. The trace
  stores them under `*_variance_m2` as well as retaining the legacy-return
  arrays for auditability.
- Predicted observations are reconstructed from geometric range, the archived
  Sagnac term, receiver clock, satellite-clock correction, and the zero
  atmosphere delays.
- The seven-entry state formulation and GPS-only active substate.
- The convergence test `torch.norm(delta) > 0.0001`, ten-iteration limit,
  1000 m residual rejection, explicit inverse, and final legacy NumPy
  observation-model call.
- The residual-row construction `list(set(...))`. The trace aborts if that
  order differs from the Jacobian compact-row order rather than hiding the
  paper-era ordering hazard.

CPU execution changes only literal device placement in the disposable source.
None of these behaviors is repaired.

## Generated artifacts

- `paper_epoch_trace.npz` is the binary numerical record. It contains epoch
  time, satellite IDs, source rows, ECEF satellite positions, satellite clock
  biases/corrections, raw and corrected pseudoranges, OLS initialization,
  deterministic weights, and final state. For every WLS iteration it contains
  active indices, `H`, predicted observations, raw/effective residuals, `W`,
  `H.T W H`, `H.T W v`, delta, state before/after, row maps, and decomposed
  observation-model terms.
- `paper_epoch_manifest.json` is the human-readable provenance and run
  summary: revisions, input hashes, runtime versions, selected epoch/order,
  OLS/final states, row-alignment result, and assertions about preserved legacy
  behavior.
- `algebra_validation.json` is the independent NumPy comparison. It records
  rank, condition number, per-quantity discrepancies, tolerances, and an
  overall pass/fail result.

The NPZ is ignored by Git because it is a generated binary. The two JSON files
are reviewable generated summaries. The downloaded archive, extracted data,
and temporary source/build trees must not be committed or vendored.

## Expected numerical result and tolerances

The paper-era trace converges in two iterations. Expected active-state updates:

```text
iteration 0:
[-8.254053364817, 14.184199488487, 5.861047100540, 19.533765379045]

iteration 1:
[ 6.557240267657e-05,  3.432869360509e-05,
 -1.013433875929e-06, -8.264546985437e-06]
```

Expected final state:

```text
[-2417858.878882711,
  5384804.946954405,
  2408324.985628578,
  1581600.178188990,
  0.0, 0.0, 0.0]
```

Both normal matrices have rank 4. Their expected 2-norm condition numbers are
approximately `225.283559` and `225.283006`. The documented absolute
tolerances are:

| Check | Tolerance |
| --- | ---: |
| `H.T @ W @ H` | `5e-15` |
| `H.T @ W @ v` | `1e-12` |
| `delta`, solve versus legacy inverse | `1e-12` |
| Updated state | `1e-12` |
| Maximum condition number | `1e12` |

On the validated runtime, the largest observed discrepancies were about
`1.78e-15` for `N`, at most floating-point roundoff for `g`, below `4e-13`
for `delta`, and exactly zero for the stored updated state. Small last-digit
changes within these tolerances can result from numerical-library/runtime
differences.

## What a pass proves—and does not prove

A pass proves that, for this exact real KLT1 GPS-only epoch, fixed row ordering,
OLS initializer, and supplied `W`:

1. the archived TDL-GNSS Torch path produces a complete, internally consistent
   iteration trace; and
2. independent NumPy float64 normal-equation algebra reproduces its normal
   matrix, right-hand side, update, and updated state within the stated
   tolerances.

It does not prove that the legacy observation model is physically correct,
that the publication used this exact unpinned pyrtklib revision, that a neural
network predicts useful weights, or that this repository's current
differentiable solver matches the paper-era `H` and `v`. A complete comparison
still needs an explicit observation-model adapter for Sagnac, satellite clock,
GPS TGD/DCB-corrected observations, the legacy atmosphere behavior, and the
seven-slot/multi-clock state convention.

## Controlled observation-model equivalence

After validating the paper-era WLS algebra independently with NumPy, a separate
controlled implementation of the paper-era GPS observation model was created in:

```text
src/gnss_satellite_selection_ml/paper_observation_model.py
```

The implementation was compared against the previously frozen KLT1 reference
trace for the same GPS-only epoch.
The purpose of this comparison is to verify that the controlled implementation
reproduces the quantities that appear before the normal-equation solve, namely:

- predicted pseudorange
- Jacobian H
- residual vector v

and consequently reproduces the same WLS update.
The controlled implementation does not call the archived TDL-GNSS
H_matrix_prl_torch() or wls_solve_torch() functions at runtime. The frozen
trace is used only as the numerical reference.

### Observation model reproduced

For GPS satellite \(i\), the reference-compatible observation model is:

\[
\hat{P}_i
=
\rho_i
+
S_i
+
b_{\mathrm{GPS}}
-
c\,\delta t_{s,i}
+
I_i
+
T_i
\]

with residual:

\[
v_i
=
P_{i,\mathrm{corrected}}
-
\hat{P}_i .
\]

For this legacy paper-era Torch path:

\[
I_i = 0,
\qquad
T_i = 0.
\]

The implementation preserves the historical behavior rather than repairing it.
The following terms are reproduced explicitly:

- geometric range \(\rho_i\);
- paper-era Sagnac correction \(S_i\);
- GPS receiver clock bias \(b_{\mathrm{GPS}}\);
- satellite clock correction \(-c\,\delta t_{s,i}\);
- prange()-corrected L1 pseudorange;
- the historical Jacobian convention;
- zero ionosphere and troposphere delays for the preserved legacy path;
- the active GPS state ordering:

```text
[x, y, z, b_GPS]
```

within the seven-slot historical state:

```text
[x, y, z, b_GPS, b_BDS, b_Galileo, b_GLONASS]
```

### Numerical comparison

The controlled implementation was evaluated against the frozen paper-era trace
for both WLS iterations.
Maximum observed discrepancies were:

| Quantity | Maximum absolute discrepancy |
| --- | ---: |
| Jacobian H | 0.0 |
| Predicted observation | 0.0 m |
| Residual v | 0.0 m |
| Updated state | 0.0 |
| H.T @ W @ H | 1.7763568394002505e-15 |
| State increment delta | 5.115907697472721e-13 |

All differences are within the explicit float64 numerical tolerances used by
the validation.
The result is therefore:

```text
paper observation-model comparison PASSED
```

This demonstrates that, for the validated real KLT1 GPS-only epoch, the
controlled implementation reproduces the paper-era forward observation model
and WLS state update within float64 numerical precision.

### Automated validation

The comparison is implemented in:

```text
validation/real_klt/compare_observation_model.py
```

and the observation model is covered by:

```text
tests/test_paper_observation_model.py
```

The repository test suite must be run with the project virtual environment:

```bash
.venv/bin/python -m pytest -q
```

Validated result:

```text
20 passed
```

Bare:

```bash
pytest -q
```

may invoke the system Python interpreter instead of the project virtual
environment. On the validated machine, the system interpreter does not contain
the existing Torch dependency, so collection fails before the tests execute.
This is an environment-selection issue, not a failure of the observation-model
validation.

### What this validation proves

For the selected real KLT1 GPS-only epoch, the following chain is now
independently reproduced:

```text
real corrected GNSS observations
        |
        v
satellite positions and clocks
        |
        v
controlled paper-era observation model
        |
        +--> predicted observation
        +--> Jacobian H
        +--> residual v
        |
        v
H.T @ W @ H
H.T @ W @ v
        |
        v
WLS state increment
        |
        v
updated receiver state
```

The validation therefore extends the previous real-data algebra check from:

```text
H, v, W
  |
  v
WLS algebra
```

to:

```text
real GNSS inputs
  |
  v
controlled observation model
  |
  v
H, v, W
  |
  v
WLS algebra
```

### What this validation does not prove

This result does not yet validate:

- neural-network-generated measurement weights;
- the paper-era WeightNet training process;
- end-to-end neural-network gradients on real GNSS data;
- positioning accuracy across the full KLT1, KLT2, KLT3, or Whampoa datasets;
- multi-constellation operation with multiple receiver clock states;
- NLOS classification;
- hard or exact-\(k\) satellite selection;
- equivalence of a corrected physical atmosphere model to the preserved
  paper-era legacy behavior.

The next milestone is to replace the supplied deterministic measurement weights
with weights generated by the paper-era neural-network architecture on real
KLT data and validate the resulting end-to-end positioning chain.

## Cleanup

After retaining any reports needed for review:

```bash
rm -f \
  validation/real_klt/paper_epoch_trace.npz \
  validation/real_klt/paper_epoch_manifest.json \
  validation/real_klt/algebra_validation.json

rm -rf /tmp/gnss-klt-paper-wls
```

The second command targets only the named disposable directory created above.

## Troubleshooting

### Dataset is missing or rejected

Confirm that the Dropbox download completed and that both the archive and
`COM38_210610_025603.obs` hashes match. An HTML error page saved as `data.zip`
will fail the archive hash. Do not replace the rover file with the 2025
`test.obs`. Make sure `--ephemeris-glob` is quoted so the wildcard reaches the
script as one argument.

### pyrtklib fails to build or import

Use the pinned git archive, the project's Python interpreter, and an empty
`$PYRTKLIB_SITE`. Confirm compiler/build dependencies are available. Delete
only the temporary source/site directories, recreate them, and repeat steps
3–4. Check that the version probe prints `0.2.6` and that `PYTHONPATH` puts the
temporary site before other installations.

### The wrong epoch is selected

Verify the rover and wildcard hashes, use the default strict bounds, and check
that `prl.sortobs(obs)` is running through the script. The expected split index
is 2382 and timestamp is 1623296154.005. A result at 03:35:55 usually means
the fractional strict-bound behavior was changed.

### Satellite rows or weights do not align

The expected IDs and order are exactly
`G01 G03 G07 G14 G21 G22 G28 G30`. The script compares the natural compact
Jacobian rows with the archived `list(set(...))` residual rows and aborts on
any difference. Do not sort one side independently or remap weights by a
different order.

### The normal matrix is rank deficient

The expected `H` shape is 8 by 4 and `rank(N)` is 4. Check that the epoch was
restricted to GPS before OLS and WLS, all eight broadcast ephemerides were
available, and no measurement was dropped. Do not loosen the rank check or
replace the solve with a pseudoinverse for this reproduction.

### Numerical tolerances fail

First compare runtime versions in `paper_epoch_manifest.json` and input hashes.
Then inspect per-iteration discrepancies in `algebra_validation.json`. Confirm
float64 was retained and the CPU substitution was the only disposable TDL
edit. Do not change the equations, Sagnac sign, stopping rule, inverse, or
tolerances merely to force a pass; record a platform-specific discrepancy for
review.
