# Paper-era WeightNet initialization-seed sensitivity

## Scientific question and fixed treatment

This experiment asks how sensitive the released paper-era TDL-W training
pipeline is to neural-network random initialization.  The predefined treatment
is exactly the ten seeds `0 1 2 3 4 5 6 7 8 9`.  Every seed is reported in
numeric order; the experiment performs no hyperparameter tuning, seed ranking,
best-seed selection, or paper-value matching.

Seed `20260929` is retained only as the already reviewed historical reference.
It is excluded from every seed-0--9 statistic and need not be rerun.

## Independently verified released architecture

The primary historical source is TDL-GNSS commit
`dd5eac669676ba0a922102047e58c2dfc9be9267`.  Direct inspection of its
`model.py` and `weight_network_train.py` confirms that the training class is
`WeightNet` and that its exact path is:

```text
KLT3 population standardization
 -> Linear(3, 64)   -> sigmoid
 -> Linear(64, 128) -> sigmoid
 -> Linear(128, 64) -> sigmoid
 -> Linear(64, 1)   -> sigmoid
 -> multiply by 10
 -> clamp to [0, 10]
```

This is the architecture in the frozen local reproduction.  PyTorch's default
`Linear` initialization is used.  The explicit seed is applied to Python,
NumPy, Torch, and available CUDA RNGs immediately before model construction.
The existing seed-`20260929` trainer behavior remains unchanged.

WeightNet has no TDL-BW final-bias ReLU and therefore no "dead bias head"
criterion.  This experiment measures sigmoid activation distributions,
saturation fractions, gradients, weights, losses, and held-out positioning; it
does not define a new failure threshold.

## Frozen controls

All seeds use the same frozen KLT3 cache: 405 released-code successful epochs,
8,857 measurement rows, chronological order, and no shuffle.  Features remain
`[C/N0, elevation, equal-weight OLS residual]` and use only the KLT3 population
mean/std, preserving the released float32-then-double construction order.

The optimizer remains Adam with learning rate `0.01` and PyTorch defaults.  A
training epoch remains one full-dataset accumulated update.  Every run performs
exactly 500 updates with the released sum of unsquared per-epoch 3D ENU error
norms.  Positioning retains equal-weight OLS initialization, the historical
seven-slot differentiable WLS/observation conventions, at most 10 iterations,
and tolerance `1e-4`.

After training, the model is moved to CPU, put in evaluation mode, frozen, and
evaluated under `torch.no_grad()` on unchanged KLT1 (203 epochs / 4,676 rows)
and KLT2 (209 epochs / 4,914 rows).  Held-out inputs are not prepared until
training and checkpointing have finished.  No held-out normalization is fit,
and an independent CSV-only aggregation path must agree within `1e-12 m`.

Before a non-smoke run starts epoch 1, a fail-fast preflight resolves the
held-out paths, verifies every audited input hash, verifies the pinned TDL
source hash, and imports pyrtklib `0.2.6`.  It does not parse held-out epochs or
read held-out measurements/targets into the training path.  This prevents a
missing evaluation dependency from being discovered only after 500 updates.

`validate_controls.py` hashes every KLT3 cached array, row identity, GT,
features, normalization, architecture, optimizer/configuration, and every
retained held-out row.  It repeats one seed to prove identical parameters and
outputs, then checks that every different seed pair changes at least one
trainable tensor.

## Diagnostics and result schema

Before the first update and at every training epoch, the experiment records
preactivation and activation min/max/mean/median/std/P5/P95 for all four
sigmoids.  Each sigmoid also records activation fractions below `0.01` and
above `0.99`.  The final output records the complete KLT3 weight distribution
and these two distinct families of thresholds:

- Neural saturation: final sigmoid `<0.01` / `>0.99`, equivalently released
  final weight `<0.1` / `>9.9`.
- Descriptive GNSS weights: weight `<1e-5`, `<0.01`, `>0.5`, and `>0.99`.

Low learned weights are not classified as failures; downweighting is the
network's intended function.

Every epoch records combined weight-and-bias gradient norms for the first,
second, third, and output linear layers, plus a global norm.  It records
whether every trainable tensor received a finite gradient and whether every
expected tensor changed after the Adam step.  Epochs
`1, 2, 5, 10, 25, 50, 100, 200, 300, 400, 500` additionally retain per-tensor
gradient and update values.  A small gradient or unchanged tensor is reported,
not silently turned into a scientific failure criterion.

Each complete result includes initial/configuration/data/normalization hashes,
all 500 pre-update losses, epoch-500 and final post-update losses, the minimum
evaluated loss, activation/weight and gradient histories, final KLT3 weight
statistics, checkpoint SHA-256, timing, finite checks, held-out descriptive
metrics, and held-out row hashes.  Checkpoints and results live under the
repository's already ignored `checkpoints/` and `results/` trees.

Smoke results are named `smoke_seed_N.json`, marked
`smoke_completed_incomplete`, and have `scientific_aggregate_eligible=false`.
The summary parser rejects them.

The final summary preserves all ten 500-epoch loss curves plus their per-epoch
median and 25th--75th percentile band.  It reports mean, median, population
standard deviation, minimum, and maximum across the fixed n=10 sample.  It
does not rank seeds or choose a representative curve.  The sample
characterizes these ten controlled initializations; it is not population-level
proof.

## Exact manual commands

Run from the repository root.  The default data resolver uses the same audited
local KLT inputs as the frozen WeightNet evaluation and the validated
`../external_data/.paper_runtime` cache. The optional explicit flags accepted
by `run_seed.py`, `run_all_seeds.py`, and `validate_controls.py` are
`--data-root`, `--observation`, `--ephemeris-glob`, `--ground-truth`, and
`--runtime-dir`.

### A. One smoke seed

```bash
PYTHONPATH=src .venv/bin/python \
  validation/paper_weightnet_seed_sensitivity/run_seed.py \
  --seed 0 --smoke
```

### B. Validate controls for seeds 0--9

```bash
PYTHONPATH=src .venv/bin/python \
  validation/paper_weightnet_seed_sensitivity/validate_controls.py \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

### C. One complete 500-epoch seed

```bash
PYTHONPATH=src .venv/bin/python \
  validation/paper_weightnet_seed_sensitivity/run_seed.py \
  --seed 0
```

### D. All predefined seeds 0--9

```bash
PYTHONPATH=src .venv/bin/python \
  validation/paper_weightnet_seed_sensitivity/run_all_seeds.py \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

### E. Final aggregation

```bash
PYTHONPATH=src .venv/bin/python \
  validation/paper_weightnet_seed_sensitivity/summarize_seeds.py \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

Use `--overwrite` only when intentionally replacing generated ignored output.
The ten complete training runs are deliberately manual and are not executed by
pytest.
