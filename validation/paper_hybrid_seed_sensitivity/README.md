# TDL-BW initialization-seed sensitivity

This directory supports one controlled question: does changing only the random
initialization seed affect whether the released shared TDL-BW model's bias head
is active or dead? The predefined seeds are `0 1 2 3 4 5 6 7 8 9`. They are
conditions in a fixed sweep, not candidates in a model-selection search.

The existing paper reproduction remains unchanged. Its trainer already accepts
`--seed`, defaults to `20260929`, and seeds Python, NumPy, and Torch immediately
before constructing `HybridShareNet`.

## Controlled procedure

`run_seed.py` loads and hashes KLT3 before model construction, applies the
requested seed, creates a fresh released architecture, and uses the unchanged
100-epoch Adam/WLS training procedure. A completed run then freezes the model
and evaluates KLT1 and KLT2 with the existing held-out preparation and
positioning code. There is no argument for changing the epoch count in a
completed run.

The `--smoke` mode is explicitly not a completed seed result. It executes one
full-KLT3 optimization epoch and skips checkpointing and held-out evaluation.
The aggregator rejects smoke results.

During each training epoch, the instrumented forward obtains the raw two-column
network output once, then applies the released sigmoid/clamp weight transform
and ReLU bias transform. The exact tensors used by the loss are also used for
the diagnostics. Statistics use population standard deviation. Gradient norms
are sampled after `backward()` and before `Adam.step()`:

- bias output row: final linear weight row 1 plus bias element 1;
- weight output row: final linear weight row 0 plus bias element 0;
- shared layers: all trainable layers before the final two-row linear layer;
- global: every trainable parameter.

An exactly zero positive fraction defines inactivity. Bias status is therefore:

- **dead from initialization:** initial positive preactivation fraction is zero
  and final positive bias fraction is zero;
- **initially active, later dead:** initial fraction is positive and final
  fraction is zero;
- **initially dead, later active:** initial fraction is zero and final fraction
  is positive;
- **active:** both fractions are positive.

## Outputs

By default, compact JSON results are written to
`results/paper_hybrid_seed_sensitivity/seed_N.json`, and checkpoints are written
to `checkpoints/paper_hybrid_seed_sensitivity/`. Both parent directories are
already ignored by Git. No per-epoch held-out CSV or plot is generated.

Each result records the initial parameter/output hashes, invariant control
hashes, initial and per-epoch activation fractions, complete per-epoch output
and gradient diagnostics, loss landmarks, final output statistics, checkpoint
hash, compact KLT1/KLT2 metrics, and finite-status checks. The configuration
hash deliberately excludes the seed; the seed is stored separately as the
only treatment variable.

`validate_controls.py` performs no training. It constructs fresh models for
the requested seeds, repeats the first seed, and checks that data/row/GT/feature
identity, normalization, architecture, optimizer, learning rate, epoch count,
solver settings, and evaluation procedure are invariant. It also checks that a
repeated seed gives identical initial parameters and outputs and that every
different seed changes at least one initial parameter tensor.

`summarize_seeds.py` requires completed, finite 100-epoch results with identical
control hashes. It emits rows in numeric seed order and category/metric
aggregates; it performs no metric-based ranking or selection.

## Manual commands

Run these from the repository root after the historical KLT/pyrtklib inputs
used by the paper reproduction have been prepared.

### A. One smoke seed

```bash
.venv/bin/python validation/paper_hybrid_seed_sensitivity/run_seed.py --seed 0 --smoke
```

### B. One complete seed

```bash
.venv/bin/python validation/paper_hybrid_seed_sensitivity/run_seed.py --seed 0
```

### C. All predefined seeds 0–9

```bash
.venv/bin/python validation/paper_hybrid_seed_sensitivity/run_all_seeds.py --seeds 0 1 2 3 4 5 6 7 8 9 --overwrite
```

### D. Control validation

```bash
.venv/bin/python validation/paper_hybrid_seed_sensitivity/validate_controls.py --seeds 0 1 2 3 4 5 6 7 8 9
```

### E. Final aggregation

```bash
.venv/bin/python validation/paper_hybrid_seed_sensitivity/summarize_seeds.py --seeds 0 1 2 3 4 5 6 7 8 9
```

The all-seed command uses `--overwrite` because command B has already produced
seed 0 if these examples are run in order; seed 0 is rerun from a fresh model.
Omit that flag when the output directories are empty. Use `--overwrite` only
when intentionally replacing an existing generated result. Optional explicit
historical-input paths are available through
`--data-root`, `--observation`, `--ephemeris-glob`, `--ground-truth`,
`--tdl-dir`, and `--pyrtklib-site` on the run scripts.
