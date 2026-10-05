# Corrected-GT BiasNet initialization-seed sensitivity

This directory supports one controlled question: how sensitive is the released
BiasNet / TDL-B training trajectory to neural-network random initialization
after fixing the demonstrated ground-truth duplication defect? The predefined
seeds are `0 1 2 3 4 5 6 7 8 9`. They are fixed experimental conditions, not
candidates for tuning or best-seed selection.

The scientifically frozen baseline remains
`validation/paper_biasnet_corrected_gt`. This sweep imports its corrected
one-to-one KLT3 loss and does not modify its code or artifacts. The existing
seed `20260929` results are retained only as a separately labelled historical
reference and are excluded from all seed 0–9 aggregates.

## Frozen procedure

Every complete run uses the same KLT3 cache (405 released-code epochs and
8,857 rows), feature construction/order and KLT3 normalization, released
`BiasNetTest` architecture, Adam optimizer at `0.01`, 500 chronological
full-dataset updates without shuffling, and the sum of the 405 unsquared 3D ENU
error norms. The differentiable identity-weight WLS solver retains the released
per-epoch OLS initialization, maximum 10 iterations, and `1e-4` tolerance.

The verified network is:

```text
3 -> Linear(64) -> ReLU -> Linear(128) -> ReLU -> Linear(1)
```

The final output is linear, unclipped, and measured in metres. Negative biases
are valid. The observation convention remains:

```text
P_corrected = P_RTKLIB - predicted_bias
```

The requested seed is applied to Python, NumPy, and Torch immediately before
constructing a fresh model. There is no complete-run epoch-count argument, so
a scientifically complete run always performs exactly 500 updates.

## Diagnostics and outputs

The instrumented forward evaluates the original sequential layers once and
returns the tensors immediately after each hidden ReLU plus the unchanged
linear output. Tests verify bit-exact outputs and gradients against the normal
forward method.

For the initial model and every training epoch, the JSON result records hidden
activation distributions, exactly-zero and positive fractions, and the number
of hidden units whose activation is zero for every KLT3 row. This last
condition is the only definition of a dead hidden neuron used here. It does not
imply that the whole BiasNet is dead. The linear bias distribution records
minimum, maximum, mean, median, population standard deviation, P5, and P95.

Gradient norms are recorded for the first hidden, second hidden, final output,
and full network after `backward()` and before `Adam.step()`. Parameter-change
checks are recorded after every update. Per-parameter detail is retained for
epochs `1 2 5 10 25 50 100 200 300 400 500`.

Complete results are written to
`results/paper_biasnet_seed_sensitivity/seed_N.json`; checkpoints are written
to `checkpoints/paper_biasnet_seed_sensitivity/`. Both parent directories are
ignored by Git. Each complete result also contains compact frozen KLT1/KLT2
metrics (mean, median, RMS, P68, and P95 for 2D and 3D error). The established
CSV-only independent checker is run through temporary per-epoch CSVs, requires
`1e-12 m` agreement, and retains no large CSV.

`--smoke` performs one full-KLT3 update, skips checkpointing and held-out
evaluation, writes `smoke_seed_N.json`, and marks the output as incomplete and
ineligible for aggregation.

`validate_controls.py` performs no training. It hashes all KLT3 arrays, the
identity-ordered GT mapping, normalization, architecture, optimizer, full
configuration, and retained KLT1/KLT2 evaluation rows. It repeats the first
seed to prove identical parameters and outputs, compares every pair of
different seeds, and explicitly asserts that the historical duplicated-GT
defect is absent.

`summarize_seeds.py` requires complete, finite results for all seeds 0–9 with
identical control hashes. It emits numeric-seed-order rows without ranking and
reports population descriptive statistics (`mean`, `median`, `std`, `min`,
`max`). It also preserves all individual loss curves and produces epoch-wise
median and 25th–75th percentile arrays. This fixed `n=10` sample characterizes
sensitivity; it is not population-level statistical proof.

## Manual commands

Run from the repository root. The historical KLT/pyrtklib inputs used by the
existing reproduction must remain available in their documented `/tmp`
locations, or be supplied with the optional input-path arguments.

### A. One smoke seed

```bash
.venv/bin/python validation/paper_biasnet_seed_sensitivity/run_seed.py --seed 0 --smoke
```

### B. Control validation for seeds 0–9

```bash
.venv/bin/python validation/paper_biasnet_seed_sensitivity/validate_controls.py --seeds 0 1 2 3 4 5 6 7 8 9
```

### C. One complete 500-epoch seed

```bash
.venv/bin/python validation/paper_biasnet_seed_sensitivity/run_seed.py --seed 0
```

### D. All predefined seeds 0–9

```bash
.venv/bin/python validation/paper_biasnet_seed_sensitivity/run_all_seeds.py --seeds 0 1 2 3 4 5 6 7 8 9 --overwrite
```

The `--overwrite` above intentionally replaces seed 0 produced by command C
with a fresh independent run. Omit it when the result/checkpoint directories
are empty.

### E. Final aggregation

```bash
.venv/bin/python validation/paper_biasnet_seed_sensitivity/summarize_seeds.py --seeds 0 1 2 3 4 5 6 7 8 9
```

No command ranks seeds, chooses a representative curve, or compares this
experiment with TDL-W or TDL-BW.
