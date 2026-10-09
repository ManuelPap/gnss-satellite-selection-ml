# Frozen model-response probe

This diagnostic runs all 30 released seed checkpoints over the exact frozen
KLT1, KLT2, and Ibiza raw feature rows. Its hard execution boundary is:

`raw feature -> checkpoint StandardizeLayer -> frozen network -> neural output -> STOP`

It performs no WLS or position solution, reads no ground truth, creates no
optimizer, calls no backward/training operation, and fits no normalization.
The strict existing checkpoint loader verifies checkpoint bytes before loading
the repository's released model definitions. Every model is in evaluation
mode with gradients disabled. Full forward passes are repeated exactly, and
all parameters and buffers are compared bit for bit before and after.

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib-model-response \
  .venv/bin/python -m validation.domain_shift.model_response_probe
```

The command refuses to overwrite an existing output and atomically publishes
the completed directory at:

`../external_data/domain_shift/model_response/`

## Inputs and semantics

The input columns are exactly `[SNR[0]/1000, elevation_rad,
OLS_residual_m]`. The embedded normalization is exactly:

- mean: `[29.084903717041016, 0.8471900224685669,
  -1.7873233559839719e-07]`
- standard deviation: `[5.891841888427734, 0.2879891097545624,
  4.3018035888671875]`

TDL-B emits the released raw signed bias in metres. TDL-W emits its released
sigmoid output multiplied by 10 and clamped to `[0, 10]`. TDL-BW returns the
released `(weight, bias)` tuple: sigmoid/clamped weight in `[0, 1]` and ReLU
non-negative bias in metres. Exact zero/positive bias fractions are therefore
reported only for TDL-BW.

## Outputs

There are 90 deterministic row-level NPZ files, one for each dataset,
architecture, and seed. Each retains raw and normalized features, row and
epoch indices, satellite and constellation identity, architecture, seed, and
the applicable neural outputs. Ibiza's source-row, source-observation,
RTKLIB-satellite, PRN, and constellation-code identities are also retained;
KLT row-within-epoch and numeric satellite identities are retained.

Per-seed summaries keep observations within their seed. Across-seed summaries
then summarize those ten seed-level statistics and never pool observations
across seeds as independent samples. Percentiles use
`numpy.percentile(method="linear")` and standard deviations are population
standard deviations (`ddof=0`).

Feature/output relationships use Pearson and average-rank Spearman
correlations plus fixed, documented bins. Density plots display the
across-seed median response per observation; ECDF plots retain the ten seed
curves separately. KS distances are descriptive distribution distances, not
hypothesis tests or evidence that response shifts cause positioning error.

The generated manifest records every input and checkpoint hash before and
after inference, the exact per-row schemas, forward controls, all generated
artifact hashes, fixed-bin definitions, and the descriptive comparison
definition.
