# KLT3 versus Ibiza feature-domain audit

This read-only diagnostic compares the exact paper-era KLT3 raw training
features with the already frozen Ibiza raw feature artifact. It uses only

`[SNR[0]/1000, elevation_rad, equal-weight OLS_residual_m]`.

It does not train, fit an Ibiza scaler, run inference, deserialize a model,
change WLS, or use ground truth as an input. Percentiles use
`numpy.percentile(method="linear")`; standard deviations are population
standard deviations (`ddof=0`). Elevation relationship bins are fixed at
0, 15, ..., 90 degrees (stored and evaluated in radians), and C/N0 bins are
fixed at 0, 5, ..., 60 in `SNR[0]/1000` units.

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib-domain-shift \
  .venv/bin/python -m validation.domain_shift.audit
```

Bulk CSVs and plots are written outside Git to
`../external_data/domain_shift/klt3_vs_ibiza/`. The generated manifest hashes
the exact KLT3/Ibiza artifacts, all 30 checkpoints, and all 30 frozen Ibiza
inference JSONLs before and after analysis.

## Provenance

KLT3 is reused from `validation/paper_weightnet/klt3_features.npz` (SHA-256
`ae1457a933992f57e850a0abda64c1150f4586640d311b1880f00ed1262b5815`).
It contains 8,857 paper-era rows produced from TDL-GNSS commit
`dd5eac669676ba0a922102047e58c2dfc9be9267` with the pinned pyrtklib 0.2.6
runtime hypothesis. The detailed raw-input hashes and the documented
405-versus-404 epoch discrepancy remain in
`validation/paper_weightnet/klt3_feature_manifest.json`.

Ibiza is reused from `../external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz`
(SHA-256 `edb0189e9eadf3266d75984e3041a90306dd44b3ebecc0101eb188f933dc88c5`),
with 73,204 rows. No Ibiza row is recomputed.

## Follow-on model-response probe

The historical feature-domain audit correctly stopped because complete KLT1
and KLT2 pre-inference rows were not then available. Those rows have since
been frozen and validated independently. The separate, forward-only follow-on
probe is documented in [model_response_probe.md](model_response_probe.md).
The completed scientific interpretation is recorded in
[model_response_results.md](model_response_results.md).
The subsequent frozen-output positioning ablation is documented in
[positioning_ablation.md](positioning_ablation.md).
Its completed interpretation is recorded in
[positioning_ablation_results.md](positioning_ablation_results.md).
The original audit and its `model_response_blocker.json` remain an immutable
record of that earlier stop condition; they are not rewritten retroactively.
