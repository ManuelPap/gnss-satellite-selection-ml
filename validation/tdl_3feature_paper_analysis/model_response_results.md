# KLT1/KLT2 versus Ibiza frozen-network responses

The completed forward-only probe contains 90 row-level artifacts (three
datasets by three architectures by ten seeds), 120 per-seed output summaries,
126 across-seed metric summaries, 480 correlation rows, 3,680 fixed-bin rows,
400 matched-absolute-residual comparisons, and 20 plots. The external
manifest SHA-256 is
`a9844b905ae6fa616460da72d76c3412f1bb7d0e8fc01ced3d0a4fd0fa0eb0d2`.
All 119 files covered by its artifact index are hash-verified.

This is a descriptive network-response result. It does not show, and must not
be cited as showing, that a response difference causes positioning
degradation. No position solution or ground truth enters the analysis.

## Main result

Ibiza response distributions are visibly and quantitatively separated from
the two held-out KLT distributions. There was no pre-specified inferential
threshold for “substantial”; the unitless descriptive comparison is the
two-sample empirical KS distance computed separately for each seed, output,
and KLT reference.

| Architecture | Mean KS over outputs, seeds, and KLT references | Range |
| --- | ---: | ---: |
| TDL-B | 0.334 | 0.083–0.476 |
| TDL-W | 0.303 | 0.130–0.378 |
| TDL-BW | 0.539 | 0.000–0.765 |

TDL-BW changes most by this declared architecture-level score. Its weight
branch is the most seed-stable large shift: KS is 0.482–0.756 across all 20
seed/reference comparisons (mean 0.624). Its bias branch is heterogeneous:
seven seeds shift positively, while three released seeds emit zero bias for
both domains and therefore have KS zero. TDL-B's Ibiza-minus-KLT median bias
is positive in 18/20 comparisons. TDL-W's mean Ibiza weight is lower in all
20 comparisons, although its median-difference sign is less stable (7
positive, 13 negative).

The across-seed median of the ten per-seed output medians is:

| Output | KLT1 | KLT2 | Ibiza |
| --- | ---: | ---: | ---: |
| TDL-B bias (m) | -2.115 | -1.929 | 1.452 |
| TDL-W weight | 0.483 | 0.363 | 0.478 |
| TDL-BW bias (m) | 2.922 | 2.985 | 5.601 |
| TDL-BW weight | 0.136 | 0.092 | 0.866 |

The TDL-W medians alone conceal its distribution change: its across-seed
median of per-seed means falls from 3.594/3.531 on KLT1/KLT2 to 0.959 on
Ibiza. The complete seed-level distributions, including all requested
percentiles and TDL-BW zero/positive fractions, are retained in
`per_seed_summary.csv`; `across_seed_summary.csv` contains the non-pooled
across-seed summaries.

## Feature-conditioned findings

Matching only absolute OLS-residual bins does not remove the response
difference. Across the populated fixed bins, seeds, and two references, mean
matched-bin KS remains 0.370 for TDL-B bias, 0.361 for TDL-W weight, 0.408 for
TDL-BW bias, and 0.619 for TDL-BW weight. This shows that residual magnitude
alone does not explain the observed response shifts; it does not isolate a
causal feature.

The strongest consistent monotone relationships are:

- TDL-W weight versus elevation: median Spearman 0.867 on each KLT dataset
  and 0.871 on Ibiza.
- TDL-BW weight versus C/N0: 0.863 on KLT1, 0.862 on KLT2, and 0.940 on
  Ibiza.
- TDL-BW bias versus elevation: -0.892 on KLT1, -0.887 on KLT2, and -0.682
  on Ibiza. The zero-bias seeds make some individual correlations zero.
- TDL-B bias is most strongly related to elevation on KLT1/KLT2 (median
  Spearman -0.456/-0.536), but to signed OLS residual on Ibiza (-0.500).

The correlation table, fixed-bin medians/P05/P95, and density plots retain
the seed and dataset separation needed to inspect these relationships.

## High-C/N0 and nominal Ibiza strata

The KLT3 training maximum was 39 in `SNR[0]/1000` units. Exactly 65,109 of
73,204 Ibiza rows (88.94%) are above that boundary; no KLT1 or KLT2 held-out
row is above it. Across the ten seeds, high-C/N0 Ibiza per-seed medians span:

- TDL-B bias: -3.205 to 10.182 m;
- TDL-W weight: 0.139 to 1.789;
- TDL-BW bias: 0 to 8.826 m;
- TDL-BW weight: approximately 4.47e-10 to 0.999999999.

Thus the high-C/N0 extrapolation region produces strongly seed-dependent
responses, including near-boundary hybrid weights. This is a response audit,
not a weight-quality classification.

The transparent “nominal Ibiza” stratum uses the same pooled thresholds for
all datasets: C/N0 at least 49.7, elevation at least 0.924073 rad, and absolute
OLS residual at most 0.622725 m. It contains 4,688 Ibiza rows and no KLT1/KLT2
rows because both held-out C/N0 maxima are below 49.7. Some seeds assign large
responses in that stratum (per-seed medians reach 15.478 m for TDL-B bias,
8.799 m for TDL-BW bias, and 9.627 for TDL-W weight), while TDL-BW weight
medians span roughly 1.17e-7 to 1.0. An exact joint feature-matched KLT
comparison is therefore unavailable for this nominal stratum; the
matched-residual result above is the supported, narrower comparison.
