# Ibiza frozen-output positioning-ablation results

All 30 original paths reproduce the published frozen Ibiza inference over all
2,856 accepted epochs. Solved status, final rank, and iteration count are
exact. The maximum seven-state/ECEF difference is `2.02097e-7 m`, within the
documented `5e-7 m` sub-micrometre tolerance. The difference is explained by
full-cache-batch versus per-epoch neural-forward arithmetic; the validation
case differed by only `4.74e-20` in weight and exactly zero in bias.

Constant positive weight scaling also passed: scale 0.25 was bit-identical to
unit weighting and scale 7 differed by at most `7.11e-14` in state over 64
evenly spaced epochs.

## Main result

Under the unchanged historical Torch WLS path, the frozen learned outputs do
not explain the Ibiza degradation by making the neutral solution worse.
Instead, learned weighting is associated with the largest improvement over
neutral WLS. Bias generally provides a smaller additional improvement, with
greater seed sensitivity for standalone TDL-B.

The across-seed median 3D RMS errors are:

| Architecture/ablation | 2D RMS (m) | Up RMS (m) | 3D RMS (m) | 3D RMS seed range (m) |
| --- | ---: | ---: | ---: | ---: |
| Neutral (common) | 6.895 | 35.069 | 35.741 | 35.741–35.741 |
| TDL-B original | 6.217 | 32.376 | 33.018 | 27.799–38.949 |
| TDL-W original | 5.124 | 20.482 | 21.074 | 16.003–25.756 |
| TDL-BW bias only | 6.675 | 31.314 | 32.018 | 29.990–35.741 |
| TDL-BW weight only | 5.283 | 27.630 | 28.072 | 25.746–32.758 |
| TDL-BW full | 5.165 | 26.005 | 26.715 | 22.168–32.484 |

The common neutral solution is not the stored preprocessing OLS solution. The
neutral historical Torch solver starts from that seven-state OLS result but
then iterates with its reproduced zero-atmospheric-term observation model.
The resulting position displacement from the initializer has mean 37.711 m,
median 33.563 m, P95 68.585 m, and maximum 236.628 m. Correspondingly, neutral
Torch-WLS 3D RMS is 35.741 m, whereas the separately validated stored OLS
initializer has 3D RMS 16.503 m. This distinction prevents attributing the
full OLS-to-model gap to learned bias or weight.

## Seed-level 3D RMS

| Seed | TDL-B original | TDL-W original | BW bias only | BW weight only | BW full |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38.949 | 24.839 | 32.858 | 28.737 | 27.387 |
| 1 | 28.267 | 25.756 | 35.741 | 32.484 | 32.484 |
| 2 | 33.197 | 21.737 | 33.280 | 32.758 | 31.423 |
| 3 | 38.742 | 19.230 | 31.038 | 25.795 | 22.168 |
| 4 | 35.812 | 16.423 | 35.741 | 27.067 | 27.067 |
| 5 | 33.904 | 25.309 | 31.079 | 25.746 | 22.982 |
| 6 | 31.601 | 16.003 | 30.505 | 27.406 | 23.050 |
| 7 | 31.128 | 20.411 | 31.178 | 28.883 | 25.013 |
| 8 | 27.799 | 17.689 | 35.741 | 27.218 | 27.218 |
| 9 | 32.839 | 24.020 | 29.990 | 30.920 | 26.362 |

The neutral value is 35.741 m for every seed. TDL-B improves 3D RMS in seven
seeds and worsens it in three; TDL-W improves it in all ten. TDL-BW weight-only
and full improve 3D RMS in all ten. Seeds 1, 4, and 8 have identically zero
TDL-BW bias, so bias-only equals neutral and full equals weight-only.

## Paired epoch effects

The table reports the across-seed median of each seed's mean paired delta.
Delta is candidate error minus reference-variant error, so negative is better.

| Comparison | 2D delta (m) | Absolute-up delta (m) | 3D delta (m) | Median fraction of epochs improved in 3D |
| --- | ---: | ---: | ---: | ---: |
| TDL-B original − neutral | -0.586 | -2.546 | -2.483 | 0.704 |
| TDL-W original − neutral | -2.581 | -17.113 | -17.263 | 0.946 |
| BW bias-only − neutral | -0.162 | -3.861 | -3.836 | 0.998 |
| BW weight-only − neutral | -1.761 | -9.099 | -9.387 | 0.935 |
| BW full − neutral | -1.830 | -12.970 | -12.658 | 0.963 |
| BW full − bias-only | -1.759 | -7.793 | -8.099 | 0.910 |
| BW full − weight-only | -0.082 | -2.122 | -2.119 | 0.940 |

The three zero-bias seeds contribute exact equality to bias comparisons; the
reported median improvement fraction for active TDL-BW bias seeds is near one.
Vertical error accounts for most of the 3D change. For example, TDL-W lowers
across-seed mean Up RMS by 14.577 m but 2D RMS by 1.721 m. In TDL-BW,
weight-only lowers mean Up RMS by 6.918 m versus 3.044 m for bias-only.

## TDL-BW factorial contrast

For the seven seeds with an active bias branch, the mean interaction is
positive, from 84.193 to 176.811 m², and 93.7%–99.8% of epoch interactions are
positive. Seeds 1, 4, and 8 are numerically zero apart from roundoff because
their bias output is identically zero. Across all ten seeds, the median of the
per-seed mean interaction is 106.851 m².

Positive
`L(full) - L(bias-only) - L(weight-only) + L(neutral)` means the combined
squared-error reduction is smaller than the sum of the two isolated
reductions: a descriptive diminishing-return/antagonistic numerical contrast.
It is not a physical causal interaction. The full branch combination still
improves 3D error relative to either isolated branch for active-bias seeds.

These findings apply only to the frozen Ibiza outputs, accepted observations,
historical solver, and propagated EPN reference used here. They do not show
that C/N0 shift causes failure, that retraining would solve it, or that a
learned branch is universally beneficial or harmful.

