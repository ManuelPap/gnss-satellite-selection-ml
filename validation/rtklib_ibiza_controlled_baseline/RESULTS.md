# Controlled Official RTKLIB Ibiza SPP Baseline

## Objective

This experiment compares a conventional official RTKLIB single-point
positioning (SPP) solution against (1) neutral TASGNSS and (2) the frozen,
KLT3-trained HybridShareSysNet + TASGNSS pipeline. All three results use the
same frozen Ibiza evaluation support and the same ground truth.

This is a comparison of complete positioning pipelines, not of mathematically
identical solvers.

## Frozen RTKLIB configuration

The predeclared conventional baseline used official RTKLIB v2.4.3-b34 with:

- SPP / single mode;
- L1 single-frequency pseudorange;
- forward solution direction;
- GPS + GLONASS + Galileo (`navsys=13`);
- 7 degree elevation mask;
- SNR masks disabled;
- broadcast ionosphere correction;
- Saastamoinen troposphere correction;
- broadcast satellite ephemerides; and
- no excluded satellites.

The 7 degree elevation mask and all other scientific configuration choices
were frozen before inspection of the full-day RTKLIB accuracy.

## Evaluation support

RTKLIB first processed the complete Ibiza observation interval. Timestamp
selection for the comparison occurred only after the raw RTKLIB positions had
been produced and serialized.

- Full Ibiza source epochs: 2,880
- RTKLIB full-day solutions: 2,880
- Frozen required comparison epochs: 2,849
- RTKLIB solutions on required support: 2,849
- Missing required epochs: 0
- Extra RTKLIB epochs outside the frozen support: 31
- Exact timestamp matching: true

The comparison support consists of the predeclared 2,849 TDL/TASGNSS
timestamps. No epoch was removed according to its positioning error.

## Ground-truth isolation

Ground truth was unavailable during RTKLIB execution. The raw RTKLIB ECEF
positions and timestamps were serialized before the ground-truth file was
opened. The manifest's perturbation audit reports that changing the ground
truth changed the ENU errors and 2D/3D metrics, while the raw ECEF positions,
timestamps, epoch-support matching, solver status, configuration, and input
hashes remained unchanged.

## Controlled positioning results

All values are metres and are displayed to three decimal places. The learned
column is the mean of ten seed-level metric summaries; seed-by-epoch values
were not pooled. RTKLIB and neutral TASGNSS are single deterministic
solutions.

| Metric | Official RTKLIB SPP | Neutral TASGNSS | HybridShareSysNet + TASGNSS (10-seed mean) |
|---|---:|---:|---:|
| 2D mean | 2.142 | 4.609 | 2.582 |
| 2D median | 1.745 | 2.728 | 1.915 |
| 2D RMS | 2.619 | 7.580 | 3.597 |
| 2D P95 | 5.110 | 11.769 | 6.654 |
| 3D mean | 2.925 | 7.734 | 4.613 |
| 3D median | 2.283 | 4.413 | 3.536 |
| 3D RMS | 3.333 | 14.948 | 6.557 |
| 3D P95 | 6.045 | 22.405 | 11.078 |
| East RMS | 1.322 | 3.231 | 1.877 |
| North RMS | 2.261 | 6.857 | 3.059 |
| Up RMS | 2.061 | 12.884 | 5.369 |

The exact machine-readable values remain in `controlled_comparison.json`.

## Main observation

The learned current TDL+TASGNSS pipeline substantially improves over neutral
TASGNSS in this Ibiza experiment. Nevertheless, the controlled official
RTKLIB SPP baseline has lower aggregate positioning error than the ten-seed
learned mean for the reported metrics. The present learned pipeline therefore
does not outperform this conventional RTKLIB SPP benchmark on the controlled
Ibiza test.

This observation is specific to this dataset, frozen support, configuration,
and experiment. It neither implies that machine learning does not work nor
identifies why RTKLIB performs better. Attribution to an individual solver
component would require a separate controlled ablation.

## Important interpretation limits

RTKLIB and TASGNSS are not solver-identical. Audited material differences
include:

- RTKLIB's 7 degree elevation mask versus no equivalent TASGNSS-level mask;
- conventional RTKLIB stochastic weighting versus neutral TASGNSS identity
  weights;
- different solver, state, iteration, and validation semantics; and
- different signal and code handling.

Accordingly, this is **a controlled same-data, same-ground-truth,
same-epoch-support comparison of complete positioning pipelines**, not an
isolation of one individual algorithmic difference.

## Artifact provenance

The experiment manifest records:

- Project Git branch: `research/current-tdl-tasgnss-reproduction`
- Project Git commit: `7c1c5478ff2c57088ceda5edb1ec9f2a3663c150`
- Official RTKLIB branch: `experiment/spp-baseline`
- Official RTKLIB commit: `180043ee24b6d2b168f98b64be15f69d50046b1a`
- Official RTKLIB tag/description: `v2.4.3-b34`
- Configuration SHA256: `78c1d79ab21c23b2eeeb18fd434dd90e561ceccc1f6455c456b900d7f7ed5d66`
- Observation RINEX SHA256: `18dfcaa247fb69b8260eda1257279411ca2317a244d3c3afd37eb74bfaa8000b`
- Navigation RINEX SHA256: `423c488a7f0a4825c2e88d848775726b86f65b37ad6a6a63bc98a26fc8c0e1bb`
- Frozen support SHA256: `e2b0c17c6f717d5e5d98450fb20d0e20fc99adc7f5c677814d5d80d187ad58f4`
- Ground-truth SHA256: `801309b88988ee7407bf2568016b580ba592b7a15bac4789f7f9a720116b94cd`

The RTKLIB executable and five final result artifacts were newly hashed for
this documentation pass:

- `bin/rnx2rtkp`: `fcbc09b688312343d3e1f9bb5a14c6868f9b0b2cc54fed8cb2f32aa71ed4e5f0`
- `rtklib_raw_full.pos`: `191b0194aa4c3103d59b8be324cd0954ac706c838c0e9931b95869e0090be4fd`
- `rtklib_positions_pre_ground_truth.npz`: `07570115e75307fe7b1e1568e5490be7406279ed2ea9901eb2161a3efbaa16a9`
- `rtklib_primary_metrics.json`: `02ad49f35c0f0a8a10fc04eff8d83a22bf90d6e295304f20bd79c3c182614f92`
- `controlled_comparison.json`: `e488cd0bc986e14d61e932f8f75e11fd4851cac32801cb5579645e403f40e313`
- `rtklib_ibiza_manifest.json`: `de8509c0cf722892c41d02300f2ff896cccbf8aa58acdc1f019dd40673457224`

The newly computed hashes of the first four result artifacts agree with the
hashes recorded inside the manifest. The manifest's own hash cannot be stored
inside itself and is reported only as a newly computed documentation hash.
