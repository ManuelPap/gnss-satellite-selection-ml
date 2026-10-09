# Current TDL-GNSS + TASGNSS independent reproduction

This directory contains the current nine-feature `HybridShareSysNet` + TASGNSS
pipeline: KLT3 training, ten-seed KLT1/KLT2 held-out validation, and the future
zero-shot Ibiza generalization evaluation.

This harness reproduces the released current-stack execution path at pinned
TDL-GNSS `a640b283`, TASGNSS `fdd7e8e`, and pyrtklib `1c468db`. It is an
independent training experiment. The unavailable official
`HybridShareSysNet` checkpoint, its random seed, and the historical
three-feature implementation are not reproduced.

The enforced sequence is KLT3 training, immutable checkpoint hashing, then
KLT1/KLT2 evaluation. The evaluator refuses to open either held-out split until
all ten 120-epoch checkpoints pass the freeze gate. It evaluates every seed;
there is no seed selection, recalibration, normalization refit, or training
update.

## Current source ownership

| Operation | Authoritative current source |
| --- | --- |
| RINEX reading | `TASGNSS/tasgnss/core.py::read_obs`, calling pyrtklib/RTKLIB `readrnx` |
| satellite positions and clocks | `core.py::get_sat_pos`, calling RTKLIB `satposs` |
| pseudorange correction | `core.py::prange` |
| ionosphere | `core.py::get_atmosphere_error`, RTKLIB `ionocorr(..., IONOOPT_BRDC, ...)` |
| troposphere | `core.py::get_atmosphere_error`, RTKLIB `tropcorr(..., TROPOPT_SAAS, ...)` |
| Sagnac | `core.py::get_sagnac_corr` |
| neutral solution | `TDL-GNSS/preprocess.py::align_data` calling `tas.wls_pnt_pos(..., w=1, return_residual=True)` |
| neutral residual | `TASGNSS/tasgnss/core.py::wls_pnt_pos`, final `pr_tensor - psr - b` returned in `residual_info` |
| neural features | `TDL-GNSS/train.py`, columns SNR, elevation, azimuth, residual, then G/R/E/C/J one-hot |
| normalization | `TDL-GNSS/train.py` population mean/std plus `model/model.py::StandardizeLayer` |
| neural network | `model/model.py::HybridShareSysNet` |
| learned weight | output column 0, sigmoid then clamp to `[0,1]` |
| learned bias | output column 1, ReLU, non-negative metres |
| differentiable WLS | `tas.wls_pnt_pos(..., w=weight, b=bias, enable_torch=True)` |
| position loss | `TDL-GNSS/train.py`, Euclidean norm of 3D ENU error |

The solver forms `W = diag(w)` and solves `lstsq(W @ H, W @ residual)`, so the
least-squares objective contains squared network outputs. Its bias sign is
`corrected pseudorange - predicted pseudorange - b`.

## Features and training semantics

The exact feature order is:

```text
[SNR, elevation, azimuth, neutral residual, G, R, E, C, J]
```

SNR is current TASGNSS's RTKLIB `SNR[0]/1000` value and is cast through NumPy
`int8` before conversion to model `float64`. Elevation and azimuth are radians,
the residual and bias are metres, and constellation fields are dimensionless
one-hot values. NumPy population statistics are computed over KLT3 for the
first four fields; one-hot means/stds are fixed to zero/one. Upstream creates
those arrays as float32 tensors and then converts the entire model to float64,
which the harness preserves.

The current trainer uses Adam at `0.01`, default zero weight decay, no
scheduler, 120 epochs, NumPy shuffling each epoch, a 3000-sample gradient
accumulation parameter, a 200 m position-loss rejection threshold, checkpoints
every 10 epochs, and final `multinet_3d.pth`. It constructs
`MSELoss(reduction='sum')` but never calls it; the actual objective is the 3D
ENU Euclidean norm. Its reported epoch average divides accepted accumulated
loss by the original record count, which is also preserved.

`TDL-GNSS/config/train.json` references
`KLTDataset/config/0610_klt3_404.json`, but KLTDataset is not vendored or pinned
by TDL-GNSS. The current official KLTDataset configs supply these UTC bounds:

| Split | UTC bounds passed to current `filter_obs` | Current epochs |
| --- | --- | ---: |
| KLT1 | `[1623296137, 1623296340]` | 203 |
| KLT2 | `[1623296900, 1623297109]` | 209 |
| KLT3 | `[1623297134, 1623297538]` | 404 |

Current TASGNSS subtracts 18 leap seconds when filtering. Consequently KLT3
starts at GPST-like `1623297152.006`, not the historical released-code
405-epoch boundary at `1623297151.006`. Ground truth is reconstructed from the
public 100 Hz source into the current documented
`timestamp,latitude,longitude,height,roll,pitch,yaw` rows, nearest-matched at
each epoch, then passed through the exact current TDL lever-arm transform.

## Leakage boundary

Neutral preprocessing receives only an observation epoch and navigation data.
Ground truth is attached after the neutral position, corrections, residual,
and solver arrays have been produced. The smoke test changes ground-truth
coordinates and hashes the nine-feature tensor, neutral solution, and learned
solver arrays before and after. All three must remain bitwise identical while
the calculated position loss must change.

## Commands

Run the non-scientific deterministic smoke test:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src .venv/bin/python -m validation.tdl_9feature_tasgnss_analysis.train \
  --output-root /home/manuelpap/PhD/external_data/current_tdl_reproduction \
  --threads 1 smoke --seed 0 --subset-size 404 --epochs 1
```

Run focused validation, including the real-data leakage boundary:

```bash
CURRENT_TDL_RUN_INTEGRATION=1 PYTHONPATH=src \
  .venv/bin/python -m pytest -q tests/test_current_tdl_reproduction.py
```

The single manual command for the full experiment is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src .venv/bin/python -m validation.tdl_9feature_tasgnss_analysis.train \
  --output-root /home/manuelpap/PhD/external_data/current_tdl_reproduction \
  --threads 1 all --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 120
```

The experiment was run only after the smoke and review gate passed. Non-empty
seed directories are never overwritten.

After all ten runs completed, the exact freeze-and-evaluation command was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src .venv/bin/python -m validation.tdl_9feature_tasgnss_analysis.evaluate_klt \
  --output-root /home/manuelpap/PhD/external_data/current_tdl_reproduction \
  --threads 1
```

Evaluation reports the requested 2D/3D statistics, axis RMS values, exact epoch
reconciliation, per-seed paired deltas against uniform-weight/zero-bias current
TASGNSS, and distributions over the ten per-seed summaries. Epoch×seed rows are
not pooled.

Yin et al. (Sensors 2026, 26, 5622) values are recorded only as an external
literature sanity reference in the evaluation manifest. They are not tuning
targets or evidence of exact reproduction.

## Completed held-out result summary

The scientific interpretation and two-decimal result tables are in
[`KLT_HELDOUT_RESULTS.md`](KLT_HELDOUT_RESULTS.md). Check that the tracked note
still agrees with the immutable JSON artifacts using the read-only renderer:

```bash
PYTHONPATH=src .venv/bin/python \
  -m validation.tdl_9feature_tasgnss_analysis.render_klt_results \
  --check validation/tdl_9feature_tasgnss_analysis/KLT_HELDOUT_RESULTS.md
```

Without `--check`, the renderer prints the regenerated note to standard output.
It never loads or modifies a checkpoint, preprocesses data, trains a model, or
selects a seed.
