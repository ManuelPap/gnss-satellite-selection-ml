# Ibiza external-generalization preprocessing

This package creates one deterministic, observation-only Ibiza dataset and
provides common frozen inference for TDL-B, TDL-W, and TDL-BW. Preprocessing
runs the validated paper-era
equal-weight OLS preprocessing and stores the raw three-column feature matrix:

```text
[SNR[0]/1000, elevation radians, equal-weight OLS residual metres]
```

Preprocessing does not load, train, or evaluate a neural network. Neither
preprocessing nor inference computes Ibiza normalization or accepts a
ground-truth/reference-coordinate input. Frozen inference strictly loads each
checkpoint's own KLT3 `StandardizeLayer` through `load_state_dict`.

## Frozen checkpoint inventory and inference smoke

`frozen_checkpoint_manifest.json` records the repository-relative path,
SHA-256, expected state-dict keys, and embedded normalization tensors for all
30 final seed checkpoints. `checkpoints.py` verifies the complete inventory,
including exactly seeds 0--9 for each architecture, 30 distinct hashes, and
bit-identical normalization. Each checkpoint's bytes are hashed before those
same bytes are deserialized for inference.

The shared `inference.py` path consumes the NPZ's raw `features` rows, applies
the checkpoint `StandardizeLayer`, runs the released project-side network in
eval/inference mode, and sends bias, weight, or both into the already validated
paper-era WLS. No optimizer, backward pass, parameter update, or ground-truth
data route exists. Every returned epoch includes per-iteration and final normal
matrix rank/condition diagnostics plus an explicit convergence status.

Run only the intentionally small seed-0 smoke (accepted epochs 0, 1, and 2):

```bash
.venv/bin/python -m validation.ibiza_generalization.smoke
```

This command runs three checkpoints total, prints diagnostics without writing
an accuracy artifact, and makes no positioning-performance claim.

## Full-day frozen batch runner

`full_day_batch.py` is the deterministic export layer for all 30 frozen
architecture/seed combinations. It loads the verified Ibiza NPZ once and
shares that same in-memory dataset and chronological accepted-epoch sequence
with every model. Scientific calculations continue to come exclusively from
`inference.py`; the runner does not preprocess observations, normalize Ibiza
features, train, create an optimizer, call backward, change checkpoints, use
ground truth, or alter WLS.

The tracked `full_day_batch_manifest.json` fixes the dataset and checkpoint-
manifest hashes, 30-job ordering, runtime provenance, common epoch policy,
result layout, and exact row schema. Generated files are outside Git under:

```text
../external_data/ibiza_2025_01_01/results/frozen_tdl/
```

Each `tdl_<architecture>_seed_<seed>.jsonl` contains exactly one row per
accepted epoch. A WLS/evaluation exception is retained in place with its epoch
identity, source metadata, `solution_status="exception"`, error type/message,
and nullable unavailable fields. The runner never filters the common epoch
sequence. A deterministic `run_manifest.json` records file hashes, row and
satellite counts, failure counts, runtime versions, and scientific controls.

The full run requires an explicit safety acknowledgement and is not launched
by the test suite. Run it manually from the repository root:

```bash
.venv/bin/python -m validation.ibiza_generalization.full_day_batch \
  --dataset ../external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz \
  --output-dir ../external_data/ibiza_2025_01_01/results/frozen_tdl \
  --confirm-full-run
```

If that exact result set already exists and replacement is intentional, append
`--overwrite`. A completed run with retained exception epochs exits nonzero
after writing every job and marks the manifest
`completed_with_failed_epochs`.

## Runtime provenance

The compatibility target is:

- TDL-GNSS commit `dd5eac669676ba0a922102047e58c2dfc9be9267`;
- pyrtklib 0.2.6 at commit-hypothesis
  `916d3cc8eb202718a16097cea4a5729bd6b27ac5`.

As in the earlier real-KLT validation, the pyrtklib revision is a validated
hypothesis, not a proven publication dependency. The generated compatibility
runtime is stored persistently under `../external_data/.paper_runtime/` so it
survives a reboot. It is reconstructed from pinned commits without modifying
either external reference repository and is never installed globally.

Prepare it once, or re-run this idempotent command whenever Python, the
operating system, or CPU architecture changes:

```bash
.venv/bin/python -m validation.ibiza_generalization.prepare_runtime
```

The command validates and reuses a compatible cache. If it is absent or
incompatible, it builds a replacement atomically. The generated cache contains
its own `README.md` and `runtime_manifest.json`, explaining its purpose and
recording its pinned commits, platform signature, and artifact hashes. It can
be deleted safely and reconstructed with the same command.

## Generate the frozen dataset

From the repository root, after preparing the persistent runtime:

```bash
.venv/bin/python -m validation.ibiza_generalization.preprocess \
  --runtime-dir ../external_data/.paper_runtime \
  --observation ../external_data/ibiza_2025_01_01/raw/observation/IBIZ00ESP_R_20250010000_01D_30S_MO.rnx \
  --navigation ../external_data/ibiza_2025_01_01/raw/navigation/BRDM00DLR_S_20250010000_01D_MN.rnx \
  --output ../external_data/ibiza_2025_01_01/derived/ibiza_preprocessed.npz \
  --manifest validation/ibiza_generalization/ibiza_preprocessed_manifest.json
```

All paths shown above are the defaults, so the preprocessing command may be
shortened to:

```bash
.venv/bin/python -m validation.ibiza_generalization.preprocess
```

No personal absolute path is written to the scientific manifest. The NPZ and
runtime cache both live outside Git.

The writer fixes ZIP member order, timestamps, permissions, NPY encoding, and
compression settings. Repeated runs on the same runtime and inputs therefore
produce identical arrays, the same canonical content hash, and identical NPZ
bytes/SHA-256.

## Paper-era semantics retained

The RINEX observation header advertises `C1C/L1C/S1C` first for GPS,
GLONASS, and Galileo. pyrtklib maps those measurements to `P[0]`, `L[0]`, and
`SNR[0]`; no alternate signal is substituted. The paper-era single-frequency
`prange()` code/TGD corrections, `satposs()` broadcast satellite states,
chronological `sortobs()` plus 0.05-second `split_obs()`, seven-state
constellation-clock layout, physical NumPy OLS atmosphere path, explicit
normal-equation inverse, convergence threshold, and residual rejection are
preserved.

An observed row for which that exact historical path cannot form a satellite
state is recorded as `no_broadcast_ephemeris`; it is never silently replaced
with another signal or Galileo ephemeris-selection mode. Rejected epochs and
all raw satellite rows remain represented in numerical audit arrays with
manifest-defined reason codes.

## NPZ organization

The NPZ contains numerical arrays only. The principal groups are:

- `features` and aligned per-satellite arrays: source row/observation indices,
  repeated epoch timestamp, RTKLIB satellite number, numeric
  constellation/PRN, raw and corrected pseudorange, satellite ECEF and clock
  terms, C/N0, elevation, residual, and clock-state index;
- accepted-epoch arrays: offsets, timestamps, split indices, raw/accepted
  counts, seven-state OLS initializer, active-state mask/count, design rank,
  `cond(H)`, `cond(H.T@H)`, and convergence status;
- raw-epoch arrays: timestamp, observation count, acceptance status, rejection
  reason, and accepted-epoch mapping;
- `audit_*` arrays: one row per raw RINEX satellite-epoch observation, retaining
  deterministic source identity, first-signal values, status, and exclusion
  reason.

The complete names, dtypes, and shapes are recorded in
`ibiza_preprocessed_manifest.json`. Numeric constellation and exclusion codes
are also defined there.

## Validation

Run the scientific/unit controls with:

```bash
.venv/bin/python -m pytest -q \
  tests/test_ibiza_generalization.py \
  tests/test_ibiza_frozen_inference.py \
  tests/test_ibiza_full_day_batch.py
```

The controls cover deterministic NPZ bytes and canonical content, structural
ground-truth independence, identical raw TDL-B/W/BW adapters, checkpoint
inventory/hash enforcement, frozen normalization, inference repeatability and
immutability, released output semantics, row ordering, feature units, rank and
condition diagnostics, RINEX first-signal mapping, and a KLT1 compatibility
smoke test when the disposable KLT environment is available. The KLT smoke
tolerance is `1e-10` absolute for each raw feature and `1e-7 m` absolute for
the seven-state OLS initializer.

For a full-data determinism check, run the generation command twice to two
temporary output/manifest paths and compare both NPZ SHA-256 values. The
tracked manifest records the accepted full-day result and hashes.
