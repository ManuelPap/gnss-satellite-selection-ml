# Frozen KLT1/KLT2 pre-inference feature caches

`freeze_klt_features.py` exports the exact held-out KLT feature rows needed by
the later frozen-network response audit. Its scientific boundary is:

```text
audited KLT observation/navigation files
  -> TDL-GNSS dd5eac6 preprocessing under pyrtklib 0.2.6
  -> historical equal-weight OLS (`get_ls_pnt_pos`)
  -> [SNR[0]/1000, elevation_rad, OLS_residual_m]
  -> deterministic NPZ
  -> stop
```

No ground-truth path is accepted by the feature-only resolver or preparation
function. Ground truth remains part of the separate historical evaluation
wrapper, but is neither resolved nor read here. The generator does not load a
checkpoint, construct or call a neural model, normalize features, invoke
learned WLS, train, or compute a learned position.

The small refactor in `validation/paper_weightnet/held_out.py` exposes its
existing preprocessing/OLS loop as `prepare_feature_dataset()`. The original
`prepare_dataset()` now attaches ground truth after that feature stage. Feature
numerics, timestamp filters, constellation handling, row order, and
`get_ls_pnt_pos()` are unchanged.

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m validation.domain_shift.freeze_klt_features
```

The command refuses to overwrite its output directory. It executes the full
KLT1/KLT2 preprocessing twice in fresh temporary directories, requires exact
array equality and byte-identical deterministic NPZ files, then publishes:

```text
../external_data/domain_shift/klt_features/
  klt1_paper_features.npz
  klt2_paper_features.npz
  feature_cache_manifest.json
```

The expected cardinalities are 203 epochs / 4,676 rows for KLT1 and 209 epochs
/ 4,914 rows for KLT2. A mismatch aborts publication without filtering or
adjusting rows.

## NPZ schema

Both caches use the same schema. All raw numeric feature values are `float64`.

| Array | Scope | Meaning |
| --- | --- | --- |
| `features` | row × 3 | Exact raw feature matrix in required column order |
| `cn0_snr0_div_1000` | row | Column 0, with exact equality to `raw_snr_units / 1000` |
| `elevation_rad` | row | Column 1, final historical OLS elevation |
| `ols_residual_m` | row | Column 2, historical equal-weight OLS residual |
| `row_index` | row | Contiguous global retained-row index |
| `epoch_index` | row | Retained epoch index for every row |
| `row_index_within_epoch` | row | Retained order within its epoch |
| `epoch_offsets` | epoch + 1 | CSR-style row boundaries |
| `epoch_row_counts` | epoch | Rows per epoch |
| `epoch_times_gpst_like_s` | epoch | Observation epoch identity |
| `candidate_epoch_indices` | epoch | Strict-window candidate index |
| `split_epoch_indices` | epoch | Historical `split_obs` index |
| `satellite_ids` | row | Historical satellite identifier |
| `satellite_numbers` | row | pyrtklib numeric satellite identifier |
| `constellations` | row | System letter from `satellite_ids` |
| `raw_snr_units` | row | Observation `SNR[0]` before division by 1,000 |
| `raw_pseudorange_m` | row | Observation `P[0]` |
| `satellite_positions_ecef_m` | row × 3 | Historical broadcast satellite position |
| `satellite_clock_bias_s` | row | Historical satellite clock bias |
| `corrected_pseudorange_m` | row | Historical corrected pseudorange |
| `system_clock_indices` | row | Seven-state constellation clock slot |
| `initial_states` | epoch × 7 | Equal-weight OLS solution |
| `feature_column_names` | 3 | Ordered feature labels |
| `feature_column_units` | 3 | Ordered feature units |
| `dataset`, `schema_version` | metadata | Dataset label and schema version |

The regression authority is the existing full-held-out identity record created
by `paper_weightnet_seed_sensitivity.run_seed.prepared_dataset_identity()` from
the `PreparedDataset.epochs` returned by the historical held-out path. The
generator must reproduce these nine non-ground-truth arrays exactly:

1. `epoch_offsets`: row boundaries from all full-held-out epoch counts;
2. `epoch_times`: one observation timestamp per full-held-out epoch;
3. `features`: all raw feature rows in held-out order;
4. `satellite_ids`: retained satellite identifiers in that same order;
5. `satellite_positions`: retained broadcast ECEF satellite positions;
6. `satellite_clock_bias`: retained satellite clock biases;
7. `corrected_pseudorange`: retained historical corrected pseudoranges;
8. `system_clock_indices`: retained seven-state constellation clock slots;
9. `initial_states`: one full-support equal-weight OLS state per epoch.

The manifest records each historical hash beside its reproduced hash. It does
not use the historical `gt_times` or `ground_truth` hashes. Consequently, the
historical aggregate identity hash—which includes both ground-truth arrays—is
recorded for provenance but deliberately not reproduced.

The old eight-satellite KLT1 traces are not an independent numerical
regression oracle for the complete held-out features. Both perform GPS-only
filtering before OLS, while the held-out path solves the complete 17-satellite
multi-constellation epoch. OLS residuals depend on the complete observation and
design matrices and on the estimated state; changing measurement support
therefore changes residuals even for common satellites. This discrepancy is a
different preprocessing/support stage, not numerical error, and is never
handled by increasing a tolerance.

The traces remain supplementary diagnostics only. For their eight overlapping
GPS observations, the generator requires exact raw C/N0 identity and records
the elevation, residual, and OLS-state differences without treating them as
same-stage regressions. `paper_epoch_trace.npz`, which does not store the raw
three-column feature matrix, is additionally checked for common observation
identity such as epoch time, satellite IDs, pseudoranges, and satellite
states/clocks.
