# TASGNSS Ibiza solver-stack audit

This directory contains the solver-focused Ibiza audit of historical
differentiable Torch-WLS behavior versus the current TASGNSS stack, including
neutral TASGNSS analysis. It is not the current neural-network training
experiment.

This audit preserves a crucial provenance boundary: current TASGNSS at
`fdd7e8e` is a later refactor, not the original Hu paper-era implementation.
No neural model, learned bias, learned weight, training operation, or ground
truth enters positioning.

## Compatibility decision

Strict same-input Phase A is stopped. The frozen Ibiza NPZ has the exact paper
rows, code-bias-corrected pseudoranges, satellite positions and clocks, clock
mapping, and OLS states. It does not have the ionosphere and troposphere arrays
that current TASGNSS freezes during `preprocess_obs`. Those arrays are evaluated
at a separate RTKLIB `pntpos` receiver position. Recomputing them from RINEX and
navigation data would be a new preprocessing result rather than recovery of a
frozen same input. Zeroing them would omit corrections required by the current
model. Passing the already-corrected values as raw observations would cause
`prange` to apply code-bias/TGD corrections again.

The seven-state paper solution does have an exact state embedding into current
TASGNSS: `[x,y,z,G,C,E,R]` maps to the matching entries of
`[x,y,z,G,C,E,R,J,I,SBAS]`, with the three absent clock systems inactive. That
does not cure the missing fixed-correction inputs.

## Old versus current solver

| Property | Paper OLS | Historical neutral Torch-WLS | Current TASGNSS |
|---|---|---|---|
| Entry/model | `get_ls_pnt_pos` / `H_matrix_prl` | `get_ls_pnt_pos_torch` / `H_matrix_prl_torch` | `wls_pnt_pos` / `pseudorange_observe_func` |
| State | XYZ + active G/C/E/R clocks | same seven slots, active columns | XYZ + seven G/C/E/R/J/I/SBAS clock slots; public solver starts all at zero |
| Earth rotation | inside RTKLIB `geodist` | explicit Sagnac in Torch range | Sagnac precomputed once at `pntpos` position |
| LOS/elevation | `geodist` fills LOS before `satazel` | LOS passed to `satazel` is never populated | `geodist` fills LOS before `satazel` |
| Atmosphere | broadcast ionosphere + Saastamoinen, updated each iteration | established zero-delay defect | broadcast ionosphere + Saastamoinen, frozen before iterations |
| Code bias | `prange`: DCB/TGD/BGD | same preprocessing | `prange`: DCB/TGD/BGD |
| Iteration/convergence | 10 / `1e-4 m` | 10 / `1e-4 m` | 20 / `1e-3 m` |
| Linear algebra | explicit inverse of `H.T@W@H` | same, Torch | NumPy `lstsq(W@H,W@v)`; Torch pseudoinverse |
| Rank/condition rejection | none in released function | none | none; adapter reports diagnostics only |
| Custom weight meaning | direct precision `W` | direct precision `W` | residual multiplier; effective precision `W.T@W` |

Current TASGNSS therefore removes the exact unpopulated-LOS mechanism, but it
does not update atmospheric corrections within its iterative solve. Its
atmosphere and Sagnac values are constants, including in Torch mode.

## Phase B

Because Phase A is impossible, `run_current_stack.py` performs the distinct
`CURRENT-STACK END-TO-END BASELINE` on the original Ibiza RINEX files. It uses
the documented default `origin` backend, current pyrtklib 0.2.7 built from the
read-only reference commit in a disposable copy, the first pseudorange slot,
current TASGNSS preprocessing, `w=1`, and `b=None`.

The current preprocessing's `pntpos` call uses `prcopt_default` only to anchor
fixed corrections (GPS `navsys`, 15-degree mask, ionosphere/troposphere off in
the selected backend). The later TASGNSS atmosphere functions then explicitly
apply broadcast ionosphere and Saastamoinen terms to all retained systems. The
main iterative solver itself starts from an all-zero ECEF/clock state.

Run from the comparison repository root:

```bash
PYTHONPATH=src:. .venv/bin/python -m \
  validation.tasgnss_ibiza_solver_audit.run_current_stack
```

Generated artifacts live outside Git under
`../external_data/tasgnss_comparison/ibiza_neutral/`. The runner writes
`current_stack_positions_pre_ground_truth.npz` before opening the frozen EPN
reference, then produces accuracy, paired summaries, an epoch NPZ, a dynamic
audit/manifest, and hashes. Phase-B displacement from the paper OLS initializer
is reported only as an end-to-end descriptive quantity; it is not labelled a
solver-only result.

## Result

The current-stack run solved all 2,880 raw epochs. Results must be split by
denominator:

| Solution / slice | Epochs | 2D RMS (m) | 3D RMS (m) | 3D median (m) |
|---|---:|---:|---:|---:|
| Paper OLS, frozen accepted epochs | 2,856 | 8.3736 | 16.5034 | 4.4225 |
| Historical neutral Torch-WLS, same epochs | 2,856 | 6.8949 | 35.7408 | 31.7095 |
| Current TASGNSS, common paper-accepted epochs | 2,856 | 8.4081 | 16.4614 | 4.4235 |
| Current TASGNSS, all raw epochs | 2,880 | 173.0630 | 412.1404 | 4.4689 |

On the 2,856 common epochs, current-stack displacement from the paper OLS
position is 0.2472 m mean, 0.0477 m median, 0.0693 m P68, 0.2890 m P95, and
40.9393 m maximum. The established historical neutral Torch displacement is
37.7112 m mean, 33.5635 m median, 40.7076 m P68, 68.5852 m P95, and 236.6281 m
maximum. Current support exactly matches paper support and order in 2,840 of
2,856 epochs, so this remains descriptive end-to-end evidence, not Phase A.

The 24 additional epochs were precisely the raw epochs rejected by paper OLS
preprocessing. Current TASGNSS reported all 24 as solved, but their 3D errors
have 333.10 m median, 4,511.20 m RMS, and 21,910.50 m maximum. They explain the
large all-raw RMS and must not be silently pooled into the common-epoch table.

Thus the later current stack stays dramatically closer to the good initializer
than the historical neutral Torch path on common epochs, and the exact old
unpopulated-LOS mechanism is absent. Because fixed atmospheric inputs and some
satellite supports differ, this run does **not** establish that solver changes
alone removed the degradation. It also reveals that current convergence status
alone does not reject the 24 catastrophic paper-rejected epochs.
