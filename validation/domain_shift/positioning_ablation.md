# Frozen-output Ibiza positioning ablation

This controlled experiment uses the immutable per-row Ibiza outputs from the
completed model-response probe. It does not rerun a neural network, deserialize
or change a checkpoint, refit normalization, alter preprocessing, or change
the accepted epoch/satellite support.

The positioning stage uses the existing historical implementations:

- `solve_paper_bias_position`: subtracts the frozen metre-valued bias from the
  already corrected pseudorange, then applies identity-weight WLS;
- `solve_paper_weighted_position`: uses the supplied output directly as the
  diagonal of `W` in `(H.T @ W @ H)^-1 @ H.T @ W @ v`;
- `solve_paper_hybrid_position`: applies both operations with the released
  TDL-BW `(weight, bias)` semantics.

Every solve starts from the same stored seven-state
`epoch_ols_initial_state`. Active states are ECEF indices 0–2 plus the sorted
constellation clock indices present in that epoch (3–6). The convergence
tolerance is `1e-4`, the maximum is ten iterations, and success requires a
finite state, a full-rank final normal matrix with finite condition number,
and convergence before the iteration maximum.

Uniform weight is `1.0`. Constant positive scaling cancels algebraically from
the historical gain; the experiment also validates this numerically on 64
evenly spaced accepted epochs.

The neutral solve is not assumed to equal `epoch_ols_initial_state`. That
stored state is the output of preprocessing's `get_ls_pnt_pos()` path, which
includes RTKLIB ionosphere/troposphere corrections. The released Torch WLS
reproduction starts from it but uses the already-established zero-atmospheric
term observation model. The generated neutral regression quantifies the
resulting difference.

Ground truth is deliberately absent from input validation and all positioning
functions. Only after every original position reproduces the published frozen
inference result and the neutral controls pass is the propagated EPN reference
loaded for ENU/error evaluation.

The original-position gate uses `5e-7 m` absolute state tolerance. This is a
sub-micrometre, empirically justified same-path tolerance: the immutable
response artifact was forwarded as one full 73,204-row batch, whereas the
published inference forwarded one epoch at a time. A validation-only check of
the first strict failure found a maximum weight difference of `4.74e-20` and
exact bias; highly conditioned weighted normal matrices amplified it to at
most `2.03e-7 m` across all TDL-BW seeds. Solved status, rank, and iteration
count remained exact. This diagnostic did not alter or regenerate an artifact.

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib-positioning-ablation \
  .venv/bin/python -m validation.domain_shift.positioning_ablation
```

The command refuses to overwrite an existing directory and atomically
publishes to `../external_data/domain_shift/positioning_ablation/`.

Paired deltas are always `candidate/original error - reference-variant error`.
Negative values improve the error and positive values worsen it. Equality is
defined by absolute delta at most `1e-12 m`. Seeds are summarized separately
before any across-seed aggregation. The TDL-BW factorial quantity is the exact
numerical contrast on squared 3D position error; it is not described as a
physical causal interaction.
