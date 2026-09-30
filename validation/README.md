# Synthetic differentiable WLS validation

This milestone validates a small, independent mathematical implementation of
GNSS weighted least squares (WLS). It uses no real observations, RTKLIB solve,
satellite selection, or Top-k operation. The implementation was informed by
Hu et al. and the paper-era TDL-GNSS reference, but no upstream source file was
copied.

## Reproduce

From the repository root:

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python experiments/synthetic_precision_learning.py --seed 20260929
```

The accepted run uses Python float64 throughout, 48 synthetic training epochs,
24 held-out synthetic epochs, and 60 optimizer epochs. PyTorch is installed in
the ignored local `.venv`; it is not vendored.

## State, units, and equations

There is one constellation and one receiver clock state:

```text
p = [x, y, z, beta]
```

The position is ECEF in metres. `beta` is receiver clock bias expressed as a
range in metres. For satellite ECEF position `s_i`, the prediction and
linearization are

```text
h_i(p) = ||s_i - r||_2 + beta
H_i    = [(r - s_i)^T / ||r - s_i||_2, 1]
v      = z - h(p)
```

The residual sign is therefore observed minus predicted. A positive state
increment is applied at each Gauss-Newton iteration:

```text
Lambda  = diag(lambda_i)
N       = H^T Lambda H
g       = H^T Lambda v
delta_p = solve(N, g)
p_next  = p + delta_p
```

Each `lambda_i > 0` is measurement precision, not a square-root weight. If
absolute noise variances are known it has units of inverse square metres. In
the learning experiment it is only a bounded relative precision, because a
common positive scale cancels from the WLS state.

## Synthetic data assumptions

The receiver is fixed at a deterministic ECEF point. Eight satellites are
constructed from well-spread local azimuth/elevation angles and placed on a
nominal 26,560 km orbital radius. Geometric ranges are approximately
19--27 Mm. Pseudoranges contain a configurable receiver clock bias and
fixed-seed Gaussian error.

For precision learning, each epoch rotates and slightly jitters the geometry.
Two of eight pseudoranges receive an additional deterministic-sign error whose
magnitude is sampled from 14--24 m; ordinary error has 0.45 m standard
deviation. The feature vector is

```text
[quality_indicator, elevation, equal-precision OLS residual]
```

`quality_indicator` is an abstract synthetic value explicitly correlated
with the injected degradation. It is not physically modelled C/N0. The
elevation and residual are computed from an independent equal-precision NumPy
WLS solve initialized at the fixed ECEF origin. The ground-truth position is
used only in the position loss and evaluation.

Satellite clocks, ionosphere, troposphere, Earth rotation, relativity,
ephemeris error, receiver physics, and multiple-constellation clock offsets
are intentionally zero or absent. This is a mathematical test fixture, not a
full GNSS simulator.

## Literature basis and project choices

| Item | Basis or choice |
|---|---|
| Learned per-measurement weighting inside iterative GNSS WLS | Directly motivated by Hu et al. and the paper-era TDL-GNSS computational chain |
| Three inputs: quality/CN0-like value, elevation, OLS residual | Paper-inspired structure; the first value here is synthetic quality, not C/N0 |
| Position loss backpropagated through WLS | Directly motivated by Hu et al. |
| Four-state, one-constellation synthetic model | Project simplification for this milestone |
| Eight fixed-scale synthetic satellites and noise distributions | Project test-fixture choices, not literature parameters |
| Three WLS iterations during learning and six during the oracle check | Project implementation choices |
| 3-to-8-to-1 tanh network and Adam settings | Minimal project choices, not a reproduction of a published architecture |
| Sigmoid-bounded precision in [0.02, 1.00] | Paper-inspired positivity with project-selected bounds |
| Rank and condition rejection threshold | Project numerical safeguard |

The primary TDL-GNSS source snapshot inspected was
`dd5eac669676ba0a922102047e58c2dfc9be9267`; the final preserved old
implementation `76d9b684e1f7a60326514b1796ebfd829b532d46` was used for
comparison. Full external-reference revisions are recorded in
`PROVENANCE.md`.

The implementation uses `torch.linalg.solve(N, g)` rather than explicitly
forming `N^{-1}`. A solve is mathematically equivalent for a nonsingular
system and avoids the additional numerical error and work of materializing an
inverse. There is deliberately no pseudoinverse fallback: rank-deficient or
unacceptably conditioned geometry raises `WLSGeometryError`.

Current TASGNSS code was not copied because the audited `WH/Wv` construction
applies a predicted diagonal value on both sides of the normal equations,
making its effective precision proportional to the square of that value.
This milestone follows the stated `H^T Lambda H` formulation with
`Lambda` appearing once.

## Phase A: forward validation

The Torch and independently written NumPy solvers are compared after every
iteration for `h`, `v`, `H`, `lambda`, `Lambda`, `N`, `g`,
`delta_p`, state, singular values, rank, and condition number.

Every validation solve starts from the fixed ECEF origin; the true state is
not an initializer. For seed 11, the maximum absolute discrepancies were:

| Quantity | Maximum absolute | Relative infinity-scale discrepancy |
|---|---:|---:|
| predicted pseudorange | 1.118e-08 | 4.208e-16 |
| residual | 1.118e-08 | 1.823e-15 |
| Jacobian | 1.166e-15 | 1.166e-15 |
| precision / precision matrix | 0 | 0 |
| normal matrix | 4.996e-15 | 6.245e-16 |
| RHS | 3.055e-08 | 7.956e-16 |
| state increment | 2.421e-08 | 4.112e-15 |
| updated state | 2.421e-08 | 4.112e-15 |
| singular values | 3.997e-15 | 3.014e-16 |
| final state | 2.794e-09 | 5.769e-16 |

The relative value is the maximum absolute difference divided by the maximum
absolute NumPy value for that field across all iterations. Nanometre-scale
absolute differences arise when independently evaluated float64
approximately-20 Mm ranges are subtracted near convergence.

## Phase B: gradient and invariant validation

The autograd precision-gradient norm was `9.485023e+00`. Central finite
differences gave:

| Epsilon | Maximum absolute gradient error | Relative infinity-norm error |
|---:|---:|---:|
| 1e-2 | 5.602e-04 | 1.110e-04 |
| 1e-3 | 3.081e-05 | 6.102e-06 |
| 1e-4 | 1.931e-04 | 3.824e-05 |

The non-monotonic smallest-epsilon error is expected from float64 subtraction
and confirms why several step sizes are checked.

Other fixed-seed results:

- multiplying every precision by 37 changed the state by at most
  `2.141e-09 m`;
- the normalized loss-gradient projection along the scale direction was
  `1.925e-16`;
- a common satellite-row permutation changed the state by at most
  `6.345e-09 m`;
- a single +20 m error produced `18.735 m` position error at equal
  precision and `0.0312 m` when its precision was reduced to `1e-3`;
- changing ground truth after the solve changed no solver state.

Good geometry had rank 4 and normal-matrix condition number `199.04`.
Duplicate geometry was rejected at rank 1. The deliberately near-singular
rank-4 case was rejected at condition number `4.805e12`. An exactly
determined four-satellite case solved without a pseudoinverse, and multiplying
all precisions down to `1e-14` changed the final state by only
`3.725e-09 m`.

## Phase C: tiny precision network

The network is shared across satellite rows:

```text
3 standardized features -> Linear(3, 8) -> tanh -> Linear(8, 1)
                        -> sigmoid -> precision in [0.02, 1.00]
```

With seed 20260929:

| Metric | Result |
|---|---:|
| Initial training mean squared 3D error | 611.173 m2 |
| Final training mean squared 3D error | 160.814 m2 |
| First / final parameter-gradient norm | 354.628 / 8.643 |
| Equal-precision held-out mean 3D error | 25.447 m |
| Learned held-out mean 3D error | 9.386 m |
| Held-out improvement | 63.11% |
| Mean clean precision | 0.9777 |
| Mean corrupted precision | 0.02168 |
| State change after multiplying learned precisions by 23 | 2.132e-14 m |

These results demonstrate only that the synthetic computational chain can
learn useful relative precisions and receive gradients through the WLS solve.

## Limitations

This single-seed experiment is intentionally easy: the quality indicator is
strongly correlated with injected errors, the geometry is always
well-distributed, and train/test data share one generator. It does not
establish calibrated uncertainty, robustness to real receiver effects,
cross-dataset generalization, NLOS detection, RTKLIB agreement, or
satellite-selection performance. It has no discrete selection mechanism and
makes no exact-cardinality claim.
