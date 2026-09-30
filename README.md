# GNSS Satellite Selection ML

Independent experiments for machine-learning methods applied to GNSS
positioning and, in later milestones, exact-cardinality satellite selection.

## Current milestone

The repository currently contains the smallest synthetic differentiable WLS
baseline:

- deterministic one-constellation GNSS-scale geometry;
- independent NumPy and differentiable PyTorch float64 solvers;
- iteration-level forward-oracle tests;
- autograd versus central finite-difference gradient tests;
- scaling, permutation, bad-measurement, conditioning, and leakage checks;
- a tiny network that learns bounded positive relative precisions from
  synthetic quality, elevation, and equal-precision OLS residual features.

This milestone deliberately excludes real GNSS observations, RTKLIB
comparison, Top-k, and satellite selection. The synthetic quality indicator is
not real C/N0, and the reported result is not a real-world positioning claim.

## Run

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python experiments/synthetic_precision_learning.py --seed 20260929
```

The experiment prints the forward discrepancies, gradient errors, invariant
and conditioning diagnostics, and held-out precision-learning metrics. See
[validation/README.md](validation/README.md) for equations, assumptions,
fixed-seed results, provenance distinctions, and limitations.
