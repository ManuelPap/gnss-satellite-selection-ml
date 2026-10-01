# Provenance

This repository is an independent implementation informed by GNSS and
machine-learning literature. External repositories are read-only scientific
references and no upstream source file is vendored or copied here.

The initial differentiable-WLS milestone began from repository commit
`6299e69` on branch `research/pyrtklib-differentiable-wls`. The subsequent
real-data WeightNet reproduction is developed on branch
`research/paper-weightnet-reproduction` after frozen real-KLT
observation-model validation commit `65ac942`.

## External reference worktrees

### pyrtklib

- Repository: https://github.com/IPNL-POLYU/pyrtklib
- Cloned revision inspected: `1c468dbe14074f1b7b3276ce265fe0fa5d6bef8b`
- Role: Hu et al. scientific and preprocessing context
- Classification: current inspected reference; not assumed to be the exact
  publication-era snapshot

### TDL-GNSS

- Repository: https://github.com/ebhrz/TDL-GNSS
- Current cloned revision:
  `a640b2832c90daeb1ce25644a1da91c8edeb13fc`
- Primary paper-era implementation reference:
  `dd5eac669676ba0a922102047e58c2dfc9be9267`
- Final preserved old implementation used for comparison:
  `76d9b684e1f7a60326514b1796ebfd829b532d46`
- Role: computational-chain, feature, and differentiable-WLS reference

Historical files were inspected with `git show <revision>:<path>`; the
reference worktree was not checked out to a different revision.

### TASGNSS

- Repository: https://github.com/PolyU-TASLAB/TASGNSS
- Cloned revision inspected: `fdd7e8ebc0019ad9b7c73f31363de066290d057a`
- Role: later/current-equivalent comparison only
- Classification: later refactor, not a source copied into this milestone

The audited TASGNSS `WH/Wv` form was intentionally not reproduced because
using the same predicted diagonal value on both sides changes the effective
precision to its square. This project implements `H^T Lambda H` with
`Lambda` appearing once.

## Independence boundary

RTKLIB/pyrtklib preprocessing is outside autograd in the referenced design.
The first synthetic milestone went further in isolation: it had no RTKLIB
input or operation at all. Synthetic constants fed an independently written
PyTorch WLS solve.

The `validation/paper_weightnet` milestone deliberately adds real public KLT
preprocessing at the historical-source hypotheses above. Raw GNSS data,
feature caches, traces, and trained checkpoints remain local and ignored.
Only independently written reproduction code and small machine-readable audit
records belong in this repository. Full hashes, the paper-versus-code audit,
the 405/404 KLT3 discrepancy, and rerun instructions are recorded in
`validation/paper_weightnet/README.md`.
