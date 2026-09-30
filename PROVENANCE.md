# Provenance

This repository is an independent implementation informed by GNSS and
machine-learning literature. External repositories are read-only scientific
references and no upstream source file is vendored or copied here.

The implementation milestone began from repository commit
`6299e69` on branch `research/pyrtklib-differentiable-wls`.

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
This milestone goes further in isolation: it has no RTKLIB input or operation
at all. Synthetic constants feed an independently written PyTorch WLS solve,
and only that solve and the tiny precision network participate in autograd.
