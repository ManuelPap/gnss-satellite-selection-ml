# Official RTKLIB Ibiza controlled baseline

This package implements a conventional single-point-positioning control for
the frozen current-stack Ibiza experiment. It never retrains, reruns, or
modifies the TDL+TASGNSS experiment.

## Frozen scientific contract

The solve phase reads the complete 2,880-epoch observation RINEX and broadcast
navigation file. It does not accept a ground-truth path and serializes the raw
RTKLIB timestamps, ECEF estimates, solution quality, satellite count, input
hashes, configuration hash, and timestamp match before evaluation can open the
reference JSON.

Primary support is recovered from the `paired` mask, `candidate_epoch_id`, and
`epoch_time_gpst_like_s` arrays in each of the ten existing
`positions_pre_ground_truth/seed_*.npz` files. Every file hash is checked
against the frozen manifest, and the three support arrays must be byte-for-byte
identical across seeds. The resulting 2,849 `(epoch ID, GPST-like timestamp)`
pairs are hashed together. RTKLIB output is joined to this support by exact
timestamp value, never row number.

If even one required timestamp is absent, the primary result is labelled
incomplete and no primary accuracy metric or controlled table is computed.
There is no position-error filter.

## Configuration rationale

[`rtklib_config.conf`](rtklib_config.conf) freezes official RTKLIB SPP,
single-frequency code, GPS+GLONASS+Galileo, a predeclared 7 degree elevation
cutoff, no SNR mask, broadcast ionosphere/ephemeris, Saastamoinen troposphere,
no excluded satellites, and forward processing. The 7 degree mask was not
selected using Ibiza accuracy and must not be tuned using ground truth. These
choices otherwise match the systems and models used by the current TASGNSS
preprocessing as closely as RTKLIB permits. The full audit and irreducible
semantic differences are in
[`configuration_audit.json`](configuration_audit.json).

A pre-existing local, untracked `single-spp.conf` was observed before it was
intentionally deleted. It is historical provenance only, is not an input to
this experiment, and was not adopted: it used GPS+Galileo only, a 10 degree
mask, and `est-ztd`. Official `pntpos.c` treats `est-ztd` through its
Saastamoinen branch in SPP, but the constellation and mask choices excluded
observations present in the frozen TASGNSS artifact.

## Manual full run

After the official RTKLIB checkout hygiene issue reported by the preflight has
been resolved, run these as two explicit phases from the project root:

```bash
.venv/bin/python -m validation.rtklib_ibiza_controlled_baseline solve \
  --binary ../external_data/rtklib_ibiza_controlled_baseline/bin/rnx2rtkp
.venv/bin/python -m validation.rtklib_ibiza_controlled_baseline evaluate
```

Do not run `evaluate` until `solve` has completed and
`rtklib_positions_pre_ground_truth.npz` exists. The split is intentional: the
first command contains no route to the ground-truth document.
