# Frozen synthetic results

Main run: 4 October 2026 UTC, seed 2026100403. 51 independent LPs, 61 reference
solves (ten dense/CSR pairs), 116 projections, 24 grouped configurations.
Five repetitions per configuration; three at 2000 x 3000.

All main LP solves are optimal. Accepted original-feasible vectors: 0/116 raw,
0/116 dual recovery, 0/116 primal recovery. Speedups concern relaxations, not
successful original-LP solution. See the [main findings](../../README.md).

| File | Meaning |
|---|---|
| main_raw.jsonl | Unmodified measurements: 61 original and 116 projected rows |
| main_protocol.json | Frozen cases, repetitions, sketch/storage settings, time cap |
| main_summary.json | Unmodified means, sample SDs, ranges, medians and counts |
| main_vectors_*.npz | 51 lossless archives of primal vectors, equality duals, recovered candidates |
| main_environment.json | Recorded environment; private helper paths normalized |
| source_provenance.json | Original/portable runner identities and kernel equivalence |
| verification.json | Recomputed metrics, primal-dual checks, counts and summaries |
| artifact_hashes.json | SHA-256 checksums for delivered evidence |
| pilot_raw.jsonl | Separate calibration: 3 originals, 2 projections, one 120-second original timeout; excluded |

No measured time, numerical observation, protocol, vector archive, or summary
was edited. Only environment path keys and explanatory publication fields were
normalized. The portable CLI is an export, not a claim of new timing execution.

## Metrics

Bound gap = (z_original-z_projected)/max(1,abs(z_original)).
Candidate obj_gap = abs(c@x-z_original)/abs(z_original) when defined;
for infeasible candidates this is not an approximation guarantee.
feas = ||Ax-b||_1/||b||_1; neg = negative mass/||x||_1, not a fraction of
negative coordinates. Acceptance checks the scaled infinity residual and
minimum coordinate at tolerance 1e-7.

With equality marginal lambda, the lifted dual is S.T@lambda. The verifier
checks c-A.T@S.T@lambda >= -1e-6, projected equality feasibility, and the
normalized gap between c@x_projected and b.T@S.T@lambda. These numerical
checks do not constitute exact-arithmetic certificates.

Projected total = sampling + products + projected solve. Recovery totals add
the respective recovery. Generation, validation, and file writing are excluded.
Sparse basis conversion is included in recovery; storage excludes solver
internals, temporary allocations, and peak RAM.

Default verification is read-only:

```sh
python experiments/verify_synthetic.py
```

Use pinned requirements. Fresh runs need a new output directory; see
[execution instructions](../../experiments/README.md).
