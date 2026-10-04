# Running and checking the experiments

Run from the repository root using Python 3.12 and pinned requirements:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Current synthetic study

Verify [saved results](../results/synthetic) without rerunning LP optimization:

```sh
.venv/bin/python experiments/verify_synthetic.py
```

The verifier regenerates instances/sketches and checks original/projected
feasibility, dual feasibility and gaps, candidate metrics, timing sums, storage,
all 24 summary groups, complete configuration counts, and source/artifact hashes.
It shares constructors with the benchmark; it is not an independent solver or
exact-arithmetic proof. Default verification is read-only.

Run new measurements in a fresh directory:

```sh
.venv/bin/python experiments/synthetic_benchmark.py --output-dir results/rerun-synthetic-local
.venv/bin/python experiments/verify_synthetic.py --output-dir results/rerun-synthetic-local --write-summary
```

The recorded main run took about 34 minutes. The runner refuses nonempty
directories. --write-summary writes a missing summary and verification record;
it never changes raw observations/vectors. Existing summaries must agree with
recomputed values. --protocol PATH and --seed INTEGER select explicitly
different experiments; the default is results/synthetic/main_protocol.json.

Small plumbing test, not a performance measurement:

```sh
.venv/bin/python experiments/synthetic_benchmark.py --smoke --output-dir results/rerun-smoke-local
.venv/bin/python experiments/verify_synthetic.py --output-dir results/rerun-smoke-local --write-summary
```

Both LPs use highs-ds, presolve, tolerance 1e-7, and a 300-second main limit.
Single-thread settings are requested, but actual Apple Accelerate thread usage
was not independently measured. Future nonoptimal original solves cause
explicit skipped-projection records; primary outcomes are not replaced.

The two existing recovery heuristics remain unchanged. All recorded 116
projections are finite, but raw and both recovered candidates have zero
accepted original-feasible results.

## Netlib: unchanged limitation study

Fetch pinned inputs, then verify saved vectors and certificates:

```sh
.venv/bin/python experiments/fetch_netlib.py
.venv/bin/python experiments/verify_netlib.py
```

Optional new run, separate from committed evidence:

```sh
.venv/bin/python experiments/netlib_benchmark.py --output-dir results/rerun-netlib-local
.venv/bin/python experiments/verify_netlib.py --output-dir results/rerun-netlib-local
```

Seed 20261004; five fixed LPs; three projectors; row ratios 0.5/0.75; ten seeds:
300 projections plus five separate controls, not 300 independent application
datasets. The MPS reader is private to the pinned SciPy version.

## Superseded utilities and documents

rerun.py, diagnose_statuses.py, and verify_results.py belong to the old
September protocol. The unchanged rerun.py is also an imported helper for the
current synthetic and Netlib code. Old measurements remain in
[the pre-replacement snapshot](https://github.com/Jamil997/random-projections-lp-course-project/tree/25c932497f4fcecaeebd7ca58be035ce2a03ef68).
Use that checkout for the old study and synthetic_benchmark.py for the current one.

render_report_tables.py and render_netlib_tables.py are legacy document
utilities, not publication steps for this synthetic schema. No report or
presentation is stored or updated by this replacement; do not run these
generators expecting them to incorporate the new study.

See [provenance](../PROVENANCE.md).
