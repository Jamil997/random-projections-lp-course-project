# Running and checking the experiments

## Netlib extension (October 2026)

The new run is in `results/netlib/`. Download the five pinned MPS inputs, then
verify saved vectors/certificates and check the generated report tables:

```bash
python experiments/fetch_netlib.py
python experiments/verify_netlib.py
python experiments/render_netlib_tables.py --check
```

To execute a new run without replacing the committed measurements:

```bash
python experiments/netlib_benchmark.py --output-dir results/rerun-netlib-local
python experiments/verify_netlib.py --output-dir results/rerun-netlib-local
```

Defaults: seed `20261004`, 10 sketches per configuration, 5 fixed LPs, ratios
0.5/0.75, Gaussian/sparse/CountSketch = 300 projected solves, plus five
full-dimensional controls. This is not 300 independent application datasets.
Results include JSON/CSV raw and aggregate records, a compressed NPZ of candidate
vectors and recession rays, environment/source/data hashes, and diagnostics.
Only finite projected optima enter recovery. Primary statuses are never replaced
by diagnostic outcomes. Unbounded-ray solves are untimed checks.

`verify_netlib.py` reconstructs sketches and independently recomputes scalar
quality metrics, all aggregate means/SDs and acceptance counts. It checks hashes,
CSV/JSON agreement, control vectors, original-row violations, projected rays,
paired original times, and simple transformation/projector unit tests. It
shares the benchmark's MPS reader, conversion and sketch constructors; it is
not an independent solver implementation or a formal proof of numerical output.
The MPS reader is a private SciPy interface, so use the pinned versions.

Generate the marked Netlib tables after deliberately selecting a new run with
`python experiments/render_netlib_tables.py --output-dir PATH`. This changes
only its four marked blocks, not the manually authored methods/interpretation.
Review those surrounding sections if a future run changes any conclusions.

## Environment setup

Run these commands from the repository root. The recorded baseline used Python
3.12; exact numerical-library versions are pinned in the root `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Inspect the delivered baseline

The repository already includes the fresh 300-projection run in
`results/baseline/`. No expensive computation is needed to check its provenance,
paired-instance integrity, status accounting, summary means and sample standard
deviations, or report-table consistency:

```bash
python experiments/verify_results.py
python experiments/render_report_tables.py --check
```

The verifier also checks that the recorded benchmark and diagnostic source hashes
still match the scripts. If these scripts are changed later, keep the old run as
historical evidence and perform a new run for the changed source.

## Perform a fresh rerun

Use a new output directory. The benchmark deliberately refuses any nonempty
directory, including the committed baseline, to preserve recorded evidence.

```bash
python experiments/rerun.py --seed 20260913 --repetitions 10 --output-dir results/rerun-seed-20260913
python experiments/diagnose_statuses.py --output-dir results/rerun-seed-20260913
python experiments/verify_results.py --output-dir results/rerun-seed-20260913
```

`--seed` must be nonnegative and `--repetitions` positive. The fixed grid contains
three matrix sizes and two densities. With `r` repetitions, it creates `18*r`
independent originals and `30*r` projected LP solves. Each original in an
infeasible family is shared by the sparse and orthogonal projectors. The reported
10-repetition run therefore contains 180 originals and 300 projected LP solves.
For a shorter smoke run, use `--repetitions 1` and a distinct output directory.

The diagnostic command examines unresolved primary outcomes with two alternative
solver configurations. It writes `status_diagnostics.json` without changing
`raw.csv`, primary counts, or benchmark times. A diagnostic command is needed
before the verifier, even if there were no unresolved cases. Re-running this
command replaces its diagnostic JSON, so use it only when intentionally refreshing
that diagnostic record.

## Regenerate the report tables

The table generator requires only the Python standard library. It reads
`raw.csv` and `summary.json`, then updates the five marked blocks in
`report/main.tex`: `feasible_times`, `feasible_quality`, `legacy_status`,
`stress_status`, and `key_results`. It preserves report text outside these
markers. The LaTeX document must load `booktabs`, `capt-of`, and `graphicx`.

```bash
python experiments/render_report_tables.py
python experiments/render_report_tables.py --check
```

To use a new run, pass its directory consistently:

```bash
python experiments/render_report_tables.py --output-dir results/rerun-seed-20260913
python experiments/render_report_tables.py --output-dir results/rerun-seed-20260913 --check
```

An optional `--report path/to/report.tex` selects another document with the same
five marker pairs. If the selected report is absent, generation writes
`experiments/generated_report_blocks.tex` for manual insertion; `--check` instead
fails without writing. A missing or duplicate marker in an existing document is
an error. `--check` is always read-only.

Generated blocks contain all numerical tables and a concise results paragraph.
Other report prose, dates, or methodological interpretation are not automatically
rewritten; review these if the seed, repetitions, methods or conclusions change.
The table generator checks exact block content, while `verify_results.py`
independently checks the underlying numerical summaries.

## Files and interpretation

- `raw.csv`: one row per projected LP, with original-instance/projector seeds,
  statuses, timing components, and available quality metrics.
- `summary.csv` / `summary.json`: grouped counts, means and sample standard
  deviations; speedups are ratios of mean times.
- `environment.json`: versions, settings, run dates, source hash, and legacy-code
  reference; no local account or installation paths.
- `deterministic_checks.json`: unboundedness examples, full-dimensional
  projection equivalence, and a dual-sign/complementarity check.
- `status_diagnostics.json`: separate untimed follow-up outcomes.
- `run.log`: progress timestamps.

The all-negative and mixed-sign families are separate experiments. An optimal
projected LP is a relaxation result; it does not guarantee feasibility for the
original constraints. Both equality feasibility and nonnegativity are required
for an accepted recovered candidate. Timings are machine-dependent, and primary
solver errors are not counted as preserved infeasibility. See
[PROVENANCE.md](PROVENANCE.md) for the original project reference and all protocol
changes.
