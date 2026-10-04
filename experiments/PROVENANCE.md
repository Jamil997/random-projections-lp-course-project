# Experiment provenance

The current synthetic entry point is synthetic_benchmark.py, with evidence in
[results/synthetic](../results/synthetic). Its 116 projections replace the old
September synthetic baseline in the current tree, with old results in Git history.

See [repository provenance](../PROVENANCE.md) for thesis lineage, protocol
changes, source-export identity, pilot exclusion, and document boundaries.
The [original thesis repository](https://github.com/Jamil997/Random_Projections_master_thesis)
is preserved at inspected commit 3d2ff962612e1e5d76b07ccbade889576ba76416.

## Interpretation rules

- A finite projected optimum is a lower bound in exact arithmetic; numerical
  checks use finite tolerances.
- Acceptance requires ||Ax-b||_inf <= 1e-7 max(1,||b||_inf) and min(x) >= -1e-7.
  Neither existing recovery heuristic enforces nonnegativity.
- Infeasible candidates' objective discrepancies are not approximation guarantees.
- Projected wall-clock time includes construction, multiplication, and solving;
  recovery is additional. Array storage is not peak process RAM.
- Single-thread settings are requested, not independently verified on Accelerate.
- Netlib reuses five fixed LPs; the new synthetic study has 51 independent LPs.
- Old infeasible synthetic families are historical, not part of this feasible,
  positive-cost study. Unchanged rerun.py remains an implementation dependency.
