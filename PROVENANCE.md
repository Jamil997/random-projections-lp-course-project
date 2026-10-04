# Relationship to earlier work

This course project builds on the author's M.Sc. thesis, *Matrix Sketching and
Linear Programming* (Nazarbayev University, 2025), and the accompanying
[GitHub repository](https://github.com/Jamil997/Random_Projections_master_thesis).
The inspected repository snapshot is commit
`3d2ff962612e1e5d76b07ccbade889576ba76416`, dated 14 November 2025. That date is
the repository snapshot date, not the thesis submission date.

Original sources:

- [feasible_instances.ipynb](https://github.com/Jamil997/Random_Projections_master_thesis/blob/3d2ff962612e1e5d76b07ccbade889576ba76416/feasible_instances.ipynb)
- [infeasible_instances.ipynb](https://github.com/Jamil997/Random_Projections_master_thesis/blob/3d2ff962612e1e5d76b07ccbade889576ba76416/infeasible_instances.ipynb)

The new scripts independently implement the stated methods. The old notebooks,
their saved outputs, the thesis PDF, and previous draft measurements are not
copied into this repository. The earlier repository is not modified.

## Retained theory and methods

The report retains the random-projection LP formulation, relevant JL background,
and the unboundedness result and counterexample developed in the thesis.
The report does not claim that this elementary unboundedness property is a new
literature result. Its proof works for every linear projection by inclusion of
feasible sets. Nonnegative recession directions are explicit, and the JL
dimension bound and concentration proof are corrected.

The experimental methods retain Gaussian projection on feasible problems,
sparse-valued and orthogonal projectors on infeasible problems, and the two
primal-vector recovery strategies.

## Changes in the new experiments

| Aspect | Previous notebooks | New implementation |
|---|---|---|
| Matrix generation | Fixed-count `scipy.sparse.random`, then dense conversion | Independent uniform entries times a Bernoulli mask; expected density specified |
| Feasible generation | `b=A x0`, nonnegative uniform `x0` | Same conceptual construction, reproducibly seeded |
| Infeasible generation | `b=-U[0,1]^m`, all entries negative | Retained as `legacy_all_negative`; additional `mixed_sign_stress` family |
| Dimensions | Saved notebook configurations use `m=1000`, `n=1400/1600`, `k=327/333` | Three smaller sizes, two densities, explicit half-dimension compression |
| Randomness | No recorded seed or environment | Independent reproducible data/projector streams, version pins, script SHA |
| Feasible solvers | SciPy HiGHS for original; CVXPY automatic solver for projection | SciPy HiGHS dual simplex for both |
| Equality dual | CVXPY equality multiplier used directly in recovery | SciPy equality marginal with checked reduced-cost convention |
| Dual recovery | Explicit inverse of selected basis matrix | Direct linear solve; singular/ill-conditioned outcomes recorded |
| Primal recovery | Pseudoinverse of normal equations | SVD least-squares solve without squaring condition number |
| Sparse projection | Entries `±1/sqrt(k),0` | Variance-normalized `±sqrt(3/k),0` |
| Orthogonal projection | Full Gaussian QR and row subset | Thin Gaussian QR, transpose, scale `sqrt(m/k)` |
| Feasibility reporting | Equality residual plus separate negativity ratio | Explicit combined original-feasibility check as well as both metrics |
| Infeasible `acc` | False-feasible count divided by all trials; errors counted as zero | Separate optimal, infeasible, unknown/error status counts |
| Timing | Elapsed `perf_counter` values called CPU time | Explicit wall-clock sampling, multiplication, solver, recovery components |

Changing the global nonzero normalization of a projector does not change an
equality LP's feasible set in exact arithmetic, but it matters when connecting
the projector to JL distance guarantees. Thin and full Gaussian QR generate
appropriate random subspaces, but do not produce bitwise-identical matrices.

The added mixed-sign family is scientifically separate from the legacy family:
its false-feasible rates must not be reported as a direct comparison against the
old notebook's all-negative family. Both deliberately expose simple Farkas
certificates in the original problem, so neither is a representative benchmark
of all infeasible LPs.

No earlier timing table is reused. The supplied new baseline records come from
executing the current scripts, and any future rerun should be saved separately.

## October 2026 extension

The September synthetic baseline and `experiments/rerun.py` remain unchanged.
`experiments/netlib_benchmark.py` adds five fixed, application-related Netlib
benchmarks, exact standard-form conversion, real CSR storage, CountSketch,
two row ratios, ten seeds per configuration, five invertible controls, and
saved vectors/recession-ray certificates. It reuses the existing quality and
recovery helpers; it does not implement a new recovery algorithm.

The data downloader pins the upstream revision and checks hashes. Raw MPS data
are fetched on demand, not republished. The independent scalar-metric verifier
and report-table generator check the released records; times are wall-clock
and array-buffer storage is not peak process memory.

[THESIS_COMPARISON.md](THESIS_COMPARISON.md) separates computational limitations
already reported in the thesis from these new tests. The report adds relevant
sketching background with citations, without claiming a new theoretical result.
