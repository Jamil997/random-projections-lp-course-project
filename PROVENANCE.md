# Relationship to earlier work

This project builds on Jamil Zhumabek's M.Sc. thesis, *Matrix Sketching and
Linear Programming*, Nazarbayev University, 2025, and its
[original repository](https://github.com/Jamil997/Random_Projections_master_thesis).
The inspected snapshot is
[3d2ff962612e1e5d76b07ccbade889576ba76416](https://github.com/Jamil997/Random_Projections_master_thesis/tree/3d2ff962612e1e5d76b07ccbade889576ba76416).
Its 14 November 2025 repository date is not the thesis submission date.
The original repository and notebook outputs are unchanged.

## Retained theory and changed experiment

The projected equality relaxation, JL background, unboundedness implication,
and counterexample are inherited material, not newly claimed literature-level
theory. The current extension is empirical, not a new recovery algorithm.

| Aspect | Earlier work | Current synthetic study |
|---|---|---|
| Generator | Thesis fixed-count sparse randomness; September draft Bernoulli masks | Returns to fixed-count sparsity and uniform nonzero values |
| Feasible family | b=A x0; uniform nonnegative witness; c=ones(n) | Retained with explicit independent seeds |
| Sizes | Thesis includes 1000 x 1200/1400; draft stops at 500 x 750 | Matched anchors plus fixed-aspect scaling through 2000 x 3000 |
| Compression | Selected historical k; draft fixes k=m/2 | Five ratios at 1000 x 1400, including historical k=327 |
| Storage | Earlier synthetic arrays dense | Separate dense scaling and true CSR sparsity comparison |
| Projectors | Gaussian for feasible synthetic LPs | Gaussian and CountSketch; nonempty sketch rows recorded |
| Solver | Original thesis uses SciPy versus CVXPY automatic selection | Same HiGHS dual simplex and tolerances for both LPs |
| Recovery | Explicit inverse / normal equations in the notebook | Existing checked marginal sign, direct solve / SVD, stable ties |
| Evidence | Saved notebook outputs / small draft summaries | Raw records, candidate vectors/duals, scalar recomputation, hashes |

The main run finished on 4 October 2026 UTC: 116 projections of 51 independent
LPs and 61 reference solves (ten dense/CSR pairs). It is a matched-design
extension, not a bitwise replication of unseeded historical matrices or old
hardware. Two thesis-anchor densities, 0.1 and 0.5, are repeated.

Only the feasible positive-cost family is included in the new synthetic study.
The earlier all-negative and mixed-sign infeasible families are historical,
not part of its 116 projections. The former 300-projection course baseline is
[recoverable in Git history](https://github.com/Jamil997/random-projections-lp-course-project/tree/25c932497f4fcecaeebd7ca58be035ce2a03ef68/results/baseline).
The unchanged rerun.py remains an imported helper for current synthetic and
Netlib code; it is not the current synthetic entry point.

## Frozen measurements and portable export

The run was executed by an isolated temporary runner before publication.
Raw observations, measured times, protocol, vectors, and the numerical summary
are copied unchanged. Absolute helper-path keys in the environment are
normalized, with a publication note and seed phase added. The original recorded
runner hash is retained.

synthetic_benchmark.py is a portable CLI export. Before export, ASTs of the
eight numerical functions (seed, generator, solver, storage, quality, recovery,
record writer, case runner) were checked equal to the measured implementation.
[source_provenance.json](results/synthetic/source_provenance.json) records
both source identities and kernel hashes. These historical timings are not
represented as a newly executed run of the portable CLI.

The separate runtime pilot had three originals, two projections, and one
120-second original timeout. It is excluded from main estimates. The main
300-second limit was chosen before observing main-stage outcomes; all main
solves completed optimally. pilot_raw.jsonl preserves the calibration evidence.
Private filesystem inventories and local report hashes are not published.

## Netlib and document boundaries

Netlib remains unchanged: five fixed benchmarks, 300 projections, 216 numerical
unbounded-ray certificates, 84 finite optima without successful retrieval, and
five successful full-dimensional controls. Its code, data provenance, results,
vectors, and diagnostics are preserved byte-for-byte. Raw MPS inputs are fetched
from a pinned mirror and not redistributed.

The report and presentation are neither updated nor uploaded in this change.
README/provenance documentation describes the new findings without implying
that the separate documents already contain them.
