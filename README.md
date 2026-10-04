# Random Projections for Linear Programming

Algorithms for Big Data course project by Jamil Zhumabek, MBZUAI.

**Question:** when does random constraint sketching save computation, and what
objective accuracy, sparsity, and original feasibility are sacrificed?

The current synthetic study replaces the earlier small synthetic baseline.
Netlib is retained unchanged as a limitation study. The separate report and
presentation have **not** been updated to these synthetic results.

## Main finding: faster relaxations, not faster exact LP solutions

For the original LP and its sketched relaxation,

```math
z^*=\min\{c^Tx:Ax=b,\ x\ge0\},\qquad
z_S=\min\{c^Tx:SAx=Sb,\ x\ge0\}.
```

Every original-feasible vector remains projected-feasible. In exact arithmetic,
a finite projected optimum satisfies `z_S <= z*`. The experiments show substantial
runtime reductions for computing these lower bounds on the tested synthetic
family, with a measurable trade-off between compression and bound quality.

**No candidate passed original feasibility: 0/116 raw, 0/116 dual recovery,
and 0/116 primal recovery.** Therefore these speedups are not speedups to valid
solutions of the original LP. Numerical checks use finite tolerances; they are
not exact-arithmetic certificates or a demonstrated downstream application benefit.

## Current synthetic study

The main run finished on 4 October 2026 UTC, with seed 2026100403:
**116 projections of 51 independently generated LPs**, and 61 reference solves
because ten LPs are solved in both dense and CSR form. All main LP solves
returned optimal status and passed numerical checks. There are five independent
repetitions per configuration, except three at 2,000 x 3,000.

The generator returns to the thesis's fixed-count sparsity construction.
Nonzero values and a feasible witness are uniform on [0,1], with `b=A x0`
and `c=ones(n)`. This positive-cost family guarantees bounded projected
objectives; it does not test the unboundedness observed on Netlib.

### 1. Scaling and matched thesis configurations

The scaling series holds n/m=1.5 and k/m approximately 0.327 fixed.
Times below are mean wall-clock seconds. Projected time includes sketch
sampling, matrix products, and the projected solve, but excludes recovery.
Speedup is the ratio of mean original and projected times.

| Density | Dimensions | k | Original (s) | Projected (s) | Speedup | Bound gap |
|---:|---|---:|---:|---:|---:|---:|
| 10% | 500 x 750 | 164 | 0.496 | 0.080 | 6.20x | 21.23% |
| 10% | 1,000 x 1,500 | 327 | 9.278 | 0.782 | 11.86x | 15.91% |
| 10% | 2,000 x 3,000 | 654 | 198.329 | 9.305 | 21.32x | 11.76% |
| 50% | 500 x 750 | 164 | 0.877 | 0.087 | 10.07x | 9.39% |
| 50% | 1,000 x 1,500 | 327 | 12.675 | 0.827 | 15.34x | 6.65% |
| 50% | 2,000 x 3,000 | 654 | 254.648 | 10.002 | 25.46x | 4.78% |

At the largest size, original-time sample SDs are 8.911 s and 18.168 s
for 10% and 50% density; projected-time SDs are 0.357 s and 0.502 s.
Full sample SDs, ranges, and individual observations are saved.

Additional thesis anchors are **1,000 x 1,200 with k=321** and **1,000 x 1,400
with k=327**, at both densities. At 50% density, their speedups are 12.32x
and 14.68x. They match historical size/generator/sketch settings, but use
controlled current solvers and stable current recovery routines, not identical
historical code or hardware. No universal crossover at 1,000 rows or asymptotic
rate is established: gains already occur at the smallest scaling size tested.

### 2. Compression versus bound quality

At 1,000 x 1,400, five Gaussian sketch sizes share the same original instances.
The independent sketches are not nested; monotonicity is an observed pattern
in these means, not a per-instance theorem.

| Rows retained | Speedup, 10% density | Gap, 10% density | Speedup, 50% density | Gap, 50% density |
|---:|---:|---:|---:|---:|
| 25% | 22.33x | 19.76% | 26.51x | 9.05% |
| 32.7% | 11.66x | 15.40% | 14.68x | 6.77% |
| 50% | 3.81x | 9.18% | 5.09x | 3.63% |
| 75% | 1.42x | 3.84% | 1.86x | 1.44% |
| 90% | 0.87x | 1.29% | 1.12x | 0.52% |

More aggressive compression is faster but gives a looser bound. At 10% density,
retaining 90% of rows is **slower** than the original solve.
Bound gap is `(z* - z_S)/max(1,abs(z*))`; all reference objectives exceed one.
It is not the objective error of a recovered original-feasible solution.

### 3. Sparsity and projector choice

Use CSR originals at 1,000 x 1,400 and nominal k=500. Both methods share
original instances and reference solves.

| Input density | Gaussian speedup | Gaussian gap | CountSketch speedup | CountSketch gap |
|---:|---:|---:|---:|---:|
| 1% | 0.51x | 22.39% | 6.22x | 26.46% |
| 10% | 3.75x | 9.18% | 9.64x | 10.92% |
| 50% | 5.01x | 3.63% | 7.97x | 4.72% |

Gaussian makes the product fully dense. At 1% density its matrix buffer grows
from 172,004 to 5,600,000 bytes; CountSketch averages 168,466 bytes.
These are **array-buffer sizes, not peak process RAM**.

CountSketch is faster here but has larger bound gaps and approximately
434-440 nonempty rows, versus 500 for Gaussian. Equal nominal k does not mean
equal effective numbers of nonzero equations. Ten paired dense/CSR Gaussian
controls at 10% and 50% density are also included.

### Recovery and scope limitations

The existing lifted-dual basis solve and primal-support SVD solve do not enforce
nonnegativity. At 2,000 x 3,000 and 50% density, projection plus these recoveries
averages 10.064 s and 10.456 s, but neither yields an accepted original-feasible
solution. Small equality residuals alone are insufficient.

This is an empirical extension on one synthetic family with one solver setup
and three to five repetitions, not a general guarantee, industrial-scale
demonstration, new recovery algorithm, or claim that every retrieval method fails.

## Netlib: retained application-benchmark limitation study

ADLITTLE, AFIRO, BLEND, SCFXM1, and STOCFOR1 are established historical LP
benchmarks, including refinery and forestry models, not newly collected data.
Their existing code, data provenance, results, saved vectors, and diagnostics
are retained unchanged in [results/netlib](results/netlib).

| Outcome | Recorded result |
|---|---|
| Design | 300 projections of five fixed LPs; three projectors, two row ratios, ten seeds |
| Projected statuses | 84 finite optima; 216 unbounded; no errors or limits |
| Unboundedness diagnostics | 216 saved, numerically verified recession rays |
| Raw original-feasible candidates | 0/84 |
| Dual recovery | All 84 selected bases singular |
| Primal recovery | All 84 systems rank deficient; 0/84 accepted |
| Separate invertible controls | 5/5 original-feasible and objective-matching |

The converted LPs have 27-330 constraints and 51-600 variables; original solves
take below approximately 5 ms. They demonstrate limitations from overhead,
densification, unbounded relaxations, and failed retrieval, not that sketching
is useless for every application LP. No artificial bounds suppress unboundedness.

## Reproduce and verify

Use Python 3.12 and the pinned dependencies. Verification of saved synthetic
evidence does not rerun the LP optimizations:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python experiments/verify_synthetic.py
```

A new run must use a fresh output directory:

```sh
.venv/bin/python experiments/synthetic_benchmark.py --output-dir results/rerun-synthetic-local
.venv/bin/python experiments/verify_synthetic.py --output-dir results/rerun-synthetic-local --write-summary
```

The recorded main run took about 34 minutes; the longest reference solve took
about 274 seconds, within the 300-second per-solve limit. Timings vary by
machine and load. Single-thread settings are requested, but the Apple Accelerate
backend exposed no pools to threadpoolctl, so thread usage was not independently
measured.

See [execution instructions](experiments/README.md) for a smoke test and Netlib
commands. Do not use the legacy report generators for this new synthetic schema:
the separate report and presentation are intentionally not updated here.

## Evidence and provenance

- [Frozen synthetic evidence](results/synthetic): raw records, protocol,
  summaries, 51 vector archives, environment, source provenance, verification,
  and checksums. Runtime pilots are separate and excluded.
- [Provenance](PROVENANCE.md) and [thesis comparison](THESIS_COMPARISON.md).
- [Superseded September baseline](https://github.com/Jamil997/random-projections-lp-course-project/tree/25c932497f4fcecaeebd7ca58be035ce2a03ef68/results/baseline):
  removed from the current results, recoverable through Git history.
- [Original thesis repository](https://github.com/Jamil997/Random_Projections_master_thesis):
  preserved independently and not modified.

## References

- Jamil Zhumabek. *Matrix Sketching and Linear Programming*. M.Sc. thesis,
  Nazarbayev University, 2025.
- Ky Vu, Pierre-Louis Poirion, and Leo Liberti. [Random Projections for Linear
  Programming](https://doi.org/10.1287/moor.2017.0894). *Mathematics of Operations
  Research* 43(4), 1051-1071, 2018.
- Sanjoy Dasgupta and Anupam Gupta. [An Elementary Proof of a Theorem of Johnson
  and Lindenstrauss](https://doi.org/10.1002/rsa.10073). *Random Structures &
  Algorithms* 22(1), 60-65, 2003.
