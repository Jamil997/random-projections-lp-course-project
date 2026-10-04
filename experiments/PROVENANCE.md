# Experiment provenance and protocol changes

These scripts independently reimplement the experimental design of Jamil
Zhumabek's master's thesis, *Matrix Sketching and Linear Programming*. The earlier
code is [Random_Projections_master_thesis](https://github.com/Jamil997/Random_Projections_master_thesis),
inspected at commit [`3d2ff962612e1e5d76b07ccbade889576ba76416`](https://github.com/Jamil997/Random_Projections_master_thesis/tree/3d2ff962612e1e5d76b07ccbade889576ba76416).
The earlier thesis tables and notebook outputs are not included in the new results.

## What is retained

- The standard-form LP: minimize `c @ x` subject to `A @ x = b`, `x >= 0`, with
  nonnegative sparse-in-value `A` and `c = ones(n)`.
- Feasible instances generated as `b = A @ x0`, with `x0` uniform on `[0,1]^n`.
- Gaussian projection and the two support-selection recovery heuristics for
  feasible instances.
- The original infeasible right-hand-side distribution `b = -Uniform[0,1]^m`,
  tested with sparse and orthogonal projectors as family `legacy_all_negative`.

## What changes in this course-project rerun

- A small grid `(m,n) = (100,150), (250,375), (500,750)`; densities `0.1,0.5`;
  `k = floor(m/2)`; ten independent originals per setting and family.
- Independent Bernoulli masks replace the exact-count nonzero selection used by
  `scipy.sparse.random` in the earlier notebooks. Actual density is recorded.
- An additional `mixed_sign_stress` family draws `b` on `[0.5,1.5]^m` and negates
  only its first entry. Its results must not be described as a rerun of the
  original all-negative family. Both are provably infeasible using `y = e_1`.
- Reproducible per-instance and per-projector random streams, derived from a
  recorded master seed. Both infeasible projectors share each original instance.
- The same SciPy HiGHS dual-simplex solver is used throughout. The earlier
  feasible notebook used SciPy for originals and CVXPY's automatic solver choice
  for projected problems, so its timing and dual conventions are not directly
  comparable. SciPy marginals satisfy `c - (TA).T @ y >= 0`; the correct lifted
  dual is `T.T @ y`. CVXPY equality multipliers for `TA @ x - Tb == 0` have the
  opposite sign under the corresponding Lagrangian convention.
- Sparse entries are `sqrt(3/k) * {+1,0,-1}` with probabilities `1/6,2/3,1/6`,
  giving variance `1/k`. Orthogonal rows use `sqrt(m/k) * Q.T`, where `Q` is the
  economic QR factor of an `m` by `k` Gaussian matrix. Their expected squared-norm
  scaling is normalized; neither preserves every vector's norm exactly.
- The dual heuristic solves its selected square system directly. The primal
  heuristic uses SVD least squares instead of explicitly forming normal
  equations. Equal support scores are ordered stably by column index. Neither
  heuristic enforces nonnegativity of the recovered vector.
- Equality residual and nonnegativity are checked separately; a candidate passes
  only when `||A x-b||_inf <= 1e-7 * max(1, ||b||_inf)` and `min(x) >= -1e-7`.
- Timings are sequential wall-clock times, with single-thread settings requested
  for HiGHS and numerical libraries. On Apple Accelerate, `threadpoolctl` may
  report no discoverable pools; this does not independently verify thread usage.
- Primary solver errors remain in all main counts. Alternative solves are
  recorded only in a separate, untimed diagnostic file.

All matrices and vectors are synthetic. Public runtime metadata excludes local
installation and home-directory paths. The baseline includes source checksums,
seeds, timestamps, raw rows, mean/sample-standard-deviation summaries, deterministic
unboundedness and dual-sign checks, and separate status diagnostics.

The benchmark script refuses a nonempty output directory. Use a fresh path for a
rerun, then pass that same `--output-dir` to both diagnostic and verification
scripts. Wall-clock timings can vary even when numerical outcomes reproduce.
