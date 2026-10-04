# Thesis findings and course-project extensions

Source: Jamil Zhumabek, *Matrix Sketching and Linear Programming*, M.Sc. thesis,
Nazarbayev University, 2025. Page numbers below are the printed thesis numbers
(also the PDF page numbers). The thesis PDF is cited but not redistributed.
The previous [code repository](https://github.com/Jamil997/Random_Projections_master_thesis)
was rechecked on 4 October 2026 at unchanged commit
`3d2ff962612e1e5d76b07ccbade889576ba76416`.

## Did the thesis already report computational limitations?

**Yes.** The extension should not claim to discover these limitations anew.

| Original location | What was already discussed or reported |
|---|---|
| Abstract, p. 2 | Hidden computational costs and trade-offs of dimensionality reduction |
| pp. 31–35 | Literature discussion of repeated interior-point linear-system time/memory costs and randomized preconditioning |
| p. 38, Projection Overhead | Sampling/projection and dense QR can cost more than the reduction saves |
| p. 39, Solver Efficiency | Original HiGHS/presolve can quickly detect simple contradictions; projected matrices may densify |
| pp. 38–40, tables/discussion | Original solution is generally faster on the reported infeasible configurations; this is not an exception-free claim |
| Table 4.1, p. 38 | Nonnegativity violation remains substantial despite equality residuals rounding to zero |

We did **not** find an explicit out-of-memory event, measured peak RAM limit,
or statement that a particular hardware capacity determined the maximum
experiment size. Algorithmic cost/densification discussion is not the same as
an observed hardware-capacity limit. Neither the current synthetic scaling
study nor the small Netlib suite establishes a large-scale memory limitation;
array-buffer storage is not peak process RAM.

The prose on p. 39 describes projections becoming feasible, but Tables 4.2–4.3
have zero reported misclassification rates and p. 40 says no misclassification
occurred. The inspected notebook's `acc` measures originally infeasible cases
reported optimal after projection; errors are not separated in that statistic.
Therefore those tables do not substantiate the stronger prose. The superseded
September mixed-sign stress family was a separate experiment, not an explanation retroactively
attributed to the original all-negative runs.

## What is actually new in this course project?

| Aspect | Earlier work | Course-project extension |
|---|---|---|
| Theory | JL, LP relaxation, unboundedness implication and counterexample | Retained with explicit assumptions; no novelty claim |
| Sketching context | Broad survey in thesis | Focused row sampling, CountSketch, low-rank/streaming background, and distinction between constraint relaxation and IPM preconditioning |
| Equality preservation | Projection motivation | Explicit residual-subspace argument: a full-row-rank LP cannot have all residuals embedded injectively into fewer rows |
| Experimental reproducibility | Notebook outputs, no recorded seed/environment | Pinned versions, independent seeds, raw results, source hashes, deterministic checks |
| Synthetic evaluation | Original random generators and recovery heuristics | Replaces the small draft with 116 projections of 51 LPs: matched thesis anchors, scaling through 2000 x 3000, five compression ratios, CSR sparsity comparisons |
| Application benchmarks | Synthetic experiments | Five Netlib LPs, including refinery/forestry models; optima cross-checked before projection |
| Storage | Sparse-valued matrices converted to dense arrays | CSR originals/sparse products; actual array-buffer sizes, not claimed peak memory |
| Projectors | Gaussian, sparse-valued, orthogonal | Adds CountSketch on synthetic CSR instances and Netlib; records nominal and nonempty sketch rows |
| Unboundedness | Theory and counterexample | 216 stored numerical ray certificates for bounded-original/unbounded-projection outcomes |
| Controls | Limited deterministic examples | Five invertible full-dimensional controls recover each benchmark optimum |
| Retrieval reporting | Equality residual and negativity separately | Raw/native feasibility, singular dual recovery, rank-deficient primal recovery, explicit denominators |

The Netlib run contains 84 finite projected optima and 216 unbounded projections.
None of the 84 raw vectors passes original feasibility. All 84 dual-recovery
bases are singular; all 84 primal-recovery systems are rank deficient and their
candidates fail feasibility. All five full-dimensional controls pass.
These are informative negative results, not an improved recovery algorithm or
a claim that every sketching method fails. Later retrieval methods and
randomized IPM preconditioners are relevant related work, not evaluated methods.

## Relationship between datasets and repetitions

The current synthetic run contains 116 projections of 51 independently generated
LPs and 61 reference solves: ten instances are solved in both dense and CSR form.
It uses five repetitions per configuration, except three at 2000 x 3000.
The earlier September 300-projection synthetic study is removed from current
results but preserved in Git history.

Netlib still contains 300 projections of **five fixed LPs**, plus five separate
controls; its data and code are unchanged. Repeated seeds are not independent
application datasets. The original thesis repository is also unchanged.

## What the replacement establishes

At fixed aspect/compression ratios, observed synthetic projection speedups grow
with size; at 2000 x 3000 they reach 21.32x and 25.46x for densities 0.1 and 0.5.
Changing compression exposes the speed-versus-bound-quality trade-off.
Gaussian densification can reverse the timing advantage on very sparse inputs;
CountSketch avoids much of that storage growth but has looser objective bounds.

There are still 0/116 accepted original-feasible candidates for each of raw,
dual, and primal retrieval. These are faster relaxation/lower-bound computations,
not faster exact original-LP solutions. Findings are conditional on the tested
generator, solver, tolerances, and small repetition counts.
The separate report and presentation have not been updated to this replacement.
