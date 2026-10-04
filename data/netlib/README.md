# Netlib data provenance

Five fixed historical benchmarks are fetched by `experiments/fetch_netlib.py`:
ADLITTLE, AFIRO, BLEND, SCFXM1, STOCFOR1. Descriptions and reference optima come
from the [Netlib LP index](https://www.netlib.org/lp/data/readme). BLEND is an
oil-refinery model variant; STOCFOR1 is the deterministic member of a
seven-period forestry family. These are benchmarks, not contemporary field data.

The MPS mirror is [COIN-OR Data-Netlib](https://github.com/coin-or-tools/Data-Netlib)
at revision `f1cc423067407d55d579c9c35fb01edf860dbc24`.
`manifest.json` records exact URLs, compressed and uncompressed SHA-256 hashes,
reference objectives, and download time. Repeat downloads must match it.

Raw `.mps` and `.mps.gz` files are ignored by Git and are **not redistributed**
in this repository. Run the downloader before benchmarking or verifying vectors.
The upstream `LICENSE` is included verbatim as `UPSTREAM_LICENSE.txt` for
transparency: it explicitly states that its terms concern the build system,
not the `.mps.gz` benchmark files. We do not assign a new license to those data.

The experiment accepts only continuous minimization MPS models with zero
objective offset and variable bounds `[0,+infinity)`. Finite row inequalities
become equalities with slack/surplus columns; two-sided ranges are represented
by two inequalities. Native and standard-form optima are cross-checked against
the published reference before projection. No arbitrary upper bounds are added.
