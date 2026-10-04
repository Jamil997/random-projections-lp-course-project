#!/usr/bin/env python3
"""Paired, seeded Netlib LP sketching experiments; synthetic baseline unchanged.

MPS input uses the HiGHS reader bundled with the pinned SciPy version. Only
continuous minimization models with nonnegative, unbounded-above variables
are accepted. Finite row bounds become equalities with slack/surplus columns.
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings
import numpy as np
import scipy
from scipy import sparse, linalg
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._highspy import _core
from threadpoolctl import threadpool_limits, threadpool_info
import rerun

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gaussian", "sparse", "countsketch")
RATIOS = (0.5, 0.75)
OPTIONS = {**rerun.LP_OPTIONS, "time_limit": 30}
TOL = 1e-7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_case(path):
    h = _core._Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(str(path))
    if status != _core.HighsStatus.kOk:
        raise ValueError(f"MPS reader failed: {Path(path).name}, {status}")
    lp = h.getLp()
    if lp.sense_ != _core.ObjSense.kMinimize or lp.offset_ != 0:
        raise ValueError("Only zero-offset minimization models supported")
    if lp.integrality_ or not np.all(np.asarray(lp.col_lower_) == 0):
        raise ValueError("Only continuous models with zero lower bounds supported")
    if not np.all(np.isposinf(lp.col_upper_)):
        raise ValueError("Finite upper bounds require an additional transformation")
    a = lp.a_matrix_
    if a.format_ != _core.MatrixFormat.kColwise:
        raise ValueError("Expected column-compressed MPS reader output")
    C = sparse.csc_matrix((a.value_, a.index_, a.start_),
                          shape=(lp.num_row_, lp.num_col_)).tocsr()
    return C, np.asarray(lp.row_lower_), np.asarray(lp.row_upper_), np.asarray(lp.col_cost_)


def standard_form(C, lower, upper, c):
    """Exact <= / >= / equality conversion, including two-sided row ranges."""
    row_indices, rhs, slack_rows, slack_signs = [], [], [], []
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if np.isfinite(lo) and lo == hi:
            row_indices.append(i)
            rhs.append(lo)
        else:
            if np.isfinite(hi):
                slack_rows.append(len(rhs))
                slack_signs.append(1.0)
                row_indices.append(i)
                rhs.append(hi)
            if np.isfinite(lo):
                slack_rows.append(len(rhs))
                slack_signs.append(-1.0)
                row_indices.append(i)
                rhs.append(lo)
    m, q = len(rhs), len(slack_rows)
    slacks = sparse.coo_matrix((slack_signs, (slack_rows, np.arange(q))), shape=(m, q))
    A = sparse.hstack([C[row_indices], slacks], format="csr")
    A.eliminate_zeros()
    return A, np.asarray(rhs), np.r_[c, np.zeros(q)]


def solve(A, b, c, *, method="highs-ds"):
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning,
                                message="Unrecognized options detected.*")
        result = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method=method, options=OPTIONS)
    return result, time.perf_counter() - start


def native_solve(C, lower, upper, c):
    eq = np.isfinite(lower) & (lower == upper)
    up = np.isfinite(upper) & ~eq
    low = np.isfinite(lower) & ~eq
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning,
                                message="Unrecognized options detected.*")
        return linprog(c, A_eq=C[eq], b_eq=upper[eq],
                       A_ub=sparse.vstack([C[up], -C[low]], format="csr"),
                       b_ub=np.r_[upper[up], -lower[low]], bounds=(0, None),
                       method="highs-ds", options=OPTIONS)


def projector(kind, m, k, seed):
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.standard_normal((k, m)) / np.sqrt(k)
    if kind == "sparse":
        u = rng.random((k, m))
        return sparse.csr_matrix(np.where(u < 1/6, 1.0,
                                np.where(u < 5/6, 0.0, -1.0)) * np.sqrt(3/k))
    if kind == "countsketch":
        buckets = rng.integers(k, size=m)
        signs = rng.choice([-1.0, 1.0], size=m)
        return sparse.coo_matrix((signs, (buckets, np.arange(m))), shape=(k, m)).tocsr()
    raise ValueError(kind)


def array_bytes(A):
    if sparse.issparse(A):
        return int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)
    return int(A.nbytes)


def density(A):
    nnz = A.nnz if sparse.issparse(A) else np.count_nonzero(A)
    return float(nnz / np.prod(A.shape))


def native_quality(x, C, lower, upper):
    activity = C @ x
    lo = np.isfinite(lower)
    hi = np.isfinite(upper)
    violation = max(0.0, float(np.max(lower[lo] - activity[lo], initial=0)),
                    float(np.max(activity[hi] - upper[hi], initial=0)))
    bound_scale = max(1.0, float(np.max(np.abs(lower[lo]), initial=0)),
                      float(np.max(np.abs(upper[hi]), initial=0)))
    return {"native_row_violation": violation,
            "native_accepted": int(violation <= TOL * bound_scale and np.min(x) >= -TOL)}


def seed_for(seed, dataset_index, ratio_index, method_index, repetition):
    return int(np.random.SeedSequence(
        [seed, dataset_index, ratio_index, method_index, repetition]
    ).generate_state(1, dtype=np.uint64)[0])


def summarize(rows):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("dataset", "ratio", "projector"))
        groups.setdefault(key, []).append(row)
    output = []
    metrics = ("original_solve_s", "sample_s", "multiply_s", "projected_solve_s",
               "projected_total_s", "projected_density", "projected_matrix_bytes",
               "projector_bytes", "raw_feas", "raw_obj_gap", "dual_feas",
               "dual_neg", "dual_obj_gap", "primal_feas", "primal_neg", "primal_obj_gap",
               "dual_total_s", "primal_total_s")
    for (dataset, ratio, method), group in groups.items():
        summary = {"dataset": dataset, "ratio": ratio, "projector": method,
                   "m": group[0]["m"], "n": group[0]["n"], "k": group[0]["k"],
                   "runs": len(group)}
        for status in rerun.STATUS.values():
            summary[f"{status}_count"] = sum(r["projected_status"] == status for r in group)
        for prefix in ("raw", "dual", "primal"):
            summary[f"{prefix}_available"] = sum(f"{prefix}_accepted_feasible" in r for r in group)
            summary[f"{prefix}_accepted"] = sum(r.get(f"{prefix}_accepted_feasible", 0) for r in group)
        summary["ray_verified"] = sum(r.get("ray_verified", 0) for r in group)
        for metric in metrics:
            vals = [r[metric] for r in group if r.get(metric) is not None]
            summary[metric + "_count"] = len(vals)
            summary[metric + "_mean"] = float(np.mean(vals)) if vals else None
            summary[metric + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if vals else None)
        output.append(summary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/netlib")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/netlib")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.seed < 0 or args.repetitions < 1:
        parser.error("nonnegative seed and positive repetitions required")
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error("output directory is nonempty; choose a fresh directory")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    for item in manifest["datasets"]:
        if sha(args.data_dir / (item["name"] + ".mps")) != item["sha256_mps"]:
            raise ValueError(f"Dataset checksum mismatch: {item['name']}")
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()
    rows, cases, controls, vectors, diagnostics = [], [], [], {}, []
    with threadpool_limits(limits=1):
        solve(sparse.eye(2, format="csr"), np.ones(2), np.ones(2))
        pools = [{k: v for k, v in p.items() if k != "filepath"} for p in threadpool_info()]
        for di, entry in enumerate(manifest["datasets"]):
            name = entry["name"]
            C, lower, upper, native_c = read_case(args.data_dir / f"{name}.mps")
            A, b, c = standard_form(C, lower, upper, native_c)
            dense_A = A.toarray()  # Used only by the thesis's dense recovery heuristics.
            m, n = A.shape
            native = native_solve(C, lower, upper, native_c)
            original, _ = solve(A, b, c)
            if native.status != 0 or original.status != 0:
                raise ValueError(f"Original not solved to optimum: {name}")
            if not np.isclose(original.fun, entry["reference_objective"], rtol=1e-6, atol=1e-6):
                raise ValueError(f"Published objective mismatch: {name}")
            if not np.isclose(native.fun, original.fun, rtol=1e-8, atol=1e-7):
                raise ValueError(f"Standard-form conversion changed objective: {name}")
            assert rerun.quality(original.x, A, b, c, original.fun)["accepted_feasible"]
            vectors[f"{name}_original"] = original.x
            cases.append({"dataset": name, "native_rows": C.shape[0], "native_columns": C.shape[1],
                          "m": m, "n": n, "nnz": A.nnz, "density": density(A),
                          "csr_bytes": array_bytes(A), "dense_bytes": dense_A.nbytes,
                          "reference_objective": entry["reference_objective"],
                          "objective": float(original.fun), "native_objective": float(native.fun),
                          "sha256_mps": entry["sha256_mps"], "description": entry["description"]})
            # No-compression orthogonal control: exact invertibility retains all equalities.
            Q, _ = linalg.qr(np.random.default_rng(args.seed + di).standard_normal((m, m)))
            control, _ = solve(Q @ A, Q @ b, c)
            assert control.status == 0
            cq = rerun.quality(control.x, A, b, c, original.fun)
            assert cq["accepted_feasible"] and cq["obj_gap"] < 1e-6
            vectors[f"{name}_control"] = control.x
            controls.append({"dataset": name, "k": m, "status": "optimal", **cq})
            for ri, ratio in enumerate(RATIOS):
                k = max(1, int(np.floor(m * ratio)))
                for repetition in range(args.repetitions):
                    reference, t0 = solve(A, b, c)
                    assert reference.status == 0
                    # Rotate method order deterministically to avoid a fixed method always going first.
                    for mi in np.roll(np.arange(len(METHODS)), repetition % len(METHODS)):
                        kind = METHODS[int(mi)]
                        seed = seed_for(args.seed, di, ri, int(mi), repetition)
                        t = time.perf_counter()
                        S = projector(kind, m, k, seed)
                        ts = time.perf_counter() - t
                        t = time.perf_counter()
                        SA, Sb = S @ A, S @ b
                        if sparse.issparse(SA):
                            SA = SA.tocsr()
                            SA.eliminate_zeros()
                        tm = time.perf_counter() - t
                        projected, tp = solve(SA, Sb, c)
                        row_id = f"{name}_{ri}_{kind}_{repetition}"
                        row = {"id": row_id, "dataset": name, "ratio": ratio,
                               "projector": kind, "repetition": repetition, "projector_seed": seed,
                               "m": m, "n": n, "k": k, "original_objective": float(reference.fun),
                               "original_solve_s": t0, "sample_s": ts, "multiply_s": tm,
                               "projected_solve_s": tp, "projected_total_s": ts + tm + tp,
                               "projected_status": rerun.STATUS.get(projected.status, "error"),
                               "projected_message": projected.message,
                               "projected_density": density(SA), "projected_matrix_bytes": array_bytes(SA),
                               "projector_bytes": array_bytes(S), "original_matrix_bytes": array_bytes(A)}
                        if projected.status == 0:
                            assert projected.fun <= reference.fun + 1e-6 * max(1, abs(reference.fun))
                            vectors[row_id + "_raw"] = projected.x
                            row.update({"raw_" + key: value for key, value in
                                        rerun.quality(projected.x, A, b, c, reference.fun).items()})
                            row.update(native_quality(projected.x[:C.shape[1]], C, lower, upper))
                            for method in ("dual", "primal"):
                                x, tr, status, rank, warning = rerun.recovery(
                                    projected.x, projected.eqlin.marginals, dense_A, b, c, S, method)
                                row.update({method + "_recovery_status": status,
                                            method + "_recovery_s": tr, method + "_total_s": ts+tm+tp+tr,
                                            method + "_warning": warning})
                                if x is not None and np.all(np.isfinite(x)):
                                    vectors[row_id + "_" + method] = x
                                    row.update({method + "_" + key: value for key, value in
                                                rerun.quality(x, A, b, c, reference.fun).items()})
                        elif projected.status == 3:
                            # Untimed certificate: find a nonnegative recession direction with c^T d=-1.
                            ray_lp, _ = solve(sparse.vstack([sparse.csr_matrix(SA), c[None, :]], format="csr"),
                                              np.r_[np.zeros(k), -1.0], np.zeros(n))
                            if ray_lp.status == 0:
                                d = ray_lp.x
                                err = float(np.linalg.norm(SA @ d, np.inf))
                                norm = float(c @ d)
                                ok = err <= 1e-6 and np.min(d) >= -TOL and abs(norm + 1) <= 1e-6
                                row.update(ray_verified=int(ok), ray_residual=err, ray_objective=norm,
                                           original_ray_residual=float(np.linalg.norm(A @ d, np.inf)))
                                vectors[row_id + "_ray"] = d
                            else:
                                row["ray_verified"] = 0
                        else:
                            alternate, _ = solve(SA, Sb, c, method="highs-ipm")
                            diagnostics.append({"id": row_id, "primary": row["projected_status"],
                                                "alternate": rerun.STATUS.get(alternate.status, "error"),
                                                "message": alternate.message})
                        rows.append(row)
                print(f"{name} k={k}/{m}: " + str(dict(Counter(
                    r["projected_status"] for r in rows if r["dataset"] == name and r["ratio"] == ratio))), flush=True)
    np.savez_compressed(out / "vectors.npz", **vectors)
    rerun.write_csv(out / "raw.csv", rows)
    summaries = summarize(rows)
    rerun.write_csv(out / "summary.csv", summaries)
    for filename, value in (("raw.json", rows), ("summary.json", summaries),
                            ("datasets.json", cases), ("controls.json", controls),
                            ("diagnostics.json", diagnostics)):
        (out / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    environment = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "wall_seconds": time.perf_counter() - run_start,
                   "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                   "highs": _core._Highs().version(), "platform": platform.platform(),
                   "seed": args.seed, "repetitions": args.repetitions, "ratios": RATIOS,
                   "methods": METHODS, "runs": len(rows), "distinct_originals": len(cases),
                   "solver": "scipy.optimize.linprog highs-ds", "options": OPTIONS,
                   "threads": "one requested; reported pools below may omit Apple Accelerate",
                   "threadpools": pools, "dataset_manifest_sha256": sha(args.data_dir / "manifest.json"),
                   "source_hashes": {p: sha(Path(__file__).parent / p)
                                     for p in ("netlib_benchmark.py", "rerun.py", "fetch_netlib.py")},
                   "artifact_hashes": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
                   "timing_scope": "sampling + matrix multiplication/format conversion + projected solver; recovery separately; MPS conversion, quality, ray certificates and retries excluded",
                   "storage_scope": "CSR original and sparse sketches; dense Gaussian. Reported bytes count array buffers only, not peak process memory. Dense recovery copy excluded from projection storage comparison."}
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(f"Complete: {len(rows)} projections, {len(cases)} fixed LPs, {len(controls)} controls; "
          f"{environment['wall_seconds']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
