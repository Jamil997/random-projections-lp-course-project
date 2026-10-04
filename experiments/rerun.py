#!/usr/bin/env python3
"""Updated synthetic experiments for the Algorithms for Big Data course project.

Independent reimplementation of the author's master-thesis experiment design:
https://github.com/Jamil997/Random_Projections_master_thesis
legacy commit 3d2ff962612e1e5d76b07ccbade889576ba76416.
This is a fresh rerun with an explicitly revised protocol, not a reproduction of
the previous tables. It uses Bernoulli sparsity, fixed random streams, normalized
projectors and the same HiGHS dual-simplex solver throughout. The all-negative
infeasible family follows the legacy b distribution; the mixed-sign family is an
additional stress test. All timing runs are sequential with one library thread.
"""
from __future__ import annotations

import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
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
from scipy import linalg
from scipy.optimize import OptimizeWarning, linprog
from scipy.optimize._highspy import _core as highs_core
from threadpoolctl import threadpool_info, threadpool_limits

MASTER_SEED = 20260913
GRID = [(100, 150), (250, 375), (500, 750)]
DENSITIES = [0.1, 0.5]
REPETITIONS = 10
LEGACY_REPOSITORY = "https://github.com/Jamil997/Random_Projections_master_thesis"
LEGACY_COMMIT = "3d2ff962612e1e5d76b07ccbade889576ba76416"
FAMILIES = (
    (1, "feasible", "feasible"),
    (3, "infeasible", "legacy_all_negative"),
    (2, "infeasible", "mixed_sign_stress"),
)
TOL = 1e-7
LP_OPTIONS = {
    "presolve": True,
    "dual_feasibility_tolerance": TOL,
    "primal_feasibility_tolerance": TOL,
    "threads": 1,
    "parallel": False,
}
STATUS = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded", 4: "error"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def seed_for(scenario, m, n, density_index, repetition, stream, master_seed=MASTER_SEED):
    # Data and projector streams are separate; method comparisons share A,b,c.
    return int(np.random.SeedSequence(
        [master_seed, scenario, m, n, density_index, repetition, stream]
    ).generate_state(1, dtype=np.uint64)[0])


def generate_instance(family, m, n, density, instance_seed):
    """Regenerate an instance without storing large matrices or hidden state."""
    rng = np.random.default_rng(instance_seed)
    A = rng.uniform(0.0, 1.0, size=(m, n)) * (rng.random((m, n)) < density)
    c = np.ones(n)
    if family == "feasible":
        x0 = rng.uniform(0.0, 1.0, n)
        b = A @ x0
    elif family == "legacy_all_negative":
        b = -rng.uniform(0.0, 1.0, m)
        assert np.all(b < 0)
    elif family == "mixed_sign_stress":
        b = rng.uniform(0.5, 1.5, m)
        b[0] = -b[0]
    else:
        raise ValueError(family)
    if family != "feasible":
        # y=e_1 is an explicit Farkas certificate: y^T A >= 0, y^T b < 0.
        assert np.min(A[0]) >= 0 and b[0] < 0
    return A, b, c


def solve_lp(A, b, c):
    start = time.perf_counter()
    with warnings.catch_warnings():
        # SciPy forwards these supported HiGHS options to HiGHS verbatim.
        warnings.filterwarnings("ignore", category=OptimizeWarning,
                                message="Unrecognized options detected.*")
        result = linprog(c, A_eq=A, b_eq=b, bounds=(0, None),
                         method="highs-ds", options=LP_OPTIONS)
    return result, time.perf_counter() - start


def projector(kind, m, k, rng):
    if kind == "gaussian":
        return rng.standard_normal((k, m)) / np.sqrt(k)
    if kind == "sparse":
        u = rng.random((k, m))
        return np.where(u < 1 / 6, 1.0, np.where(u < 5 / 6, 0.0, -1.0)) * np.sqrt(3 / k)
    if kind == "orthogonal":
        q, _ = linalg.qr(rng.standard_normal((m, k)), mode="economic", check_finite=False)
        return np.sqrt(m / k) * q.T
    raise ValueError(kind)


def quality(x, A, b, c, reference_objective):
    residual = A @ x - b
    norm_b1 = float(np.linalg.norm(b, 1))
    norm_x1 = float(np.linalg.norm(x, 1))
    eq_inf = float(np.linalg.norm(residual, np.inf))
    min_x = float(np.min(x))
    objective = float(c @ x)
    return {
        "feas": float(np.linalg.norm(residual, 1) / norm_b1) if norm_b1 else None,
        "neg": float(np.sum(np.maximum(-x, 0)) / norm_x1) if norm_x1 else 0.0,
        "eq_residual_inf": eq_inf,
        "min_x": min_x,
        "objective": objective,
        "obj_gap": abs(objective - reference_objective) / abs(reference_objective)
        if reference_objective is not None and reference_objective != 0 else None,
        "accepted_feasible": int(eq_inf <= TOL * max(1.0, float(np.linalg.norm(b, np.inf)))
                                  and min_x >= -TOL),
    }


def recovery(x_projected, dual_projected, A, b, c, T, kind):
    start = time.perf_counter()
    m, n = A.shape
    rank = None
    warning_text = ""
    if kind == "dual":
        approximate_dual = T.T @ dual_projected
        norms = np.linalg.norm(A, axis=0)
        scores = np.divide(c - A.T @ approximate_dual, norms,
                           out=np.full(n, np.inf), where=norms > 0)
        support = np.argsort(scores, kind="stable")[:m]
        try:
            with warnings.catch_warnings(record=True) as caught:
                values = linalg.solve(A[:, support], b, assume_a="gen", check_finite=False)
            warning_text = "; ".join(str(w.message) for w in caught)
            status = "ill_conditioned" if caught else "computed"
        except linalg.LinAlgError as exc:
            return None, time.perf_counter() - start, "singular", None, str(exc)
    elif kind == "primal":
        support = np.argsort(-x_projected, kind="stable")[:m]
        # A_H^+ b = (A_H^T A_H)^+ A_H^T b in exact arithmetic. SVD avoids
        # squaring the condition number by explicitly forming normal equations.
        values, _, rank, _ = linalg.lstsq(A[:, support], b, lapack_driver="gelsd",
                                         check_finite=False)
        status = "computed" if rank == m else "rank_deficient"
    else:
        raise ValueError(kind)
    x = np.zeros(n)
    x[support] = values
    return x, time.perf_counter() - start, status, rank, warning_text


def summarize(rows):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in ("scenario", "family", "projector", "m", "n", "density", "k"))
        groups.setdefault(key, []).append(row)
    output = []
    excluded = {"repetition", "instance_seed", "projector_seed", "m", "n", "k", "density",
                "original_status_code", "projected_status_code"}
    for key, subset in groups.items():
        aggregate = dict(zip(("scenario", "family", "projector", "m", "n", "density", "k"), key))
        aggregate["n_repetitions"] = len(subset)
        for prefix in ("original", "projected"):
            for name in STATUS.values():
                aggregate[f"{prefix}_{name}_count"] = sum(r[f"{prefix}_status"] == name for r in subset)
        for prefix in ("dual", "primal"):
            for name in ("computed", "ill_conditioned", "singular", "rank_deficient"):
                aggregate[f"{prefix}_{name}_count"] = sum(r.get(f"{prefix}_recovery_status") == name for r in subset)
        numeric_keys = sorted(set().union(*(r.keys() for r in subset)) - excluded)
        for name in numeric_keys:
            values = [r[name] for r in subset if isinstance(r.get(name), (int, float))]
            if not values:
                continue
            aggregate[f"{name}_count"] = len(values)
            aggregate[f"{name}_mean"] = float(np.mean(values))
            aggregate[f"{name}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if name.endswith("accepted_feasible"):
                aggregate[f"{name}_total"] = int(sum(values))
        # Ratios of mean times, not means of paired time ratios.
        original_time = aggregate["original_solve_s_mean"]
        for label in ("projected_total_s", "dual_total_s", "primal_total_s"):
            if label + "_mean" in aggregate:
                aggregate[label.replace("_total_s", "_speedup")] = original_time / aggregate[label + "_mean"]
        output.append(aggregate)
    return output


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def deterministic_checks():
    # Same original bounded / projected unbounded counterexample as thesis.
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    c = np.array([0.0, -1.0])
    T = np.array([[1.0, 0.0]])
    orig, _ = solve_lp(A, b, c)
    proj, _ = solve_lp(T @ A, T @ b, c)
    assert orig.status == 0 and abs(orig.fun + 2) < 1e-9
    assert proj.status == 3
    counterexample = {
        "A": A.tolist(), "b": b.tolist(), "c": c.tolist(), "T": T.tolist(),
        "original_status": STATUS[orig.status], "original_objective": float(orig.fun),
        "projected_status": STATUS[proj.status],
    }
    A = np.array([[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    b = np.array([1.0, 1.0])
    c = np.array([-1.0, 0.0, 0.0])
    x0 = np.array([1.0, 0.0, 1.0])
    d = np.array([1.0, 1.0, 0.0])
    T = np.array([[1.0, 1.0]])
    assert np.all(d >= 0) and np.allclose(A @ x0, b) and np.allclose(A @ d, 0) and c @ d < 0
    orig, _ = solve_lp(A, b, c)
    proj, _ = solve_lp(T @ A, T @ b, c)
    assert orig.status == proj.status == 3
    preservation = {
        "A": A.tolist(), "b": b.tolist(), "c": c.tolist(), "T": T.tolist(),
        "x0": x0.tolist(), "d": d.tolist(), "original_status": STATUS[orig.status],
        "projected_status": STATUS[proj.status], "Ad_inf": float(np.max(np.abs(A @ d))),
        "TAd_inf": float(np.max(np.abs(T @ A @ d))), "c_dot_d": float(c @ d),
    }
    A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    b = np.array([2.0, 3.0])
    c = np.array([1.0, 1.0, 1.5])
    T = np.array([[1.0, 0.5], [-0.5, 1.0]])
    original, _ = solve_lp(A, b, c)
    projected, _ = solve_lp(T @ A, T @ b, c)
    assert original.status == projected.status == 0
    assert abs(original.fun - projected.fun) < 1e-9
    assert np.max(np.abs(A @ projected.x - b)) < 1e-9
    slack = c - (T @ A).T @ projected.eqlin.marginals
    assert np.min(slack) >= -TOL
    assert np.max(np.abs(projected.x * slack)) <= TOL
    full_dimension_and_kkt = {
        "A": A.tolist(), "b": b.tolist(), "c": c.tolist(), "T": T.tolist(),
        "original_objective": float(original.fun), "projected_objective": float(projected.fun),
        "original_equality_residual_inf": float(np.max(np.abs(A @ projected.x - b))),
        "dual_slack_min": float(np.min(slack)),
        "complementarity_inf": float(np.max(np.abs(projected.x * slack))),
        "dual_convention": "SciPy marginals y satisfy c-(TA)^T y >= 0; lift with T^T y.",
    }
    return {"counterexample": counterexample, "preservation_example": preservation,
            "full_dimension_and_kkt": full_dimension_and_kkt, "all_assertions_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "baseline")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args()
    if args.seed < 0 or args.repetitions < 1:
        parser.error("--seed must be nonnegative; --repetitions must be positive")
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error("output directory is not empty; choose another --output-dir to preserve previous results")
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "run.log").open("w", encoding="utf-8")

    def progress(message):
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    started = utc_now()
    run_start = time.perf_counter()
    environment = {
        "started_utc": started,
        "python": sys.version,
        "python_executable": Path(sys.executable).name,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "highs": highs_core._Highs().version(),
        "numpy_blas_name": np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {}).get("name", "unavailable"),
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "thread_environment": {key: os.environ[key] for key in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "master_seed": args.seed, "grid": GRID, "densities": DENSITIES,
        "repetitions": args.repetitions, "families": [family for _, _, family in FAMILIES],
        "family_seed_indices": {family: index for index, _, family in FAMILIES}, "k_rule": "floor(m/2)",
        "solver": "scipy.optimize.linprog(method='highs-ds')", "solver_options": LP_OPTIONS,
        "array_representation": "Dense float64 arrays for original and projected LPs; sparse projector also stored dense.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": "Fresh independent reimplementation and revised protocol based on the author's master-thesis experiments; previous tables not reused.",
        "legacy_repository": LEGACY_REPOSITORY, "legacy_commit": LEGACY_COMMIT,
        "matrix_generation": "A_ij=U_ij M_ij, independent U~Uniform[0,1], M~Bernoulli(density); dense float64 storage. Legacy notebooks used exact-count scipy.sparse.random instead.",
        "infeasible_families": {
            "legacy_all_negative": "b_i=-Uniform[0,1], matching the legacy b distribution.",
            "mixed_sign_stress": "Additional stress test: b_i~Uniform[0.5,1.5], then negate b_0.",
        },
        "timing_scope": "Wall-clock seconds; projection construction+matrix products+projected solve; recovery separately. No data generation, metrics, or diagnostic solves included.",
        "primary_status_policy": "Record every primary solver status without replacement by diagnostic retries.",

    }
    rows = []
    with threadpool_limits(limits=1):
        # Initialize BLAS and HiGHS before benchmark timing.
        solve_lp(np.eye(2), np.ones(2), np.ones(2))
        np.ones((20, 20)) @ np.ones((20, 20))
        linalg.lstsq(np.eye(2), np.ones(2), lapack_driver="gelsd")
        environment["threadpools"] = [
            {key: value for key, value in pool.items() if key != "filepath"}
            for pool in threadpool_info()
        ]
        checks = deterministic_checks()
        (out / "deterministic_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        progress("Deterministic unboundedness checks passed; start sequential benchmarks.")
        for scenario_index, scenario, family in FAMILIES:
            for m, n in GRID:
                k = m // 2
                for density_index, density in enumerate(DENSITIES):
                    for repetition in range(args.repetitions):
                        instance_seed = seed_for(scenario_index, m, n, density_index, repetition, 0, args.seed)
                        A, b, c = generate_instance(family, m, n, density, instance_seed)
                        original, original_time = solve_lp(A, b, c)
                        expected = 0 if scenario == "feasible" else 2
                        if original.status != expected:
                            raise RuntimeError(f"Generator/solver mismatch: {scenario} {m} {n} {density} {repetition}: {original.message}")
                        methods = ("gaussian",) if scenario == "feasible" else ("sparse", "orthogonal")
                        for method in methods:
                            stream = {"gaussian": 1, "sparse": 2, "orthogonal": 3}[method]
                            projection_seed = seed_for(scenario_index, m, n, density_index, repetition, stream, args.seed)
                            projector_rng = np.random.default_rng(projection_seed)
                            start = time.perf_counter()
                            T = projector(method, m, k, projector_rng)
                            sample_time = time.perf_counter() - start
                            start = time.perf_counter()
                            TA, Tb = T @ A, T @ b
                            multiply_time = time.perf_counter() - start
                            projected, projected_time = solve_lp(TA, Tb, c)
                            row = {
                                "scenario": scenario, "family": family, "projector": method, "m": m, "n": n,
                                "density": density, "actual_density": float(np.count_nonzero(A) / A.size),
                                "k": k, "repetition": repetition, "instance_seed": instance_seed,
                                "projector_seed": projection_seed,
                                "original_status_code": int(original.status), "original_status": STATUS.get(original.status, "other"),
                                "projected_status_code": int(projected.status), "projected_status": STATUS.get(projected.status, "other"),
                                "original_solve_s": original_time, "sample_s": sample_time,
                                "multiply_s": multiply_time, "projected_solve_s": projected_time,
                                "projected_total_s": sample_time + multiply_time + projected_time,
                                "projected_density": float(np.count_nonzero(TA) / TA.size),
                                "original_iterations": int(original.nit), "projected_iterations": int(projected.nit),
                                "projected_message": projected.message,
                            }
                            if scenario == "infeasible":
                                row.update(farkas_y_dot_b=float(b[0]), farkas_min_yA=float(np.min(A[0])))
                            if original.status == 0:
                                row["original_objective"] = float(original.fun)
                                row.update({"original_" + key: value for key, value in quality(original.x, A, b, c, original.fun).items()})
                            if projected.status == 0:
                                reference = float(original.fun) if original.status == 0 else None
                                row.update({"raw_" + key: value for key, value in quality(projected.x, A, b, c, reference).items()})
                                row["projected_eq_residual_inf"] = float(np.max(np.abs(TA @ projected.x - Tb)))
                                if scenario == "feasible":
                                    assert projected.fun <= original.fun + TOL * max(1.0, abs(original.fun))
                                    for recovery_kind in ("dual", "primal"):
                                        x_rec, recovery_time, recovery_status, rank, warning_text = recovery(
                                            projected.x, projected.eqlin.marginals, A, b, c, T, recovery_kind)
                                        row[f"{recovery_kind}_recovery_s"] = recovery_time
                                        row[f"{recovery_kind}_total_s"] = row["projected_total_s"] + recovery_time
                                        row[f"{recovery_kind}_recovery_status"] = recovery_status
                                        row[f"{recovery_kind}_rank"] = rank
                                        row[f"{recovery_kind}_warning"] = warning_text
                                        if x_rec is not None:
                                            row.update({recovery_kind + "_" + key: value for key, value in
                                                        quality(x_rec, A, b, c, reference).items()})
                            rows.append(row)
                    progress(f"{family}: m={m}, n={n}, density={density}, k={k}, {args.repetitions} instances complete.")
                    write_csv(out / "raw.csv", rows)
    aggregate = summarize(rows)
    write_csv(out / "summary.csv", aggregate)
    (out / "summary.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
    environment["finished_utc"] = utc_now()
    environment["wall_seconds"] = time.perf_counter() - run_start
    environment["projection_runs"] = len(rows)
    environment["independent_instances"] = len(FAMILIES) * len(GRID) * len(DENSITIES) * args.repetitions
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    progress(f"Complete: {len(rows)} projection runs; {environment['wall_seconds']:.2f} seconds.")
    log.close()


if __name__ == "__main__":
    main()
