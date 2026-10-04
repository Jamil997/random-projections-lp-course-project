#!/usr/bin/env python3
"""Check benchmark provenance, paired instances, and independently recompute summaries."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "baseline")
    args = parser.parse_args()
    out = args.output_dir
    env = json.loads((out / "environment.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    checks = json.loads((out / "deterministic_checks.json").read_text())
    diagnostics = json.loads((out / "status_diagnostics.json").read_text())
    with (out / "raw.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert env["script_sha256"] == hashlib.sha256(Path(__file__).with_name("rerun.py").read_bytes()).hexdigest()
    assert diagnostics["script_sha256"] == hashlib.sha256(Path(__file__).with_name("diagnose_statuses.py").read_bytes()).hexdigest()
    assert diagnostics["raw_csv_sha256"] == hashlib.sha256((out / "raw.csv").read_bytes()).hexdigest()
    assert checks["all_assertions_passed"] is True
    # Public metadata must not expose local accounts or build/installation paths.
    serialized = json.dumps(env)
    assert "/Users/" not in serialized and "/home/" not in serialized and "filepath" not in serialized
    groups = defaultdict(list)
    instances = defaultdict(list)
    group_fields = ("scenario", "family", "projector", "m", "n", "density", "k")
    for row in rows:
        key = tuple(row[name] for name in group_fields)
        groups[key].append(row)
        instances[(row["family"], row["m"], row["n"], row["density"], row["repetition"])].append(row)
        assert int(row["k"]) == int(row["m"]) // 2
        expected = "optimal" if row["family"] == "feasible" else "infeasible"
        assert row["original_status"] == expected
        if row["family"] != "feasible":
            assert float(row["farkas_y_dot_b"]) < 0 <= float(row["farkas_min_yA"])
        assert 0 <= float(row["actual_density"]) <= 1
        close(float(row["projected_total_s"]), sum(float(row[name]) for name in
              ("sample_s", "multiply_s", "projected_solve_s")))
    per_family = len(env["grid"]) * len(env["densities"]) * env["repetitions"]
    assert len(rows) == env["projection_runs"] == 5 * per_family
    assert len(instances) == env["independent_instances"] == 3 * per_family
    assert len(summary) == len(groups) == 5 * len(env["grid"]) * len(env["densities"])
    for key, instance in instances.items():
        methods = {row["projector"] for row in instance}
        assert methods == ({"gaussian"} if key[0] == "feasible" else {"sparse", "orthogonal"})
        assert len(instance) == len(methods)
        for name in ("instance_seed", "original_solve_s", "original_status", "actual_density"):
            assert len({row[name] for row in instance}) == 1
    numeric_fields_verified = 0
    for aggregate in summary:
        key = tuple(str(aggregate[name]) for name in group_fields)
        subset = groups[key]
        assert len(subset) == aggregate["n_repetitions"] == env["repetitions"]
        assert {int(row["repetition"]) for row in subset} == set(range(env["repetitions"]))
        for prefix in ("original", "projected"):
            counts = Counter(row[prefix + "_status"] for row in subset)
            for status in ("optimal", "limit", "infeasible", "unbounded", "error"):
                assert aggregate[f"{prefix}_{status}_count"] == counts[status]
        for name in aggregate:
            if not name.endswith("_mean"):
                continue
            metric = name[:-5]
            values = [float(row[metric]) for row in subset if row.get(metric, "") != ""]
            assert len(values) == aggregate[metric + "_count"]
            close(aggregate[name], fmean(values))
            close(aggregate[metric + "_sd"], stdev(values) if len(values) > 1 else 0)
            if metric.endswith("accepted_feasible"):
                assert aggregate[metric + "_total"] == sum(values)
            numeric_fields_verified += 1
        for label in ("projected", "dual", "primal"):
            if label + "_speedup" in aggregate:
                close(aggregate[label + "_speedup"], aggregate["original_solve_s_mean"] /
                      aggregate[label + "_total_s_mean"])
    unresolved = [row for row in rows if row["projected_status"] in ("error", "limit", "other")]
    assert len(unresolved) == diagnostics["number_of_cases"] == len(diagnostics["cases"])
    expected_cases = {(row["family"], row["projector"], row["instance_seed"]) for row in unresolved}
    actual_cases = {(case["family"], case["projector"], case["instance_seed"])
                    for case in diagnostics["cases"]}
    assert actual_cases == expected_cases
    print(f"PASS: {len(rows)} projections, {len(instances)} independent originals, {len(groups)} groups; "
          f"{numeric_fields_verified} numeric summary fields independently recomputed; "
          f"{len(unresolved)} unresolved primary outcomes retained with diagnostics.")
    for family in env["families"]:
        family_rows = [row for row in rows if row["family"] == family]
        print(f"{family}: " + str(dict(Counter(row["projected_status"] for row in family_rows))))


if __name__ == "__main__":
    main()
