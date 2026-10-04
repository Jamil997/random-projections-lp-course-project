#!/usr/bin/env python3
"""Untimed follow-up solves for unresolved primary statuses, never replacements.

Independent reimplementation accompanying rerun.py; prior work:
https://github.com/Jamil997/Random_Projections_master_thesis
commit 3d2ff962612e1e5d76b07ccbade889576ba76416.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import rerun
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "baseline")
    args = parser.parse_args()
    result_dir = args.output_dir
    with (result_dir / "raw.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    cases = []
    with threadpool_limits(limits=1):
        for row in rows:
            if row["projected_status"] not in ("error", "limit", "other"):
                continue
            m, n, k = (int(row[name]) for name in ("m", "n", "k"))
            A, b, c = rerun.generate_instance(row["family"], m, n, float(row["density"]),
                                            int(row["instance_seed"]))
            T = rerun.projector(row["projector"], m, k,
                               rerun.np.random.default_rng(int(row["projector_seed"])))
            TA, Tb = T @ A, T @ b
            case = {name: row[name] for name in (
                "scenario", "family", "projector", "m", "n", "density", "k", "repetition",
                "instance_seed", "projector_seed", "projected_status", "projected_message")}
            case["diagnostic_solves"] = []
            for method, presolve in (("highs-ds", False), ("highs-ipm", True)):
                options = dict(rerun.LP_OPTIONS, presolve=presolve, time_limit=30)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=rerun.OptimizeWarning,
                                            message="Unrecognized options detected.*")
                    result = rerun.linprog(c, A_eq=TA, b_eq=Tb, bounds=(0, None),
                                          method=method, options=options)
                diagnostic = {
                    "method": method, "options": options, "status_code": int(result.status),
                    "status": rerun.STATUS.get(result.status, "other"), "message": result.message,
                    "iterations": int(result.nit),
                }
                if result.status == 0:
                    diagnostic["original_constraint_quality"] = rerun.quality(result.x, A, b, c, None)
                    diagnostic["projected_equality_residual_inf"] = float(
                        rerun.np.max(rerun.np.abs(TA @ result.x - Tb)))
                case["diagnostic_solves"].append(diagnostic)
            cases.append(case)
            print(f"{row['family']}/{row['projector']} m={m} density={row['density']} "
                  f"repetition={row['repetition']}: " + ", ".join(
                      d["method"] + "/presolve=" + str(d["options"]["presolve"]) + ":" + d["status"]
                      for d in case["diagnostic_solves"]), flush=True)
    output = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Untimed sensitivity diagnostics for unresolved primary outcomes; primary status counts and timings remain unchanged.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "raw_csv_sha256": hashlib.sha256((result_dir / "raw.csv").read_bytes()).hexdigest(),
        "number_of_cases": len(cases), "cases": cases,
    }
    (result_dir / "status_diagnostics.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"Completed {len(cases)} diagnostic cases.")


if __name__ == "__main__":
    main()
