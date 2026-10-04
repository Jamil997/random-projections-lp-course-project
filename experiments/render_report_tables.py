#!/usr/bin/env python3
"""Generate or check marked report blocks from a rerun's CSV/JSON results.

This script uses only the Python standard library. It edits only content between
the five exact GENERATED markers; all surrounding report text is preserved.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
BLOCK_NAMES = ("feasible_times", "feasible_quality", "legacy_status", "stress_status", "key_results")


def scientific(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def milliseconds(group, name, with_sd=False):
    mean = 1000 * group[name + "_mean"]
    if with_sd:
        sd = 1000 * group[name + "_sd"]
        return rf"${mean:.2f}\pm {sd:.2f}$"
    return f"{mean:.2f}"


def table(caption, label, specification, headers, rows):
    # Preserve the report's small font unless the table needs shrinking. A local
    # box and minipage avoid adding package or global style changes to the report.
    tabular = [rf"\begin{{tabular}}{{{specification}}}", r"\toprule",
               " & ".join(headers) + r" \\", r"\midrule"]
    tabular.extend(" & ".join(row) + r" \\" for row in rows)
    tabular.extend([r"\bottomrule", r"\end{tabular}"])
    lines = [r"\par\medskip", r"\noindent\begin{minipage}{\linewidth}",
             r"\small\linespread{1}\selectfont\centering", r"\setlength{\tabcolsep}{4pt}",
             r"\renewcommand{\arraystretch}{1.08}",
             rf"\captionof{{table}}{{{caption}}}\label{{{label}}}",
             r"\medskip", r"\sbox0{%", *tabular, r"}%",
             r"\ifdim\wd0>\linewidth", r"  \resizebox{\linewidth}{!}{\usebox0}",
             r"\else", r"  \usebox0", r"\fi", r"\end{minipage}", r"\par\medskip"]
    return "\n".join(lines) + "\n"


def make_blocks(summary, raw):
    blocks = {}
    feasible = sorted((group for group in summary if group["family"] == "feasible"),
                      key=lambda group: (group["m"], group["density"]))
    timing_rows = [[str(group["m"]), f"{group['density']:.1f}"] + [
        milliseconds(group, metric, with_sd=True) for metric in (
            "original_solve_s", "projected_total_s", "dual_total_s", "primal_total_s")]
        for group in feasible]
    blocks["feasible_times"] = table(
        "Fresh feasible-instance timings in milliseconds (mean $\\pm$ sample standard deviation; $d$: density). "
        "Projected totals include sampling, multiplication and solving; recovery totals add the indicated recovery.",
        "tab:fresh-feasible-times", "rr rrrr",
        ["$m$", "$d$", "Original", "Projected", "Dual total", "Primal total"], timing_rows)
    quality_rows = []
    for group in feasible:
        for method, name in (("raw", "Raw"), ("dual", "Dual"), ("primal", "Primal")):
            quality_rows.append([
                str(group["m"]), f"{group['density']:.1f}", name,
                scientific(group[f"{method}_feas_mean"]),
                f"{group[f'{method}_neg_mean']:.3f}",
                f"{group[f'{method}_obj_gap_mean']:.3f}",
                f"{group[f'{method}_accepted_feasible_total']}/{group['n_repetitions']}",
            ])
    blocks["feasible_quality"] = table(
        "Mean relative equality residual, negative mass ratio and relative objective gap "
        "($d$: density; Raw: projected primal; Dual/Primal: Algorithms 1/2). "
        "Accepted counts require equality feasibility and nonnegativity.",
        "tab:fresh-feasible-quality", "rr l rrr r",
        ["$m$", "$d$", "Method", "Eq. residual", "Neg. mass", "Obj. gap", "Accepted"], quality_rows)
    for family, block_name, title in (
        ("legacy_all_negative", "legacy_status", "Fresh all-negative-family results (legacy right-hand-side distribution)"),
        ("mixed_sign_stress", "stress_status", "Fresh mixed-sign stress-test results (additional family)"),
    ):
        groups = sorted((group for group in summary if group["family"] == family),
                        key=lambda group: (group["m"], group["density"],
                                           {"sparse": 0, "orthogonal": 1}[group["projector"]]))
        status_rows = []
        for group in groups:
            if group["projected_limit_count"] or group["projected_unbounded_count"]:
                raise ValueError("Status table needs expansion for limit/unbounded outcomes")
            status_rows.append([
                str(group["m"]), f"{group['density']:.1f}", group["projector"].capitalize(),
                milliseconds(group, "original_solve_s"), milliseconds(group, "projected_total_s"),
                str(group["projected_infeasible_count"]), str(group["projected_optimal_count"]),
                str(group["projected_error_count"]),
            ])
        blocks[block_name] = table(
            title + ". Here $d$ denotes density; times are means in milliseconds. Counts are primary projected-solver outcomes; "
            "all original LPs are infeasible.", "tab:fresh-" + block_name.replace("_", "-"),
            "rr l rr rrr",
            ["$m$", "$d$", "Projector", "Orig. (ms)", "Proj. (ms)", "Inf.", "Opt.", "Error"], status_rows)
    totals = {family: Counter(row["projected_status"] for row in raw if row["family"] == family)
              for family in ("feasible", "legacy_all_negative", "mixed_sign_stress")}
    counts = Counter(row["family"] for row in raw)
    independent = len({(row["family"], row["m"], row["n"], row["density"], row["repetition"]) for row in raw})
    accepted = {method: sum(int(row[f"{method}_accepted_feasible"]) for row in raw
                           if row["family"] == "feasible") for method in ("dual", "primal")}
    speedups = [group["projected_speedup"] for group in feasible]
    stress = totals["mixed_sign_stress"]
    blocks["key_results"] = (
        f"The fresh run comprises {len(raw)} projected LPs from {independent} independent originals. "
        f"Dual and primal recoveries yielded {accepted['dual']}/{counts['feasible']} and "
        f"{accepted['primal']}/{counts['feasible']} accepted original-feasible candidates, respectively. "
        f"Feasible-grid time ratios range from ${min(speedups):.2f}$ to ${max(speedups):.2f}$, "
        "measuring relaxation computation rather than successful original-LP solution. "
        f"All-negative projections preserved infeasibility in {totals['legacy_all_negative']['infeasible']}/"
        f"{counts['legacy_all_negative']} cases. The mixed-sign stress test produced "
        f"{stress['infeasible']} infeasible, {stress['optimal']} optimal and {stress['error']} unresolved error outcomes.\n"
    )
    assert tuple(blocks) == BLOCK_NAMES
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "results" / "baseline",
                        help="directory containing the benchmark raw.csv and summary.json")
    parser.add_argument("--report", type=Path, default=REPOSITORY / "report" / "main.tex")
    parser.add_argument("--check", action="store_true", help="verify generated blocks without changing files")
    args = parser.parse_args()
    summary = json.loads((args.output_dir / "summary.json").read_text())
    with (args.output_dir / "raw.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    blocks = make_blocks(summary, raw)
    if not args.report.exists():
        if args.check:
            parser.error("report does not exist; cannot check generated blocks")
        destination = Path(__file__).with_name("generated_report_blocks.tex")
        destination.write_text("\n".join(
            f"% BEGIN GENERATED {name}\n{blocks[name]}% END GENERATED {name}\n" for name in BLOCK_NAMES))
        print("Report not present; wrote experiments/generated_report_blocks.tex for insertion.")
        return
    original = args.report.read_text()
    updated = original
    stale = []
    # Validate every block before writing; an absent/duplicate marker never
    # causes an append or a rewrite of unrelated report prose.
    for name in BLOCK_NAMES:
        begin = f"% BEGIN GENERATED {name}"
        end = f"% END GENERATED {name}"
        if original.count(begin) != 1 or original.count(end) != 1:
            parser.error(f"expected exactly one pair of {name} markers")
        pattern = re.compile(r"(?m)^" + re.escape(begin) + r"\n(.*?)^" + re.escape(end) + r"$", re.DOTALL)
        match = pattern.search(updated)
        if match is None:
            parser.error(f"invalid marker boundaries for {name}")
        if match.group(1) != blocks[name]:
            stale.append(name)
        updated = updated[:match.start(1)] + blocks[name] + updated[match.end(1):]
    if args.check:
        if stale:
            print("FAIL: stale generated blocks: " + ", ".join(stale), file=sys.stderr)
            raise SystemExit(1)
        print("PASS: all five generated report blocks match raw.csv and summary.json exactly.")
    elif stale:
        args.report.write_text(updated)
        print("Updated generated blocks: " + ", ".join(stale))
    else:
        print("All generated report blocks are already current.")


if __name__ == "__main__":
    main()
