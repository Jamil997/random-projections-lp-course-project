#!/usr/bin/env python3
"""Generate/check Netlib report blocks from committed JSON results."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from render_report_tables import table

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'gaussian': 'Gaussian', 'sparse': 'Sparse', 'countsketch': 'CountSketch'}


def blocks(out):
    cases = json.loads((out/'datasets.json').read_text())
    rows = json.loads((out/'raw.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    content = {}
    content['netlib_datasets'] = table(
        'Benchmark dimensions after exact standard-form conversion. Original storage is CSR; '
        'objective values are those recomputed by HiGHS.', 'tab:netlib-datasets', 'l rrr rr',
        ['LP', '$m$', '$n$', r'$\operatorname{nnz}(A)$', 'CSR (KiB)', 'Optimum'],
        [[d['dataset'].upper(), str(d['m']), str(d['n']), str(d['nnz']),
          f"{d['csr_bytes']/1024:.2f}", f"{d['objective']:.6f}"] for d in cases])
    index = {(s['dataset'],s['ratio'],s['projector']): s for s in summary}
    status_rows = []
    for d in cases:
        for method in NAMES:
            pair = [index[d['dataset'],ratio,method] for ratio in (0.5,0.75)]
            assert all(sum(s[k+'_count'] for k in ('infeasible','limit','error')) == 0 for s in pair)
            status_rows.append([d['dataset'].upper(), NAMES[method]] + [
                str(s[key+'_count']) for s in pair for key in ('optimal','unbounded')])
    content['netlib_status'] = table(
        'Projected status counts, ten sketches per cell pair. O: finite optimum; U: unbounded. '
        'No infeasible, limit, or error outcomes occurred.', 'tab:netlib-status', 'll rrrr',
        ['LP', 'Projector', r'$\rho=.5$: O', 'U', r'$\rho=.75$: O', 'U'], status_rows)
    totals = Counter(r['projected_status'] for r in rows)
    ray = sum(r.get('ray_verified',0) for r in rows)
    accepted = {p: sum(r.get(p+'_accepted_feasible',0) for r in rows) for p in ('raw','dual','primal')}
    dual_singular = sum(r.get('dual_recovery_status') == 'singular' for r in rows)
    primal_rank = sum(r.get('primal_recovery_status') == 'rank_deficient' for r in rows)
    content['netlib_findings'] = (
        f"Across {len(rows)} projected LPs, {totals['unbounded']} are unbounded and "
        f"{totals['optimal']} have finite optima. All {ray} unbounded outcomes have passing "
        "numerical ray certificates; none of the stored directions satisfies the original equalities "
        "at the certificate tolerance. "
        f"Of the {totals['optimal']} finite projected optima, {accepted['raw']} raw vectors "
        "pass original feasibility. "
        f"Dual recovery encounters a singular selected basis in {dual_singular}/{totals['optimal']} attempts "
        "and produces no candidate. "
        f"Primal recovery produces {totals['optimal']} candidates, all from rank-deficient "
        f"selected systems ({primal_rank}/{totals['optimal']}); {accepted['primal']} pass original feasibility. "
        "These are failures of the specific recovery heuristics, not evidence that no feasible "
        "retrieval method exists.\n")
    resource_rows = []
    for d in cases:
        for method in NAMES:
            s = index[d['dataset'],.75,method]
            resource_rows.append([d['dataset'].upper(), NAMES[method],
                f"{s['original_solve_s_mean']*1000:.2f}", f"{s['projected_total_s_mean']*1000:.2f}",
                f"{s['projected_density_mean']*100:.1f}", f"{s['projected_matrix_bytes_mean']/1024:.2f}"])
    content['netlib_resources'] = table(
        r'Mean time and projected-matrix storage at $\rho=0.75$ (ten sketches per row). '
        'Original/projected times are milliseconds; projected totals include sampling and multiplication. '
        'Density is the percentage of nonzero matrix entries. All statuses are included.',
        'tab:netlib-resources', 'll rrrr',
        ['LP', 'Projector', 'Orig. (ms)', 'Proj. (ms)', r'Density (\%)', '$SA$ (KiB)'], resource_rows)
    return content


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/netlib')
    p.add_argument('--report', type=Path, default=ROOT/'report/main.tex')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    original = a.report.read_text()
    updated = original
    for name, value in blocks(a.output_dir).items():
        begin, end = f'% BEGIN GENERATED {name}', f'% END GENERATED {name}'
        assert original.count(begin) == original.count(end) == 1
        pattern = re.compile(r'(?m)^'+re.escape(begin)+r'\n(.*?)^'+re.escape(end)+r'$', re.DOTALL)
        match = pattern.search(updated)
        assert match is not None
        updated = updated[:match.start(1)] + value + updated[match.end(1):]
    if a.check:
        assert original == updated, 'Stale Netlib report blocks'
        print('PASS: all four Netlib report blocks match the recorded results.')
    else:
        a.report.write_text(updated)
        print('Generated four Netlib report blocks.')


if __name__ == '__main__':
    main()
