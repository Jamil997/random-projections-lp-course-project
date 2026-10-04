#!/usr/bin/env python3
"""Read-only checks of recorded Netlib vectors, certificates and summaries.

Reconstructs the sketches from saved seeds; does not rerun the timed experiment.
Scalar metrics are recomputed independently of the benchmark's quality helper.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import numpy as np
from scipy import sparse
import netlib_benchmark as bench

ROOT = Path(__file__).resolve().parents[1]


def close(a, b):
    if a is None or b is None:
        assert a is b, (a, b)
    elif isinstance(a, (float, int)):
        assert np.isclose(a, b, rtol=1e-8, atol=1e-9), (a, b)
    else:
        assert a == b, (a, b)


def metrics(x, A, b, c, optimum):
    residual = A @ x - b
    eq = max(abs(residual))
    return dict(feas=sum(abs(residual))/sum(abs(b)),
                neg=sum(np.maximum(-x, 0))/sum(abs(x)) if sum(abs(x)) else 0,
                eq_residual_inf=eq, min_x=min(x), objective=c @ x,
                obj_gap=abs(c @ x - optimum)/abs(optimum),
                accepted_feasible=int(eq <= 1e-7*max(1, max(abs(b))) and min(x) >= -1e-7))


def unit_checks():
    C = sparse.csr_matrix([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]])
    lo = np.array([5., -np.inf, 7., 2., -np.inf])
    hi = np.array([5., 9., np.inf, 20., np.inf])
    A, b, c = bench.standard_form(C, lo, hi, np.array([1., -2.]))
    np.testing.assert_array_equal(A.toarray(), [
        [1, 2, 0, 0, 0, 0], [3, 4, 1, 0, 0, 0],
        [5, 6, 0, -1, 0, 0], [7, 8, 0, 0, 1, 0], [7, 8, 0, 0, 0, -1]])
    np.testing.assert_array_equal(b, [5, 9, 7, 20, 2])
    np.testing.assert_array_equal(c, [1, -2, 0, 0, 0, 0])
    for method in bench.METHODS:
        S = bench.projector(method, 17, 8, 123)
        T = bench.projector(method, 17, 8, 123)
        a = S.toarray() if sparse.issparse(S) else S
        t = T.toarray() if sparse.issparse(T) else T
        np.testing.assert_array_equal(a, t)
        assert a.shape == (8, 17)
        if method == 'countsketch':
            np.testing.assert_array_equal(np.sum(abs(a), axis=0), np.ones(17))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/netlib')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/netlib')
    args = parser.parse_args()
    unit_checks()
    out = args.output_dir
    env = json.loads((out/'environment.json').read_text())
    manifest = json.loads((args.data_dir/'manifest.json').read_text())
    assert bench.sha(args.data_dir/'manifest.json') == env['dataset_manifest_sha256']
    for name, digest in env['source_hashes'].items():
        assert bench.sha(ROOT/'experiments'/name) == digest, name
    for name, digest in env['artifact_hashes'].items():
        assert bench.sha(out/name) == digest, name
    rows = json.loads((out/'raw.json').read_text())
    summaries = json.loads((out/'summary.json').read_text())
    cases = json.loads((out/'datasets.json').read_text())
    controls = json.loads((out/'controls.json').read_text())
    vectors = np.load(out/'vectors.npz', allow_pickle=False)
    assert len(rows) == env['runs'] == len(cases)*len(env['ratios'])*len(env['methods'])*env['repetitions']
    assert len({r['id'] for r in rows}) == len(rows)
    csv_rows = list(csv.DictReader((out/'raw.csv').open(newline='')))
    assert len(csv_rows) == len(rows)
    for row, csv_row in zip(rows, csv_rows):
        for key, value in row.items():
            assert csv_row[key] == (str(value) if value is not None else ''), (row['id'], key)
    expected_vectors = set()
    for di, entry in enumerate(manifest['datasets']):
        name = entry['name']
        path = args.data_dir/(name+'.mps')
        assert bench.sha(path) == entry['sha256_mps']
        C, lower, upper, nc = bench.read_case(path)
        A, b, c = bench.standard_form(C, lower, upper, nc)
        case = next(d for d in cases if d['dataset'] == name)
        close(case['objective'], entry['reference_objective'])
        for key in ('original', 'control'):
            vkey = name+'_'+key
            expected_vectors.add(vkey)
            q = metrics(vectors[vkey], A, b, c, case['objective'])
            assert q['accepted_feasible'] and q['obj_gap'] < 1e-6
            assert bench.native_quality(vectors[vkey][:C.shape[1]], C, lower, upper)['native_accepted']
            if key == 'control':
                control = next(x for x in controls if x['dataset'] == name)
                for metric, value in q.items():
                    close(control[metric], value)
        for ri, ratio in enumerate(env['ratios']):
            for rep in range(env['repetitions']):
                group = [r for r in rows if (r['dataset'], r['ratio'], r['repetition']) == (name, ratio, rep)]
                assert {r['projector'] for r in group} == set(env['methods'])
                assert len({r['original_solve_s'] for r in group}) == 1
                for row in group:
                    mi = env['methods'].index(row['projector'])
                    assert row['projector_seed'] == bench.seed_for(env['seed'], di, ri, mi, rep)
                    assert row['k'] == int(A.shape[0]*ratio)
                    S = bench.projector(row['projector'], A.shape[0], row['k'], row['projector_seed'])
                    SA = S @ A
                    if sparse.issparse(SA):
                        SA = SA.tocsr()
                        SA.eliminate_zeros()
                    close(row['projected_density'], bench.density(SA))
                    assert row['projected_matrix_bytes'] == bench.array_bytes(SA)
                    assert row['original_matrix_bytes'] == bench.array_bytes(A)
                    close(row['projected_total_s'], sum(row[k] for k in ('sample_s','multiply_s','projected_solve_s')))
                    for prefix in ('raw', 'dual', 'primal'):
                        vkey = row['id']+'_'+prefix
                        if prefix+'_accepted_feasible' not in row:
                            assert vkey not in vectors.files
                            continue
                        expected_vectors.add(vkey)
                        x = vectors[vkey]
                        for key, value in metrics(x, A, b, c, row['original_objective']).items():
                            close(row[prefix+'_'+key], value)
                        if prefix == 'raw':
                            assert max(abs(SA@x-S@b)) <= 1e-6*max(1,max(abs(S@b)))
                            assert min(x) >= -1e-7
                            assert c@x <= row['original_objective'] + 1e-6*max(1,abs(row['original_objective']))
                            for key, value in bench.native_quality(x[:C.shape[1]], C, lower, upper).items():
                                close(row[key], value)
                        else:
                            close(row[prefix+'_total_s'], row['projected_total_s']+row[prefix+'_recovery_s'])
                    if row['projected_status'] == 'unbounded':
                        vkey = row['id']+'_ray'
                        expected_vectors.add(vkey)
                        d = vectors[vkey]
                        assert min(d) >= -1e-7 and max(abs(SA@d)) <= 1e-6
                        assert abs(c@d+1) <= 1e-6 and row['ray_verified'] == 1
                        close(row['ray_residual'], max(abs(SA@d)))
                        close(row['original_ray_residual'], max(abs(A@d)))
                        assert max(abs(A@d)) > 1e-6
    assert expected_vectors == set(vectors.files)
    for s in summaries:
        group = [r for r in rows if all(r[k] == s[k] for k in ('dataset','ratio','projector'))]
        assert len(group) == s['runs'] == env['repetitions']
        counts = Counter(r['projected_status'] for r in group)
        for status in bench.rerun.STATUS.values():
            assert s[status+'_count'] == counts[status]
        for prefix in ('raw','dual','primal'):
            assert s[prefix+'_available'] == sum(prefix+'_accepted_feasible' in r for r in group)
            assert s[prefix+'_accepted'] == sum(r.get(prefix+'_accepted_feasible',0) for r in group)
        assert s['ray_verified'] == sum(r.get('ray_verified',0) for r in group)
        for key in s:
            if key.endswith('_mean'):
                metric = key[:-5]
                vals = [r[metric] for r in group if r.get(metric) is not None]
                assert s[metric+'_count'] == len(vals)
                close(s[key], statistics.mean(vals) if vals else None)
                close(s[metric+'_sd'], statistics.stdev(vals) if len(vals)>1 else (0 if vals else None))
    print(f'PASS: {len(rows)} projections, {len(cases)} fixed LPs, {len(controls)} controls; '
          f'{sum(r.get("ray_verified",0) for r in rows)} unbounded-ray certificates; '
          'hashes, vectors, metrics, means/SDs and conversion/sketch unit checks.')


if __name__ == '__main__':
    main()
