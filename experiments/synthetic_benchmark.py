#!/usr/bin/env python3
"""Portable runner for the scaling, compression, and sparsity experiments."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings
import numpy as np
import scipy
from scipy import sparse, linalg
from scipy.optimize import linprog, OptimizeWarning
from threadpoolctl import threadpool_limits, threadpool_info

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'experiments'))
import rerun
import netlib_benchmark as netlib
OUT = None
MASTER = 2026100403
TOL = 1e-7
OPTIONS = {**rerun.LP_OPTIONS, 'time_limit': 120.0}

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def seed(*parts):
    return int(np.random.SeedSequence([MASTER, *map(int, parts)]).generate_state(1, dtype=np.uint64)[0])

def generate(m, n, d, rep, phase):
    s = seed(1, phase, m, n, round(d * 10000), rep)
    rng = np.random.default_rng(s)
    # Same fixed-count sparsity and U[0,1] data distribution as the thesis,
    # now with explicit reproducible RNG streams. Not the later Bernoulli mask.
    A = sparse.random(m, n, density=d, format='csr', random_state=rng,
                      data_rvs=rng.random, dtype=np.float64)
    A.sort_indices()
    x0 = rng.random(n)
    # Build b in dense arithmetic, as in the original thesis, identically for
    # dense/CSR timing tracks. Generation is outside all reported solve times.
    b = A.toarray() @ x0
    return A, b, np.ones(n), s

def solve(A, b, c):
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=OptimizeWarning, message='Unrecognized options detected.*')
        r = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method='highs-ds', options=OPTIONS)
    return r, time.perf_counter() - start

def bytes_of(A):
    return int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) if sparse.issparse(A) else int(A.nbytes)

def quality(x, A, b, c, z):
    q = rerun.quality(x, A, b, c, z)
    q['relative_equality_inf'] = float(np.linalg.norm(A @ x - b, np.inf) / max(1., np.linalg.norm(b, np.inf)))
    q['negative_mass_absolute'] = float(np.maximum(-x, 0).sum())
    q['accepted_feasible_1e_3'] = int(q['relative_equality_inf'] <= 1e-3 and np.min(x) >= -1e-3)
    return q

def recover(xp, lam, A, b, c, S, method):
    # Same current two recovery heuristics, but only the selected CSR basis
    # is densified. Dense conversion cost, if any, is timed inside recovery.
    start = time.perf_counter()
    m, n = A.shape
    rank, warning_text = None, ''
    if method == 'dual':
        lifted = np.asarray(S.T @ lam).ravel()
        norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=0)).ravel()) if sparse.issparse(A) else np.linalg.norm(A, axis=0)
        scores = np.divide(c - np.asarray(A.T @ lifted).ravel(), norms,
                           out=np.full(n, np.inf), where=norms > 0)
        idx = np.argsort(scores, kind='stable')[:m]
    else:
        idx = np.argsort(-xp, kind='stable')[:m]
    B = A[:, idx]
    if sparse.issparse(B):
        B = B.toarray()
    if method == 'dual':
        try:
            with warnings.catch_warnings(record=True) as caught:
                vals = linalg.solve(B, b, assume_a='gen', check_finite=False)
            warning_text = '; '.join(str(w.message) for w in caught)
            status = 'ill_conditioned' if caught else 'computed'
        except linalg.LinAlgError as exc:
            return None, time.perf_counter()-start, 'singular', None, str(exc)
    else:
        vals, _, rank, _ = linalg.lstsq(B, b, lapack_driver='gelsd', check_finite=False)
        rank = int(rank)
        status = 'computed' if rank == m else 'rank_deficient'
    x = np.zeros(n)
    x[idx] = vals
    return x, time.perf_counter()-start, status, rank, warning_text

def save_record(stage, row):
    with (OUT / f'{stage}_raw.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
        f.flush()

def run_case(stage, m, n, d, rep, phase, modes):
    Acsr, b, c, instance_seed = generate(m, n, d, rep, phase)
    ident = dict(m=m, n=n, density=d, repetition=rep, instance_seed=instance_seed,
                 generation='legacy_fixed_count_uniform', stage=stage)
    vectors = {}
    baseline = {}
    for storage in sorted({x[0] for x in modes}):
        A = Acsr if storage == 'csr' else Acsr.toarray()
        original, original_time = solve(A, b, c)
        q = quality(original.x, A, b, c, original.fun) if original.status == 0 else {}
        baseline[storage] = (A, original, original_time)
        save_record(stage, {**ident, 'kind':'original', 'storage':storage,
                           'solve_s':original_time, 'status':rerun.STATUS[original.status],
                           'message':original.message, 'objective':float(original.fun) if original.fun is not None else None,
                           'iterations':int(original.nit), 'quality':q,
                           'matrix_bytes':bytes_of(A), 'nnz':int(Acsr.nnz)})
        if original.status == 0:
            vectors[f'original_{storage}'] = original.x
            vectors[f'original_dual_{storage}'] = original.eqlin.marginals
        print(f'{stage}: baseline {m}x{n} d={d:g} rep={rep} {storage}: {rerun.STATUS[original.status]}, {original_time:.3f}s', flush=True)
    # Randomized fixed-seed order prevents consistently favoring one projector.
    order_rng = np.random.default_rng(seed(99, phase, m, n, round(d*10000), rep))
    for j in order_rng.permutation(len(modes)):
        storage, kind, k, tags = modes[j]
        A, original, original_time = baseline[storage]
        if original.status != 0:
            save_record(stage, {**ident, 'kind':'skipped', 'storage':storage, 'projector':kind, 'k':k, 'tags':tags,
                               'reason':'original did not solve optimally'})
            continue
        ps = seed(2, phase, m, n, round(d*10000), rep, k, {'gaussian':1, 'countsketch':2}[kind])
        t0 = time.perf_counter()
        S = netlib.projector(kind, m, k, ps)
        sample_time = time.perf_counter()-t0
        t0 = time.perf_counter()
        SA, Sb = S @ A, np.asarray(S @ b).ravel()
        if sparse.issparse(SA):
            SA = SA.tocsr()
            SA.eliminate_zeros()
        multiply_time = time.perf_counter()-t0
        projected, solve_time = solve(SA, Sb, c)
        total = sample_time + multiply_time + solve_time
        row = {**ident, 'kind':'projection', 'storage':storage, 'projector':kind, 'k':k,
               'ratio':k/m, 'tags':tags, 'projector_seed':ps,
               'original_solve_s':original_time, 'sample_s':sample_time,
               'multiply_s':multiply_time, 'projected_solve_s':solve_time, 'projected_total_s':total,
               'projected_status':rerun.STATUS[projected.status], 'projected_message':projected.message,
               'original_objective':float(original.fun), 'original_iterations':int(original.nit),
               'projected_iterations':int(projected.nit), 'input_nnz':int(Acsr.nnz),
               'projected_nnz':int(SA.nnz if sparse.issparse(SA) else np.count_nonzero(SA)),
               'input_matrix_bytes':bytes_of(A), 'projector_bytes':bytes_of(S), 'projected_matrix_bytes':bytes_of(SA),
               'nonempty_sketch_rows':int(np.count_nonzero(np.diff(S.indptr))) if sparse.issparse(S) else k}
        if projected.status == 0:
            xp, lam = projected.x, projected.eqlin.marginals
            row['raw'] = quality(xp, A, b, c, original.fun)
            row['projected_objective'] = float(projected.fun)
            row['relative_lower_bound_gap'] = float((original.fun-projected.fun)/max(1.,abs(original.fun)))
            lifted = np.asarray(S.T @ lam).ravel()
            reduced = c - np.asarray(A.T @ lifted).ravel()
            row['checks'] = {
                'projected_eq_relative_inf':float(np.linalg.norm(SA@xp-Sb,np.inf)/max(1.,np.linalg.norm(Sb,np.inf))),
                'lifted_dual_min_reduced_cost':float(np.min(reduced)),
                'projected_dual_min_reduced_cost':float(np.min(c-np.asarray(SA.T@lam).ravel())),
                'dual_objective':float(b@lifted),
                'relative_projected_duality_gap':float(abs(c@xp-b@lifted)/max(1.,abs(c@xp))),
                'complementarity_inf':float(np.max(np.abs(xp*reduced)))}
            prefix = f'{storage}_{kind}_{k}'
            vectors[prefix+'_projected'] = xp
            vectors[prefix+'_lambda'] = lam
            for recovery_kind in order_rng.permutation(['dual','primal']):
                x, elapsed, status, rank, warning_text = recover(xp, lam, A, b, c, S, recovery_kind)
                row[recovery_kind+'_recovery_s'] = elapsed
                row[recovery_kind+'_total_s'] = total+elapsed
                row[recovery_kind+'_recovery_status'] = status
                row[recovery_kind+'_rank'] = rank
                row[recovery_kind+'_warning'] = warning_text
                if x is not None:
                    row[recovery_kind] = quality(x, A, b, c, original.fun)
                    vectors[prefix+'_'+recovery_kind] = x
        save_record(stage, row)
        gap = row.get('relative_lower_bound_gap', float('nan'))
        print(f'  {kind} k={k} {storage}: {row["projected_status"]}, projection {total:.3f}s, speedup {original_time/total:.2f}x, boundgap {gap:.3%}', flush=True)
    vector_path = OUT / f'{stage}_vectors_{m}_{n}_{round(d*10000)}_{rep}.npz'
    np.savez_compressed(vector_path, **vectors)

def metadata(stage, protocol):
    return dict(stage=stage, started_utc=now(), master_seed=MASTER,
                python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                highs=rerun.highs_core.HIGHS_VERSION_MAJOR.__str__()+'.'+str(rerun.highs_core.HIGHS_VERSION_MINOR)+'.'+str(rerun.highs_core.HIGHS_VERSION_PATCH),
                platform=platform.platform(), solver='scipy.optimize.linprog highs-ds', options=OPTIONS,
                threadpools=threadpool_info(), thread_environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
                protocol=protocol, helper_hashes={str(p.relative_to(REPO)):digest(p) for p in (REPO/'experiments/rerun.py', REPO/'experiments/netlib_benchmark.py')},
                runner_hash=digest(Path(__file__)),
                timing_scope='Sequential solves, one thread requested for numerical libraries and HiGHS. Original solve; sampling, SA/Sb multiplication, projected solve; existing two recoveries separately. Excludes generation, quality checks, and result writes.',
                storage_scope='Array-buffer sizes only; not process peak RAM. Sparse recovery densifies selected m by m basis and includes that cost.',
                comparison_scope='Matched thesis size/generator/sketch dimensions, but controlled current solver and stable current recovery; not exact historical code or hardware replication.')

def main():
    global OUT, MASTER
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=REPO/'results/synthetic/main_protocol.json')
    p.add_argument('--seed', type=int, default=2026100403)
    p.add_argument('--smoke', action='store_true', help='Small plumbing check, not a benchmark result')
    args = p.parse_args()
    if args.seed < 0:
        p.error('--seed must be nonnegative')
    OUT = args.output_dir.resolve()
    if OUT.exists() and any(OUT.iterdir()):
        p.error('Use a fresh, empty output directory; measured evidence is never overwritten.')
    protocol = json.loads(args.protocol.read_text())
    if args.smoke:
        protocol = {'replications': 1, 'time_limit_s': 30.0, 'purpose': 'smoke test only',
                    'cases': [{'m': 30, 'n': 45, 'density': 0.5, 'modes': [
                        {'storage': 'dense', 'projector': 'gaussian', 'k': 15, 'tags': ['smoke']},
                        {'storage': 'csr', 'projector': 'gaussian', 'k': 15, 'tags': ['smoke']},
                        {'storage': 'csr', 'projector': 'countsketch', 'k': 15, 'tags': ['smoke']}]}]}
    for case in protocol['cases']:
        if case.get('replications', protocol['replications']) < 1:
            p.error('Each case requires at least one repetition')
        if not 0 < case['density'] <= 1 or case['n'] < case['m']:
            p.error('Require 0 < density <= 1 and n >= m')
        for mode in case['modes']:
            if not 0 < mode['k'] <= case['m'] or mode['storage'] not in ('dense','csr') or mode['projector'] not in ('gaussian','countsketch'):
                p.error('Invalid sketch configuration')
    MASTER = args.seed
    OPTIONS['time_limit'] = protocol['time_limit_s']
    OUT.mkdir(parents=True, exist_ok=True)
    stage = 'main'
    meta = metadata(stage, protocol)
    meta['purpose'] = 'smoke' if args.smoke else 'benchmark'
    meta['seed_phase'] = 2
    meta['portable_source'] = 'experiments/synthetic_benchmark.py'
    dump(OUT/'main_protocol.json', protocol)
    dump(OUT/'main_environment.json', meta)
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        dump(OUT/'deterministic_checks.json', rerun.deterministic_checks())
        for case in protocol['cases']:
            for rep in range(case.get('replications', protocol['replications'])):
                run_case(stage,case['m'],case['n'],case['density'],rep,2,
                         [(x['storage'],x['projector'],x['k'],x['tags']) for x in case['modes']])
    meta['finished_utc'] = now()
    meta['elapsed_s'] = time.perf_counter()-start
    dump(OUT/'main_environment.json', meta)
    print(f'Completed in {meta["elapsed_s"]:.1f}s. Next: verify_synthetic.py --output-dir "{OUT}" --write-summary')

if __name__ == '__main__':
    main()
