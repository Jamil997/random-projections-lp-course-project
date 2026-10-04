#!/usr/bin/env python3
"""Post-run verification of saved candidates, followed by JSON-only summaries."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from scipy import sparse
import argparse
import ast
import hashlib
import synthetic_benchmark as e

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=e.REPO/'results/synthetic')
parser.add_argument('--write-summary',action='store_true',help='Write missing summary and verification; never changes raw measurements')
args=parser.parse_args()
OUT=args.output_dir.resolve()
rows=[json.loads(s) for s in (OUT/'main_raw.jsonl').read_text().splitlines()]
protocol=json.loads((OUT/'main_protocol.json').read_text())
env=json.loads((OUT/'main_environment.json').read_text())
assert 'finished_utc' in env, 'Run has not finished.'
assert env['protocol']==protocol, 'Protocol differs from recorded execution.'
e.MASTER=env['master_seed']
for relative,checksum in env['helper_hashes'].items():
    assert not Path(relative).is_absolute()
    assert e.digest(e.REPO/relative)==checksum, ('Helper source changed',relative)
provenance_path=OUT/'source_provenance.json'
if provenance_path.exists():
    provenance=json.loads(provenance_path.read_text())
    assert provenance['measured_runner_sha256']==env['runner_hash']
    assert provenance['portable_runner_sha256']==e.digest(Path(e.__file__))
    functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(e.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    for name,checksum in provenance['numerical_kernel_ast_sha256'].items():
        assert hashlib.sha256(functions[name].encode()).hexdigest()==checksum, name
else:
    assert env['runner_hash']==e.digest(Path(e.__file__)), 'Runner source changed.'
manifest=OUT/'artifact_hashes.json'
if manifest.exists():
    for name,checksum in json.loads(manifest.read_text()).items():
        assert Path(name).name==name, 'Unexpected manifest path.'
        assert e.digest(OUT/name)==checksum, ('Evidence checksum mismatch',name)
expected_keys=set()
for case in protocol['cases']:
    for rep in range(case.get('replications',protocol['replications'])):
        base=(case['m'],case['n'],case['density'],rep)
        for storage in {m['storage'] for m in case['modes']}:
            expected_keys.add(('original',*base,storage,None,None))
        for mode in case['modes']:
            expected_keys.add(('projection',*base,mode['storage'],mode['projector'],mode['k']))
actual_keys=[]
for r in rows:
    actual_keys.append(('original' if r['kind']=='original' else 'projection',r['m'],r['n'],r['density'],r['repetition'],r['storage'],r.get('projector'),r.get('k')))
assert len(actual_keys)==len(set(actual_keys)), 'Duplicate primary records.'
assert set(actual_keys)==expected_keys, 'Missing or unexpected configurations.'
expected_original=sum(case.get('replications',protocol['replications'])*len({m['storage'] for m in case['modes']}) for case in protocol['cases'])
expected_projected=sum(case.get('replications',protocol['replications'])*len(case['modes']) for case in protocol['cases'])
assert sum(r['kind']=='original' for r in rows)==expected_original
assert sum(r['kind'] in ('projection','skipped') for r in rows)==expected_projected

case_groups=defaultdict(list)
for r in rows:
    case_groups[(r['m'],r['n'],r['density'],r['repetition'])].append(r)
maxima=defaultdict(float)
count=Counter()

def close(a,b,name):
    assert np.isclose(a,b,rtol=2e-8,atol=2e-9), (name,a,b)

def independent_quality(x,A,b,c,z,q):
    residual=A@x-b
    eqinf=float(np.linalg.norm(residual,np.inf))
    eql1=float(np.linalg.norm(residual,1)/np.linalg.norm(b,1))
    neg=float(np.maximum(-x,0).sum()/np.linalg.norm(x,1)) if np.linalg.norm(x,1) else 0.
    accepted=int(eqinf<=1e-7*max(1.,np.linalg.norm(b,np.inf)) and np.min(x)>=-1e-7)
    close(q['objective'],c@x,'objective')
    close(q['feas'],eql1,'equality residual')
    close(q['eq_residual_inf'],eqinf,'equality infinity residual')
    close(q['neg'],neg,'negative mass')
    close(q['min_x'],np.min(x),'minimum coordinate')
    close(q['obj_gap'],abs(c@x-z)/abs(z),'objective error')
    assert q['accepted_feasible']==accepted
    close(q['relative_equality_inf'],eqinf/max(1.,np.linalg.norm(b,np.inf)),'relative equality residual')
    close(q['negative_mass_absolute'],np.maximum(-x,0).sum(),'absolute negative mass')
    assert q['accepted_feasible_1e_3']==int(eqinf<=1e-3*max(1.,np.linalg.norm(b,np.inf)) and np.min(x)>=-1e-3)
    count['candidate_metric_checks']+=1

for (m,n,d,rep), group in case_groups.items():
    Acsr,b,c,instseed=e.generate(m,n,d,rep,2)
    assert Acsr.nnz==round(m*n*d)
    D=Acsr.toarray()
    archive=OUT/f'main_vectors_{m}_{n}_{round(d*10000)}_{rep}.npz'
    with np.load(archive) as vectors:
        for r in group:
            assert r['instance_seed']==instseed
            A=Acsr if r['storage']=='csr' else D
            if r['kind']=='original':
                if r['status']=='optimal':
                    x=vectors[f'original_{r["storage"]}']
                    lam=vectors[f'original_dual_{r["storage"]}']
                    independent_quality(x,A,b,c,r['objective'],r['quality'])
                    assert r['quality']['accepted_feasible']==1
                    assert np.min(c-A.T@lam)>=-1e-6
                    assert abs(c@x-b@lam)<=1e-6*max(1,abs(c@x))
                    count['original_optima_verified']+=1
                continue
            if r['kind']=='skipped':
                count['skipped_due_to_original_status']+=1
                continue
            close(r['projected_total_s'],r['sample_s']+r['multiply_s']+r['projected_solve_s'],'timing sum')
            if r['projected_status']!='optimal':
                continue
            S=e.netlib.projector(r['projector'],m,r['k'],r['projector_seed'])
            SA=S@A
            if sparse.issparse(SA):
                SA.eliminate_zeros()
            Sb=np.asarray(S@b).ravel()
            prefix=f'{r["storage"]}_{r["projector"]}_{r["k"]}'
            x=vectors[prefix+'_projected']
            lam=vectors[prefix+'_lambda']
            independent_quality(x,A,b,c,r['original_objective'],r['raw'])
            eq=float(np.linalg.norm(SA@x-Sb,np.inf)/max(1.,np.linalg.norm(Sb,np.inf)))
            rc=c-np.asarray(A.T@(S.T@lam)).ravel()
            dualgap=float(abs(c@x-Sb@lam)/max(1.,abs(c@x)))
            assert eq<=1e-6 and np.min(x)>=-1e-7
            assert np.min(rc)>=-1e-6 and dualgap<=1e-6
            assert c@x<=r['original_objective']+1e-6*max(1.,abs(r['original_objective']))
            close(r['relative_lower_bound_gap'],(r['original_objective']-c@x)/max(1.,abs(r['original_objective'])),'lower-bound gap')
            close(r['checks']['projected_eq_relative_inf'],eq,'projected residual')
            close(r['checks']['lifted_dual_min_reduced_cost'],np.min(rc),'lifted dual')
            close(r['checks']['relative_projected_duality_gap'],dualgap,'duality gap')
            maxima['projected_equality_relative_inf']=max(maxima['projected_equality_relative_inf'],eq)
            maxima['dual_infeasibility']=max(maxima['dual_infeasibility'],float(max(0,-np.min(rc))))
            maxima['relative_projected_duality_gap']=max(maxima['relative_projected_duality_gap'],dualgap)
            nnz=SA.nnz if sparse.issparse(SA) else np.count_nonzero(SA)
            assert r['projected_nnz']==nnz
            assert r['input_matrix_bytes']==e.bytes_of(A)
            assert r['projected_matrix_bytes']==e.bytes_of(SA)
            assert r['projector_bytes']==e.bytes_of(S)
            nonempty=int(np.count_nonzero(np.diff(S.indptr))) if sparse.issparse(S) else r['k']
            assert r['nonempty_sketch_rows']==nonempty
            original=next(o for o in group if o['kind']=='original' and o['storage']==r['storage'])
            close(r['original_solve_s'],original['solve_s'],'paired original timing')
            close(r['original_objective'],original['objective'],'paired original objective')
            assert r['projector_seed']==e.seed(2,2,m,n,round(d*10000),rep,r['k'],{'gaussian':1,'countsketch':2}[r['projector']])
            if r['projector']=='countsketch':
                assert np.all(np.diff(S.tocsc().indptr)==1)
            for method in ('dual','primal'):
                close(r[method+'_total_s'],r['projected_total_s']+r[method+'_recovery_s'],'recovery total')
                if method in r:
                    independent_quality(vectors[prefix+'_'+method],A,b,c,r['original_objective'],r[method])
            count['projected_optima_verified']+=1
    count['independent_instances_verified']+=1

groups=defaultdict(list)
for r in rows:
    if r['kind']=='projection':
        groups[(r['m'],r['n'],r['density'],r['storage'],r['projector'],r['k'])].append(r)

def stats(values):
    values=[float(v) for v in values if v is not None]
    return {'n':len(values),'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1)) if len(values)>1 else None,
            'min':min(values),'max':max(values),'median':float(np.median(values))} if values else {'n':0}

summary=[]
for key, group in sorted(groups.items()):
    m,n,d,storage,projector,k=key
    s=dict(m=m,n=n,density=d,storage=storage,projector=projector,k=k,ratio=k/m,
           runs=len(group),tags=group[0]['tags'],statuses=dict(Counter(r['projected_status'] for r in group)))
    for metric in ('original_solve_s','sample_s','multiply_s','projected_solve_s','projected_total_s',
                   'dual_total_s','primal_total_s','relative_lower_bound_gap','input_matrix_bytes',
                   'projected_matrix_bytes','input_nnz','projected_nnz','nonempty_sketch_rows'):
        s[metric]=stats(r.get(metric) for r in group)
    for method in ('raw','dual','primal'):
        candidates=[r[method] for r in group if method in r]
        s[method]={'available':len(candidates),'accepted':sum(q['accepted_feasible'] for q in candidates),
                   'accepted_1e_3':sum(q['accepted_feasible_1e_3'] for q in candidates)}
        for metric in ('feas','neg','obj_gap','relative_equality_inf','min_x'):
            s[method][metric]=stats(q[metric] for q in candidates)
        if method!='raw':
            s[method]['statuses']=dict(Counter(r.get(method+'_recovery_status','unavailable') for r in group))
    s['speedup_projection_ratio_of_means']=s['original_solve_s']['mean']/s['projected_total_s']['mean']
    for method in ('dual','primal'):
        if s[method+'_total_s']['n']:
            s['speedup_'+method+'_ratio_of_means']=s['original_solve_s']['mean']/s[method+'_total_s']['mean']
    s['paired_projection_speedups']=stats(r['original_solve_s']/r['projected_total_s'] for r in group)
    s['projected_fill_fraction']=stats(r['projected_nnz']/(r['k']*r['n']) for r in group)
    summary.append(s)


def compare(a,b,path='summary'):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(), path
        for key in a: compare(a[key],b[key],path+'.'+key)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+'.'+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        close(a,b,path)
    else:
        assert a==b,(path,a,b)

summary_path=OUT/'main_summary.json'
if summary_path.exists():
    compare(summary,json.loads(summary_path.read_text()))
elif args.write_summary:
    e.dump(summary_path,summary)
else:
    raise AssertionError('Missing summary; use --write-summary for a new run.')
verification={'passed':True,'counts':dict(count),'maxima':dict(maxima),
              'expected_original_runs':expected_original,'expected_projected_or_skipped_runs':expected_projected,
              'original_status_counts':dict(Counter(r['status'] for r in rows if r['kind']=='original')),
              'projected_status_counts':dict(Counter(r['projected_status'] for r in rows if r['kind']=='projection')),
              'summary_recomputed_and_checked':True,
              'source_and_artifact_checksums_checked':True,
              'verifier_sha256':e.digest(Path(__file__))}
if args.write_summary:
    e.dump(OUT/'verification.json',verification)
elif (OUT/'verification.json').exists():
    compare(verification,json.loads((OUT/'verification.json').read_text()),'verification')
print(json.dumps(verification,indent=2))
print('Raw, primal-recovery, dual-recovery accepted:',*[sum(s[p]['accepted'] for s in summary) for p in ('raw','primal','dual')])

