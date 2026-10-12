"""Reweight the saved EG classifiers."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('FAIRREPAIR_INPUT', ROOT / 'data'))
BASE = DATA / 'baselines'
OUT = Path(os.environ.get('FAIRREPAIR_OUTPUT', ROOT / 'results'))
CACHE = OUT / 'cache'
PARTS = ('repair', 'validation', 'test')
EPS, DELTA, RHO = .02, .005, .25


def group_definitions(raw, meta):
    definitions = []
    shape = [len(v) for v in meta['levels'][:2]]
    for a in range(2):
        for c, name in enumerate(meta['levels'][a]):
            definitions.append((meta['attributes'][a], name, ((a, c),)))
    for a in range(shape[0]):
        for b in range(shape[1]):
            definitions.append(('intersection', f'{meta["levels"][0][a]} / {meta["levels"][1][b]}', ((0, a), (1, b))))
    masks = []
    for family, name, values in definitions:
        mm = {p: np.logical_and.reduce([raw[p+'_groups'][:, a] == c for a, c in values]) for p in PARTS}
        if np.sum(mm['repair'] & (raw['repair_y'] == 1)) >= 20:
            assert all(np.sum(mm[p] & (raw[p+'_y'] == 1)) > 0 for p in PARTS)
            masks.append((family, name, mm))
    return masks


def coefficients(H, raw, base, groups):
    out = {}
    for p in PARTS:
        y = raw[p+'_y']; b = base['Original__'+p]; hh = H[p].astype(float)
        acc = (1-y).mean() + (2*y-1) @ hh / len(y)
        withdrawal = 1 - b @ hh / b.sum()
        rate = {}; differences = {}
        for family, name, mm in groups:
            rate.setdefault(family, []).append(hh[mm[p] & (y == 1)].mean(axis=0))
        for family, rr in rate.items():
            rr = np.asarray(rr)
            differences[family] = np.asarray([rr[i]-rr[j] for i in range(len(rr)) for j in range(len(rr)) if i != j])
        out[p] = dict(acc=acc, withdrawal=withdrawal, differences=differences, rates=rate)
    return out


def solve(coeff, w0, validation, trust, exact=False, epsilon=EPS, delta=DELTA, rho=RHO):
    m = len(w0); A = []; b = []; caps = {}
    for p in (('repair', 'validation') if validation else ('repair',)):
        for family, diffs in coeff[p]['differences'].items():
            cap = max(epsilon, float(np.max(diffs @ w0)))
            A.extend(np.c_[diffs, np.zeros_like(diffs)])
            b.extend([cap] * len(diffs)); caps[p+'/'+family] = cap
        A.append(np.r_[-coeff[p]['acc'], np.zeros(m)])
        b.append(-float(coeff[p]['acc'] @ w0) + delta)
    eye = np.eye(m)
    A.extend(np.c_[eye, -eye]); b.extend(w0)
    A.extend(np.c_[-eye, -eye]); b.extend(-w0)
    if trust:
        A.append(np.r_[np.zeros(m), np.ones(m)]); b.append(rho)
    A = np.asarray(A); b = np.asarray(b)
    eq = np.r_[np.ones(m), np.zeros(m)][None, :]
    rhs = [1.]
    if exact:
        match = np.vstack([coeff['repair']['rates']['intersection'], coeff['repair']['acc']])
        eq = np.vstack([eq, np.c_[match, np.zeros_like(match)]])
        rhs.extend(match @ w0)
    objective = np.r_[coeff['repair']['withdrawal'], np.zeros(m)]
    assert np.max(A @ np.r_[w0, np.zeros(m)] - b) < 1e-8
    fit = linprog(objective, A_ub=A, b_ub=b, A_eq=eq, b_eq=rhs, bounds=[(0, 1)]*m+[(0, None)]*m, method='highs')
    if not fit.success:
        raise RuntimeError(fit.message)
    tie = linprog(np.r_[np.zeros(m), np.ones(m)], A_ub=np.vstack([A, objective]),
                  b_ub=np.r_[b, fit.fun+1e-10], A_eq=eq, b_eq=rhs,
                  bounds=[(0, 1)]*m+[(0, None)]*m, method='highs')
    if not tie.success:
        raise RuntimeError(tie.message)
    w = tie.x[:m]
    residual = max(float(np.max(A @ tie.x-b)), float(np.max(np.abs(eq @ tie.x-rhs))), 0.)
    assert residual < 1e-7
    assert coeff['repair']['withdrawal'] @ w <= coeff['repair']['withdrawal'] @ w0 + 1e-8
    return w, dict(caps=caps, max_constraint_violation=residual, dictionary_size=m,
                   weight_l1=float(np.abs(w-w0).sum()), optimal_repair_W=float(fit.fun),
                   original_weights=w0.tolist(), weights=w.tolist(), validation=validation, trust=trust, exact=exact)


def evaluate(q, y, baseline, mask):
    qq=q[mask]; yy=y[mask]; bb=baseline[mask]
    return dict(n=int(mask.sum()), n_positive=int(yy.sum()), n_base_positive=int(bb.sum()),
                accuracy=float(np.mean(yy*qq+(1-yy)*(1-qq))),
                tpr=float(qq[yy==1].mean()),
                withdrawal=float(np.sum(bb*(1-qq))/bb.sum()))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    rows=[]; detailed=[]; diagnostics=[]
    for dataset in ['Credit', 'COMPAS']:
        for model in ['LR', 'HGB']:
            for seed in range(709, 714):
                tag=f'{dataset}_{model}_{seed}'
                rawpath=DATA/'scores'/f'expanded_{tag}.npz'
                paths=[rawpath, BASE/(tag+'.json'), BASE/(tag+'.npz')]
                raw=np.load(rawpath); info=json.loads(paths[1].read_text()); base=np.load(paths[2])
                fit=info['fit']['EG']; th=np.asarray(fit['thresholds']); w0=np.asarray(fit['weights'])
                shape=tuple(len(x) for x in info['meta']['levels'][:2])
                H={}
                for p in PARTS:
                    cells=np.ravel_multi_index(raw[p+'_groups'][:, :2].T, shape)
                    H[p]=(raw[p+'_score'][:, None] >= th[:, cells].T).astype(float)
                    assert np.max(np.abs(H[p] @ w0-base['EG__'+p])) < 1e-8
                groups=group_definitions(raw, info['meta']); coeff=coefficients(H, raw, base, groups)
                saved={}
                for method, validation, trust in [('Reweight-only', False, False),
                        ('Validation-only', True, False), ('Trust-only', False, True),
                        ('Retain-EG', True, True), ('Exact-match', False, False)]:
                    w, diag=solve(coeff, w0, validation, trust, exact=method=='Exact-match')
                    key=dict(dataset=dataset, model=model, seed=seed, method=method)
                    diagnostics.append(dict(**key, **diag))
                    for p in PARTS:
                        q=np.clip(H[p] @ w,0,1); y=raw[p+'_y']; baseline=base['Original__'+p]
                        saved[method+'__'+p]=q
                        stats=evaluate(q,y,baseline,np.ones(len(y),bool)); family_rates={}; inter=[]
                        for family,name,masks in groups:
                            st=evaluate(q,y,baseline,masks[p]);family_rates.setdefault(family,[]).append(st['tpr'])
                            detailed.append(dict(**key,partition=p,family=family,group=name,reported=True,**st))
                            if family=='intersection':inter.append(st)
                        stats.update({'gap_'+f:float(np.ptp(rr)) for f,rr in family_rates.items()})
                        stats.update(min_group_accuracy=min(x['accuracy'] for x in inter),
                                     max_group_withdrawal=max(x['withdrawal'] for x in inter),
                                     reported_intersections=len(inter))
                        rows.append(dict(**key,partition=p,**stats))
                np.savez_compressed(CACHE/(tag+'.npz'),**saved)
                print(tag, 'verified and reweighted', flush=True)
    result=pd.DataFrame(rows);result.to_csv(OUT/'retention_metrics.csv',index=False)
    pd.DataFrame(detailed).to_csv(OUT/'retention_groups.csv',index=False)
    (OUT/'retention_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    combined=pd.concat([pd.read_csv(OUT/'baseline_metrics.csv'),result],ignore_index=True)
    means=combined.groupby(['dataset','partition','method']).mean(numeric_only=True)
    print((100*means.xs('test',level='partition')[['gap_intersection','accuracy','withdrawal','min_group_accuracy','max_group_withdrawal']]).round(3).to_string())


if __name__=='__main__':
    run()
