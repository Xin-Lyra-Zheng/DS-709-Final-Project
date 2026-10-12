"""Sensitivity checks using the saved EG classifiers."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
from run_retention_eg import PARTS, group_definitions, coefficients, solve, evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('FAIRREPAIR_INPUT', ROOT / 'data'))
BASE = DATA / 'baselines'
RESULTS = Path(os.environ.get('FAIRREPAIR_OUTPUT', ROOT / 'results'))
OUT = RESULTS / 'sensitivity'
RADII = [0., .1, .25, .5, 1., 2.]
TOLERANCES = [0., .005, .01, .02, .05]
UTILITY_BUDGETS = [0., .001, .005, .01, .02]
FRACTIONS = [.25, .5]
REPLICATES = 5


def sampled_coefficients(H, raw, base, groups, selections):
    out = {}; missing = {}; positive_counts = {}
    for part in PARTS:
        ix = selections[part]
        y = raw[part+'_y'][ix]; b = base['Original__'+part][ix]
        hh = H[part][ix]
        assert b.sum() > 0
        acc = (1-y).mean() + (2*y-1) @ hh / len(y)
        withdrawal = 1 - b @ hh / b.sum()
        rate = {}; absent = []; counts = []
        for family, name, masks in groups:
            pos = masks[part][ix] & (y == 1)
            if family == 'intersection': counts.append(int(pos.sum()))
            if pos.sum():
                rate.setdefault(family, []).append(hh[pos].mean(axis=0))
            else:
                absent.append(family+'/'+name)
        diffs = {}
        for family, rr in rate.items():
            if len(rr) > 1:
                diffs[family] = np.asarray([a-b for i,a in enumerate(rr) for j,b in enumerate(rr) if i!=j])
        out[part] = dict(acc=acc, withdrawal=withdrawal, differences=diffs, rates=rate)
        missing[part] = absent; positive_counts[part] = counts
    return out, missing, positive_counts


def configurations():
    for radius in RADII:
        yield dict(experiment='radius', value=radius, replicate=0, epsilon=.02, rho=radius)
    for tol in TOLERANCES:
        yield dict(experiment='tolerance', value=tol, replicate=0, epsilon=tol, rho=.25)
    for budget in UTILITY_BUDGETS:
        yield dict(experiment='utility_budget', value=budget, replicate=0, epsilon=.02, rho=.25, delta=budget)
    for part in ['repair', 'validation']:
        for fraction in FRACTIONS:
            for rep in range(REPLICATES):
                yield dict(experiment=part+'_fraction', value=fraction, replicate=rep, epsilon=.02, rho=.25)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(radii=RADII, tolerances=TOLERANCES, fractions=FRACTIONS,
        replicates=REPLICATES, accuracy_budget=.005, utility_budgets=UTILITY_BUDGETS, subsample_seed_base=20261008,
        sampling='Uniform without replacement; nested fractions with common permutation per fit, partition and replicate.',
        scope='Fixed EG dictionaries learned on full repair data. Subsampling tests mixture estimation, not end-to-end learning data efficiency.',
        groups='Evaluation eligibility fixed from original full repair split. Missing-positive groups omitted from subsample constraints only and counted.',
        selection='All settings reported; original .25 radius and .02 tolerance remain the defaults. Test partitions were previously inspected.')
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows=[]; detail=[]; diagnostics=[]; decomposition=[]; hashes={}
    for ds in ['Credit', 'COMPAS']:
        for model in ['LR', 'HGB']:
            for seed in range(709,714):
                tag=f'{ds}_{model}_{seed}'
                paths=[DATA/'scores'/f'expanded_{tag}.npz', BASE/(tag+'.npz'), BASE/(tag+'.json')]
                raw=np.load(paths[0]); base=np.load(paths[1]); info=json.loads(paths[2].read_text())
                th=np.asarray(info['fit']['EG']['thresholds']); w0=np.asarray(info['fit']['EG']['weights'])
                shape=tuple(len(x) for x in info['meta']['levels'][:2])
                H={part:(raw[part+'_score'][:,None]>=th[:,np.ravel_multi_index(raw[part+'_groups'][:,:2].T,shape)].T).astype(float) for part in PARTS}
                assert all(np.max(np.abs(H[part]@w0-base['EG__'+part]))<1e-8 for part in PARTS)
                groups=group_definitions(raw,info['meta']); selections={}; saved_weights={}
                for case_id, cfg in enumerate(configurations()):
                    ix={part:np.arange(len(raw[part+'_y'])) for part in PARTS}
                    if cfg['experiment'].endswith('_fraction'):
                        part=cfg['experiment'].split('_')[0]
                        rng=np.random.default_rng(20261008+seed*100+cfg['replicate']+(0 if part=='repair' else 10000))
                        ix[part]=np.sort(rng.permutation(len(ix[part]))[:int(len(ix[part])*cfg['value'])])
                    coeff, missing, counts=sampled_coefficients(H,raw,base,groups,ix)
                    w,diag=solve(coeff,w0,True,True,epsilon=cfg['epsilon'],rho=cfg['rho'],delta=cfg.get('delta',.005))
                    key=dict(dataset=ds,model=model,seed=seed,case_id=case_id,**cfg)
                    diagnostics.append(dict(**key,**diag,missing=missing,positive_counts=counts))
                    saved_weights[str(case_id)]=w
                    for part in PARTS:
                        selections[f'{case_id}__{part}']=ix[part]
                        q=np.clip(H[part]@w,0,1); y=raw[part+'_y']; b=base['Original__'+part]
                        st=evaluate(q,y,b,np.ones(len(y),bool)); rr={}; inter=[]
                        for fam,name,masks in groups:
                            gg=evaluate(q,y,b,masks[part]); rr.setdefault(fam,[]).append(gg['tpr'])
                            if fam=='intersection':
                                inter.append(gg)
                                if part=='test':detail.append(dict(**key,group=name,**gg))
                        st.update({'gap_'+f:float(np.ptp(v)) for f,v in rr.items()})
                        st.update(min_group_accuracy=min(x['accuracy'] for x in inter),max_group_withdrawal=max(x['withdrawal'] for x in inter))
                        rows.append(dict(**key,partition=part,**st))
                np.savez_compressed(OUT/(tag+'_weights.npz'),**saved_weights)
                np.savez_compressed(OUT/(tag+'_indices.npz'),**selections)
                ext=np.load(RESULTS/'cache'/(tag+'.npz'))
                for method in ['Original','Hardt','Morina','EG','Retain-EG']:
                    q=ext[method+'__test'] if method=='Retain-EG' else base[method+'__test']
                    y=raw['test_y']; b=base['Original__test']
                    audit_groups=[('overall',np.ones(len(y),bool))]+[(name,masks['test']) for fam,name,masks in groups if fam=='intersection']
                    for name,mask in audit_groups:
                        m1=mask&(b==1)&(y==1);m0=mask&(b==1)&(y==0)
                        row=dict(dataset=ds,model=model,seed=seed,method=method,group=name,**evaluate(q,y,b,mask))
                        row.update(withdraw_true_positive=float((1-q[m1]).sum()),withdraw_false_positive=float((1-q[m0]).sum()),
                            base_true_positive=int(m1.sum()),base_false_positive=int(m0.sum()),
                            new_true_positive=float(q[mask&(b==0)&(y==1)].sum()),new_false_positive=float(q[mask&(b==0)&(y==0)].sum()))
                        decomposition.append(row)
                for path in paths:hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                print(tag, '36 sensitivity settings finished',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(detail).to_csv(OUT/'group_metrics.csv',index=False)
    pd.DataFrame(decomposition).to_csv(OUT/'withdrawal_decomposition.csv',index=False)
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    (OUT/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    summary=pd.DataFrame(rows).groupby(['dataset','partition','experiment','value']).mean(numeric_only=True)
    summary.to_csv(OUT/'means.csv')
    print((100*summary.xs('test',level='partition')[['gap_intersection','accuracy','withdrawal']]).round(3).to_string())


if __name__=='__main__':
    run()
