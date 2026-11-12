"""Rebuild excluded predictions and fit/evaluate the recorded logistic stack."""
import shutil
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import ROOT,read_json,write_json,metrics
from ensembles import families,combine

def fit_coefficients(c,y):
    x=logit(np.clip(c,1e-6,1-1e-6));scaler=StandardScaler().fit(x)
    m=LogisticRegression(C=.1,class_weight='balanced',max_iter=1000).fit(scaler.transform(x),y)
    coef=m.coef_[0]/scaler.scale_
    return dict(logit_coefficients_R_T_Q=coef.tolist(),intercept=float(m.intercept_[0]-coef@scaler.mean_),clip=1e-6,C=.1)

def fit(frame,out,device,cache_dir=None,offline=False,keep_fold_models=False):
    from transformer_models import CONFIG,train_member,predict_member
    byid=frame.set_index('id',drop=False);all_rows=[];comparisons=[]
    plan=read_json(ROOT/'configs/stack_splits.json')
    for split in plan['splits']:
        parts={k:byid.loc[ids].copy() for k,ids in split['parts'].items()}
        sets=[set(d.id) for d in parts.values()]
        assert len(set.union(*sets))==len(frame) and sum(map(len,sets))==len(frame)
        assert all(sets[a].isdisjoint(sets[b]) for a,b in [(0,1),(0,2),(1,2)])
        folder=out/f"fold{split['index']}";folder.mkdir(parents=True)
        combined=pd.concat([parts['meta'],parts['evaluation']],ignore_index=True);pred={}
        for key in CONFIG['members']:
            target=folder/key
            train_member(key,parts['base_train'],target,device,cache_dir,offline,fold=True)
            pred[key]=predict_member(target,combined,device,cache_dir,offline)
            pd.DataFrame(dict(id=combined.id,probability_AI=pred[key])).to_csv(folder/(key+'_excluded.csv'),index=False)
            if not keep_fold_models:
                for sub in ['model','adapter','tokenizer']:
                    p=target/sub
                    if p.exists():
                        assert p.resolve().is_relative_to(out.resolve())
                        shutil.rmtree(p)
                if (target/'head.pt').exists():(target/'head.pt').unlink()
        c=families(pred);n=len(parts['meta']);coef=fit_coefficients(c[:n],parts['meta'].label.to_numpy())
        write_json(folder/'stack.json',coef)
        ev=parts['evaluation'];base=combine(c[n:],'best');stack=combine(c[n:],'stack',coef)
        comparisons.append(dict(fold=split['index'],reference=metrics(ev.label,base),stack=metrics(ev.label,stack)))
        table=pd.DataFrame(dict(id=combined.id,label=combined.label,r=c[:,0],t=c[:,1],q=c[:,2],fold=split['index']))
        table.to_csv(folder/'family_excluded_predictions.csv',index=False);all_rows.append(table)
        write_json(out/'evaluation.json',comparisons)
    pooled=pd.concat(all_rows)
    assert pooled.groupby('id').label.nunique().max()==1
    final=pooled.groupby('id',sort=True).agg(label=('label','first'),r=('r','mean'),t=('t','mean'),q=('q','mean'))
    assert set(final.index)==set(frame.id)
    config=fit_coefficients(final[['r','t','q']].to_numpy(),final.label.to_numpy())
    config['training_unique_IDs']=len(final);config['note']='Post-evaluation refit; no independent score for this final refit.'
    write_json(out/'stack.json',config);final.to_csv(out/'excluded_training_features.csv')
