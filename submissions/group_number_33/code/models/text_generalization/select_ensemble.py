"""Freeze a coarse ensemble choice using three internal protocols only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from .common import OUT, sha256


def main():
    destination=OUT/'selection';destination.mkdir(parents=True,exist_ok=True)
    pools={}
    for family,names,layout_folder in [
        ('style',['style_hgb31'],'layout_stress'),
        ('style',['style_hgb15','style_extra_leaf2'],'layout_style_variants'),
        ('group_robust',None,'group_robust'),
    ]:
        results=json.loads((OUT/family/'results.json').read_text())
        for record in results:
            name=record['candidate']
            if names is not None and name not in names:continue
            standard=np.load(OUT/family/f'{name}_predictions.npz')
            layout_path=OUT/layout_folder/f'{name}_layout_predictions.npz'
            if not layout_path.exists():layout_path=OUT/layout_folder/f'{name}_predictions.npz'
            layout=np.load(layout_path)
            values={}
            for protocol,data in [('random',standard),('topic',standard),('layout',layout)]:
                values[protocol]={suffix:data[f'{protocol}_{suffix}'] for suffix in ['ids','labels','scores','z']}
            pools[name]=dict(family=family,config=record['config'],values=values)
    base=pools['style_hgb31']['values']
    for model in pools.values():
        for protocol in base:
            np.testing.assert_array_equal(model['values'][protocol]['ids'],base[protocol]['ids'])
            np.testing.assert_array_equal(model['values'][protocol]['labels'],base[protocol]['labels'])
    candidate_weights=[{name:1.} for name in pools]
    anchor='style_extra_leaf2'
    for name in pools:
        if name==anchor:continue
        for weight in [.25,.5,.75]:candidate_weights.append({anchor:weight,name:1-weight})
    group_names=[name for name in pools if pools[name]['family']=='group_robust']
    for name in group_names:
        candidate_weights.append({anchor:.5,'style_hgb15':.25,name:.25})
    plan=dict(criterion='Maximize mean AUROC across random/topic/layout subject to ordinary(random/topic) mean >= best ordinary standalone minus0.005; within0.0005 of best prefer fewer components then probability averaging',
              candidates=candidate_weights,aggregations=['probability','scaled_logit'],
              ordinary_tolerance=.005,near_tie=.0005,
              note='Layout diagnostic was added after observing format dependence; all three validation sets lie within original8000 and overlap. No outer2000 labels or Kaggle labels enter this selection.')
    (destination/'frozen_blend_plan.json').write_text(json.dumps(plan,indent=2))
    best_ordinary=max(np.mean([roc_auc_score(base[p]['labels'],v['values'][p]['scores']) for p in ['random','topic']]) for v in pools.values())
    records=[]
    for weights in candidate_weights:
        for mode in (['probability'] if len(weights)==1 else ['probability','scaled_logit']):
            record=dict(weights=weights,aggregation=mode,components=len(weights))
            for protocol in base:
                score=sum(w*(expit(pools[name]['values'][protocol]['scores']) if mode=='probability'
                             else pools[name]['values'][protocol]['z']) for name,w in weights.items())
                record[f'{protocol}_auroc']=float(roc_auc_score(base[protocol]['labels'],score))
            record['ordinary_mean']=float(np.mean([record['random_auroc'],record['topic_auroc']]))
            record['three_protocol_mean']=float(np.mean([record[f'{p}_auroc'] for p in base]))
            record['eligible']=bool(record['ordinary_mean']>=best_ordinary-.005)
            records.append(record)
    eligible=[r for r in records if r['eligible']]
    best=max(r['three_protocol_mean'] for r in eligible)
    near=[r for r in eligible if r['three_protocol_mean']>=best-.0005]
    selected=min(near,key=lambda r:(r['components'],r['aggregation']!='probability',-r['three_protocol_mean']))
    config=dict(aggregation=selected['aggregation'],components=[dict(family=pools[name]['family'],config=pools[name]['config'],weight=weight)
                                                               for name,weight in selected['weights'].items()])
    report=dict(selected=selected,model_config=config,criterion=plan['criterion'],candidate_count=len(records),
                outer_evaluation_not_used_for_selection=True,
                protocol_sha256=sha256(OUT/'protocol.json'),layout_plan_sha256=sha256(OUT/'layout_stress/frozen_plan.json'),
                candidate_records=records,
                limitations='Stress protocols are development diagnostics, not an unbiased final-test guarantee. No selection uses Kaggle test labels.')
    (destination/'selected.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    pd.DataFrame(records).to_csv(destination/'blend_scores.csv',index=False)
    print(json.dumps({k:v for k,v in report.items() if k!='candidate_records'},indent=2),flush=True)


if __name__=='__main__':main()
