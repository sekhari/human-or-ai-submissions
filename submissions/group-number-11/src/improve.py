"""Test whitespace augmentation without touching test labels or model APIs.

Selection rule fixed before execution: consider the augmented replacement if its
original-text validation AUC is within 0.0005 of the previous blend and its
whitespace-collapsed AUC is >= 0.98. Otherwise retain the original selection.
Among eligible augmented candidates choose the highest original-text AUC.
"""
import argparse
import json
import re
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from features import normalize, style_features
from train import ROOT, SEED, probability


def collapse(t):
    return re.sub(r'\s+', ' ', normalize(t)).strip()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--final',action='store_true');args=ap.parse_args()
    df=pd.read_csv(args.data/'train.csv');y=df.label.to_numpy()
    cache=joblib.load(ROOT/'artifacts/validation.joblib');tr=cache['train'];va=cache['val']
    selection=json.loads((ROOT/'results/selection.json').read_text())
    if 'previous_selection' in selection:
        selection['selected']=selection.pop('previous_selection')
        selection['best']=selection.pop('previous_best')
    orig=cache['style'];start=time.perf_counter();flat=style_features(df.text.map(collapse));feature_seconds=time.perf_counter()-start
    yy=np.concatenate([y[tr],y[tr]])
    xx=np.concatenate([orig[tr],flat[tr]])
    # Original word/character components stay fixed to isolate style augmentation.
    lexical_original=np.zeros(len(va));lexical_flat=np.zeros(len(va));lexical_train=np.zeros(len(tr))
    for kind in ['word','char']:
        m=cache['models'][selection['best'][kind]]
        v,xt,xv,_=cache['matrices'][kind];xf=v.transform(df.text.iloc[va].map(collapse))
        if isinstance(m,tuple):m,ratio=m;xt=xt.multiply(ratio);xv=xv.multiply(ratio);xf=xf.multiply(ratio)
        lexical_original+=.5*probability(m,xv);lexical_flat+=.5*probability(m,xf);lexical_train+=.5*probability(m,xt)
    baseline=selection['selected']['val_auc'];records=[];models={}
    for leaves in [15,31]:
        start=time.perf_counter()
        m=LGBMClassifier(n_estimators=700,num_leaves=leaves,learning_rate=.03,
            min_child_samples=40,colsample_bytree=.85,reg_lambda=5,verbosity=-1,n_jobs=4,
            random_state=SEED,deterministic=True)
        m.fit(xx,yy);elapsed=time.perf_counter()-start+feature_seconds;models[leaves]=m
        for w in [.75,1.0]:
            po=w*probability(m,orig[va])+(1-w)*lexical_original
            pf=w*probability(m,flat[va])+(1-w)*lexical_flat
            pt=w*probability(m,orig[tr])+(1-w)*lexical_train
            auc=roc_auc_score(y[va],po);stress=roc_auc_score(y[va],pf)
            r={'leaves':leaves,'style_weight':w,'val_auc':auc,'collapsed_auc':stress,
               'train_auc':roc_auc_score(y[tr],pt),'seconds':elapsed,
               'eligible':bool(auc>=baseline-.0005 and stress>=.98)}
            records.append(r);print(r,flush=True)
    eligible=[r for r in records if r['eligible']]
    chosen=max(eligible,key=lambda r:r['val_auc']) if eligible else None
    result={'hypothesis':'Whitespace augmentation reduces dependence on paragraph formatting.',
        'selection_rule':'Original AUC within 0.0005 of baseline; collapsed AUC >= 0.98; then maximize original AUC.',
        'original_baseline_auc':baseline,'candidates':records,'selected':chosen}
    (ROOT/'results/improvement.json').write_text(json.dumps(result,indent=2))
    if chosen is None:return
    style=models[chosen['leaves']];weight=chosen['style_weight']
    # Preserve earlier experiment records and cache; add the selected augmented model.
    name=f'augmented_style_{chosen["leaves"]}'
    cache['models'][name]=style
    selection['previous_selection']=selection['selected'];selection['previous_best']=selection['best'].copy()
    selection['best']['style']=name
    selection['selected']={'weights':[(1-weight)/2,(1-weight)/2,weight],'val_auc':chosen['val_auc'],
        'reason':'Prespecified whitespace robustness criterion; see improvement.json.'}
    (ROOT/'results/selection.json').write_text(json.dumps(selection,indent=2))
    p=weight*probability(style,orig[va])+(1-weight)*lexical_original
    pd.DataFrame({'id':df.id.iloc[va].to_numpy(),'label':y[va],'probability':p,
        'words':df.text.iloc[va].map(lambda t:len(normalize(t).split())).to_numpy()}).to_csv(ROOT/'results/validation_predictions.csv',index=False)
    joblib.dump(cache,ROOT/'artifacts/validation.joblib',compress=3)
    if args.final:
        final=joblib.load(ROOT/'artifacts/final.joblib')
        style.fit(np.concatenate([orig,flat]),np.concatenate([y,y]))
        components=[]
        for component in final:
            if component['kind']=='style':component['model']=style;component['weight']=weight
            else:component['weight']=(1-weight)/2
            if component['weight']>0:components.append(component)
        joblib.dump(components,ROOT/'artifacts/final.joblib',compress=3)
        print('Augmented final model saved',flush=True)


if __name__=='__main__':main()
