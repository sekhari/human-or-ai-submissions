"""Secondary diagnostics; do not use secondary splits for model selection."""
import argparse
import json
import re
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from features import normalize, style_features, style_feature_names, vectorizer
from train import ROOT, classifier, probability


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
    df=pd.read_csv(args.data/'train.csv');y=df.label.to_numpy();xs=style_features(df.text)
    selection=json.loads((ROOT/'results/selection.json').read_text())
    if 'previous_selection' in selection:
        selection['selected']=selection['previous_selection']
        selection['best']=selection['previous_best']
    records=json.loads((ROOT/'results/experiments.json').read_text())
    config={k:next(r['config'] for r in records if r['name']==selection['best'][k]) for k in selection['best']}
    results=[]
    # The selected settings and weights are frozen across secondary splits.
    for seed in [20261024,20261025,20261026]:
        tr,va=train_test_split(np.arange(len(df)),test_size=2000,stratify=y,random_state=seed)
        pv=np.zeros(len(va));individual={};start=time.perf_counter()
        for kind,w in zip(['word','char','style'],selection['selected']['weights']):
            c=config[kind]
            if kind=='style':
                m=LGBMClassifier(n_estimators=c['estimators'],num_leaves=c['leaves'],learning_rate=.03,
                    min_child_samples=40,colsample_bytree=.85,reg_lambda=5,verbosity=-1,n_jobs=4,
                    random_state=20261023,deterministic=True)
                m.fit(xs[tr],y[tr]);p=probability(m,xs[va])
            else:
                v=vectorizer(kind);xt=v.fit_transform(df.text.iloc[tr]);xv=v.transform(df.text.iloc[va])
                m=classifier(c['kind'],c['C']);m.fit(xt,y[tr]);p=probability(m,xv)
            individual[kind]=roc_auc_score(y[va],p);pv+=w*p
        r={'seed':seed,'ensemble_auc':roc_auc_score(y[va],pv),'components':individual,'seconds':time.perf_counter()-start}
        results.append(r);print(r,flush=True)
    cache=joblib.load(ROOT/'artifacts/validation.joblib');tr=cache['train'];va=cache['val']
    m=cache['models'][selection['best']['style']]
    names=style_feature_names();assert len(names)==xs.shape[1]
    gains=m.booster_.feature_importance(importance_type='gain')
    top=sorted(zip(names,gains),key=lambda p:p[1],reverse=True)[:20]
    # Retrain without size and sentence-count features; checks shortcut dependence.
    drops=[i for i,n in enumerate(names) if n in ['log_characters','log_words','sentence_words_mean',
           'sentence_words_std','sentence_words_max','sentences_per_word','lines_per_word','blank_lines_per_word']]
    keep=[i for i in range(len(names)) if i not in drops]
    ablated=LGBMClassifier(**m.get_params());ablated.fit(xs[tr][:,keep],y[tr])
    ablation={'dropped':[names[i] for i in drops],
              'style_baseline_auc':roc_auc_score(y[va],probability(m,xs[va])),
              'ablated_auc':roc_auc_score(y[va],probability(ablated,xs[va][:,keep]))}
    stress=[]
    transforms={
        'collapse_whitespace':lambda t:re.sub(r'\s+',' ',normalize(t)),
        'ascii_punctuation':lambda t:normalize(t).translate(str.maketrans({'“':'"','”':'"','‘':"'",'’':"'",'—':'-','–':'-'})),
        'remove_bracketed_references':lambda t:re.sub(r'\[[^\]]*\]','',normalize(t))}
    for label,transform in transforms.items():
        texts=df.text.iloc[va].map(transform);p=np.zeros(len(va))
        for kind,w in zip(['word','char','style'],selection['selected']['weights']):
            model=cache['models'][selection['best'][kind]]
            xx=style_features(texts) if kind=='style' else cache['matrices'][kind][0].transform(texts)
            if isinstance(model,tuple):model,ratio=model;xx=xx.multiply(ratio)
            p+=w*probability(model,xx)
        stress.append({'transformation':label,'auc':roc_auc_score(y[va],p)})
    out={'secondary_splits':results,'top_style_features':[(n,float(g)) for n,g in top],
         'ablation':ablation,'stress_tests':stress,
         'limitation':'Secondary splits overlap selection data; these are stability diagnostics, not independent external evaluations.'}
    (ROOT/'results/robustness.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
