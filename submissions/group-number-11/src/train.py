"""Controlled experiments and final refit. See README for commands."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from features import normalize, vectorizer, style_features

SEED = 20261023
ROOT = Path(__file__).resolve().parents[1]


def classifier(kind, c):
    if kind == 'svm':
        return LinearSVC(C=c, dual=True, max_iter=4000, random_state=SEED)
    return LogisticRegression(C=c, solver='liblinear', max_iter=1000, random_state=SEED)


def probability(model, x):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(x)[:, 1]
    z = model.decision_function(x)
    return 1/(1+np.exp(-np.clip(z,-35,35)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--final',action='store_true')
    args=ap.parse_args()
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True)
    df=pd.read_csv(args.data/'train.csv')
    unl=pd.read_csv(args.data/'unlabeled.csv')
    y=df.label.to_numpy()
    tr, va=train_test_split(np.arange(len(df)),test_size=2000,stratify=y,random_state=SEED)
    pd.DataFrame({'id':df.id,'partition':np.where(np.isin(np.arange(len(df)),va),'validation','training')}).to_csv(out/'split.csv',index=False)
    hashes=df.text.map(lambda s:hashlib.sha256(normalize(s).encode()).hexdigest())
    audit={'seed':SEED,'rows':len(df),'class_counts':df.label.value_counts().to_dict(),
           'duplicate_texts':int(hashes.duplicated().sum()),
           'cross_split_exact_duplicates':len(set(hashes.iloc[tr])&set(hashes.iloc[va])),
           'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data.glob('*.csv')}}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    records=[]; preds={}; fitted={}; matrices={}
    def record(name, model, xt, xv, elapsed, config):
        pt=probability(model,xt); pv=probability(model,xv)
        r=dict(name=name,train_auc=roc_auc_score(y[tr],pt),val_auc=roc_auc_score(y[va],pv),
               seconds=elapsed,log_loss=log_loss(y[va],pv),config=config)
        records.append(r); preds[name]=pv; fitted[name]=model
        print(json.dumps(r),flush=True)
        (out/'experiments.json').write_text(json.dumps(records,indent=2))
        return r
    for representation in ['word','char']:
        start=time.perf_counter(); v=vectorizer(representation)
        xt=v.fit_transform(df.text.iloc[tr]); xv=v.transform(df.text.iloc[va]); xu=v.transform(unl.text)
        feature_seconds=time.perf_counter()-start
        matrices[representation]=(v,xt,xv,xu)
        for kind in ['lr','svm']:
            for c in [.1,1,4,16]:
                start=time.perf_counter(); m=classifier(kind,c); m.fit(xt,y[tr])
                record(f'{representation}_{kind}_{c}',m,xt,xv,time.perf_counter()-start+feature_seconds,
                       {'representation':representation,'kind':kind,'C':c})
        # NB log-count ratios emphasize class-discriminative ngrams.
        start=time.perf_counter()
        pos=np.asarray(xt[y[tr]==1].sum(axis=0)).ravel()+1
        neg=np.asarray(xt[y[tr]==0].sum(axis=0)).ravel()+1
        ratio=np.log((pos/pos.sum())/(neg/neg.sum()))
        for c in [.1,1,4]:
            start=time.perf_counter(); m=classifier('lr',c)
            m.fit(xt.multiply(ratio),y[tr])
            name=f'{representation}_nb_{c}'
            record(name,m,xt.multiply(ratio),xv.multiply(ratio),time.perf_counter()-start+feature_seconds,
                   {'representation':representation,'kind':'nb','C':c})
            fitted[name]=(m,ratio)
    start=time.perf_counter();xs=style_features(df.text)
    style_feature_seconds=time.perf_counter()-start
    for leaves in [7,15,31]:
        for estimators in [300,700]:
            start=time.perf_counter()
            m=LGBMClassifier(n_estimators=estimators,num_leaves=leaves,learning_rate=.03,
                max_depth=-1,min_child_samples=40,colsample_bytree=.85,reg_lambda=5,
                verbosity=-1,n_jobs=4,random_state=SEED,deterministic=True)
            m.fit(xs[tr],y[tr])
            record(f'style_{leaves}_{estimators}',m,xs[tr],xs[va],time.perf_counter()-start+style_feature_seconds,
                   {'representation':'style','leaves':leaves,'estimators':estimators})
    # Select each family by validation AUC, then compare a small prespecified blend grid.
    best={r:max((a for a in records if a['config']['representation']==r),key=lambda a:a['val_auc'])['name'] for r in ['word','char','style']}
    blend=[]
    for ws in [0,.25,.5,.75,1]:
        for wc in [0,.5,1]:
            weights=np.array([(1-ws)*(1-wc),(1-ws)*wc,ws])
            p=sum(w*preds[best[k]] for w,k in zip(weights,['word','char','style']))
            blend.append({'weights':weights.tolist(),'val_auc':roc_auc_score(y[va],p)})
    chosen=max(blend,key=lambda a:a['val_auc'])
    pv=sum(w*preds[best[k]] for w,k in zip(chosen['weights'],['word','char','style']))
    (out/'selection.json').write_text(json.dumps({'best':best,'blends':blend,'selected':chosen},indent=2))
    prediction_frame=pd.DataFrame({'id':df.id.iloc[va].to_numpy(),'label':y[va],'probability':pv,
        'words':df.text.iloc[va].map(lambda t:len(normalize(t).split())).to_numpy()})
    prediction_frame.to_csv(out/'validation_predictions.csv',index=False)
    print('SELECTED',best,chosen,flush=True)
    # Required learning curve: refit vocabulary on each labeled subset.
    curve=[]
    for n in [500,1000,2000,5000,8000]:
        subset=tr if n==8000 else train_test_split(tr,train_size=n,stratify=y[tr],random_state=SEED)[0]
        start=time.perf_counter(); v=vectorizer('word'); xt=v.fit_transform(df.text.iloc[subset]); xv=v.transform(df.text.iloc[va])
        m=classifier('lr',4);m.fit(xt,y[subset])
        r={'n':n,'train_auc':roc_auc_score(y[subset],probability(m,xt)),
           'val_auc':roc_auc_score(y[va],probability(m,xv)),'seconds':time.perf_counter()-start}
        curve.append(r);print('CURVE',r,flush=True)
    (out/'learning_curve.json').write_text(json.dumps(curve,indent=2))
    # Unsupervised vocabulary/IDF learning: fixed classifier, no validation texts.
    base=next(r for r in records if r['name']=='word_lr_4')
    start=time.perf_counter(); v=vectorizer('word')
    v.fit(pd.concat([df.text.iloc[tr],unl.text]));xt=v.transform(df.text.iloc[tr]);xv=v.transform(df.text.iloc[va])
    m=classifier('lr',4);m.fit(xt,y[tr])
    semi={'method':'Vocabulary and IDF fit on training + unlabeled text; supervised LR C=4',
          'baseline_auc':base['val_auc'],'unlabeled_auc':roc_auc_score(y[va],probability(m,xv)),
          'seconds':time.perf_counter()-start}
    (out/'unlabeled.json').write_text(json.dumps(semi,indent=2));print('UNLABELED',semi,flush=True)
    joblib.dump({'predictions':preds,'models':fitted,'matrices':matrices,'style':xs,'train':tr,'val':va},artifacts/'validation.joblib',compress=3)
    if args.final:
        components=[]
        for kind,weight in zip(['word','char','style'],chosen['weights']):
            if weight==0: continue
            name=best[kind]; config=next(r['config'] for r in records if r['name']==name)
            if kind=='style':
                m=fitted[name];m.fit(xs,y);components.append({'weight':weight,'kind':kind,'model':m})
            else:
                v=vectorizer(kind); xx=v.fit_transform(df.text); ratio=None
                if config['kind']=='nb':
                    pos=np.asarray(xx[y==1].sum(axis=0)).ravel()+1;neg=np.asarray(xx[y==0].sum(axis=0)).ravel()+1
                    ratio=np.log((pos/pos.sum())/(neg/neg.sum()));xx=xx.multiply(ratio)
                m=classifier(config['kind'],config['C']);m.fit(xx,y)
                components.append({'weight':weight,'kind':kind,'model':m,'vectorizer':v,'ratio':ratio})
        joblib.dump(components,artifacts/'final.joblib',compress=3)
        print('FINAL MODEL SAVED',flush=True)


if __name__=='__main__': main()
