"""Fixed-holdout search with label-preserving format stress evaluation.

Unlike the earlier augmentation experiment, both classes receive synthetic
paragraphs. Baseline artifacts/results remain unchanged. New model candidates
are evaluated by native and format-shift AUROC, not hidden Kaggle labels.
"""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from advanced_features import expanded_features, canonical, reflow, load_lm_matrix
from train import ROOT,SEED


def model_for(config):
    kind=config['kind']
    if kind=='cat':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=config.get('trees',1200),depth=config.get('depth',5),
            learning_rate=.035,l2_leaf_reg=5,loss_function='Logloss',thread_count=3,
            random_seed=SEED,verbose=False,allow_writing_files=False)
    if kind=='lgb':
        return LGBMClassifier(n_estimators=config.get('trees',700),num_leaves=config.get('leaves',31),
            learning_rate=.03,min_child_samples=config.get('min_child',40),
            colsample_bytree=.85,reg_lambda=config.get('l2',5),verbosity=-1,n_jobs=4,
            random_state=SEED,deterministic=True)
    if kind=='extra':return ExtraTreesClassifier(n_estimators=400,min_samples_leaf=3,max_features=.75,n_jobs=4,random_state=SEED)
    return HistGradientBoostingClassifier(max_iter=500,max_leaf_nodes=15,l2_regularization=5,random_state=SEED)


def get_features(frame, path):
    if path.exists():return joblib.load(path)
    texts=frame.text.tolist();variants={}
    for name,ts in [('raw',texts),('flat',[canonical(t) for t in texts]),
                    ('single',[reflow(t,'\n') for t in texts]),('double',[reflow(t) for t in texts])]:
        start=time.perf_counter();variants[name]=expanded_features(ts)
        print('Features',name,variants[name].shape,time.perf_counter()-start,flush=True)
    joblib.dump(variants,path,compress=1);return variants


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--lm',action='store_true');ap.add_argument('--include-extra',action='store_true')
    ap.add_argument('--lm-dirs',nargs='+',default=['lm']);ap.add_argument('--name');args=ap.parse_args()
    out=ROOT/'results/optimization';art=ROOT/'artifacts/optimization';out.mkdir(exist_ok=True);art.mkdir(exist_ok=True)
    df=pd.read_csv(args.data/'train.csv');y=df.label.to_numpy()
    tr,va=train_test_split(np.arange(len(df)),test_size=2000,stratify=y,random_state=SEED)
    features=get_features(df,art/'rich_train.joblib')
    configs=[]
    for view in ['native','canonical','augmented']:
        for leaves in [7,15,31]:configs.append({'view':view,'kind':'lgb','leaves':leaves})
        if args.include_extra:configs.append({'view':view,'kind':'extra'})
        configs.append({'view':view,'kind':'hist'})
    lm=None
    if args.lm:
        lm=load_lm_matrix(art,'train',args.lm_dirs);assert len(lm)==len(df)
        configs=[]
        for view in ['native','canonical','augmented']:
            for leaves in [7,15,31]:configs.append({'view':view,'kind':'lgb','leaves':leaves,'lm':True,'lm_dirs':args.lm_dirs})
    results=[];predictions={};models={}
    if args.lm and (art/'search.joblib').exists():
        initial=joblib.load(art/'search.joblib');results=initial['records'];predictions=initial['predictions'];models=initial['models']
    variants=['raw','flat','single','double']
    for config in configs:
        name='_'.join(str(config.get(k,'')) for k in ['view','kind','leaves','lm']).rstrip('_')
        if config['view']=='canonical':
            matrices={v:features['flat'] for v in variants}
        else:matrices=features.copy()
        if config.get('lm'):
            # Frozen LM features are retained for format changes. This view tests the
            # style branch's sensitivity; separate recomputed LM stress is checked later.
            matrices={v:np.column_stack([x,lm]) for v,x in matrices.items()}
        views=variants if config['view']=='augmented' else ['flat' if config['view']=='canonical' else 'raw']
        xt=np.concatenate([matrices[v][tr] for v in views]);yt=np.tile(y[tr],len(views))
        start=time.perf_counter();model=model_for(config);model.fit(xt,yt)
        pred={v:model.predict_proba(matrices[v][va])[:,1] for v in variants}
        metrics={v:roc_auc_score(y[va],p) for v,p in pred.items()}
        record={'name':name,'config':config,'aucs':metrics,'seconds':time.perf_counter()-start,
                'native_train_auc':roc_auc_score(y[tr],model.predict_proba(matrices['raw'][tr])[:,1]),
                'robust_mean':float(np.mean(list(metrics.values()))),'worst_format_auc':float(min(metrics.values()))}
        print(json.dumps(record),flush=True);results.append(record);predictions[name]=pred;models[name]=model
        (out/((args.name or ('search_lm' if args.lm else 'search'))+'.json')).write_text(json.dumps(results,indent=2))
    joblib.dump({'records':results,'predictions':predictions,'models':models,'tr':tr,'va':va},
        art/((args.name or ('search_lm' if args.lm else 'search'))+'.joblib'),compress=3)


if __name__=='__main__':main()
