"""Audit the unusually strong style model without outer-validation labels."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .common import ROOT, OUT, load_protocol
from .style_search import StyleFeatures, FEATURE_NAMES, candidate_configs, build_pipeline


def main():
    dest=OUT/'style_audit';dest.mkdir(parents=True,exist_ok=True)
    development, protocol=load_protocol()
    test=pd.read_csv(ROOT/'data/test.csv')
    extractor=StyleFeatures()
    start=perf_counter()
    X=extractor.transform(development.text)
    T=extractor.transform(test.text)
    config=next(c for c in candidate_configs() if c['name']=='style_hgb31')
    tr=np.asarray(protocol['protocols']['random']['train']);va=np.asarray(protocol['protocols']['random']['validation'])
    model=build_pipeline(config).named_steps['classifier'].fit(X[tr],development.label.iloc[tr])
    baseline=float(roc_auc_score(development.label.iloc[va],model.predict_proba(X[va])[:,1]))
    importance=permutation_importance(model,X[va],development.label.iloc[va],scoring='roc_auc',n_repeats=2,random_state=773,n_jobs=1)
    ranking=[]
    for j,name in enumerate(FEATURE_NAMES):
        one=float(roc_auc_score(development.label.iloc[va],X[va,j]))
        ranking.append(dict(feature=name,auc_drop=float(importance.importances_mean[j]),
                            univariate_separation=max(one,1-one),
                            human_train_mean=float(X[tr,j][development.label.iloc[tr].to_numpy()==0].mean()),
                            ai_train_mean=float(X[tr,j][development.label.iloc[tr].to_numpy()==1].mean()),
                            validation_median=float(np.median(X[va,j])),test_median=float(np.median(T[:,j])),
                            test_outside_training_range=float(((T[:,j]<X[tr,j].min())|(T[:,j]>X[tr,j].max())).mean())))
    ranking=sorted(ranking,key=lambda r:-r['auc_drop'])
    pd.DataFrame(ranking).to_csv(dest/'permutation_importance.csv',index=False)
    shuffled=np.random.default_rng(773).permutation(development.label.iloc[tr].to_numpy())
    null=build_pipeline(config).named_steps['classifier'].fit(X[tr],shuffled)
    null_auc=float(roc_auc_score(development.label.iloc[va],null.predict_proba(X[va])[:,1]))
    report=dict(random_validation_auroc=baseline,label_shuffle_validation_auroc=null_auc,
                top_features=ranking[:15],feature_count=len(FEATURE_NAMES),all_features_finite=bool(np.isfinite(X).all() and np.isfinite(T).all()),
                outer_validation_labels_used=False,test_labels_used=False,
                interpretation='Permutation importance and distribution ranges are descriptive, not proof of causal generalization. Source-format shortcuts can remain even without code leakage.',
                elapsed_seconds=perf_counter()-start)
    (dest/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    np.savez_compressed(dest/'features.npz',development_ids=development.id.to_numpy(),X=X,test_ids=test.id.to_numpy(),T=T)
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
