"""Reproducible Cornell Tech AML baselines. Run: python baseline_models.py --train path/to/train.csv --out results"""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

SEED=42

def pipeline(estimator):
    return Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000,sublinear_tf=True)),('model',estimator)])

def scores(model, texts):
    if hasattr(model,'predict_proba'):
        return model.predict_proba(texts)[:,1]
    return model.decision_function(texts)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',required=True)
    parser.add_argument('--out',default='results')
    parser.add_argument('--tune',action='store_true',help='3-fold CV tuning of logistic regression and LinearSVC')
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.train)
    if df[['id','text','label']].isna().any().any():raise ValueError('Missing id/text/label values')
    if df.id.duplicated().any() or df.text.duplicated().any():raise ValueError('Exact duplicates found; inspect before splitting')
    if set(df.label.unique())!={0,1}:raise ValueError('Expected binary 0/1 labels')
    train_idx,val_idx=train_test_split(np.arange(len(df)),test_size=2000,stratify=df.label,random_state=SEED)
    tr=df.iloc[train_idx]; va=df.iloc[val_idx]
    pd.DataFrame({'id':va.id,'label':va.label}).to_csv(out/'validation_ids.csv',index=False)
    models={
        'Word TF-IDF + Logistic Regression':pipeline(LogisticRegression(C=1.0,max_iter=2000,random_state=SEED)),
        'Word TF-IDF + Linear SVM':pipeline(LinearSVC(C=1.0,random_state=SEED)),
        'Word TF-IDF + MultinomialNB':pipeline(MultinomialNB(alpha=1.0)),
    }
    rows=[]
    for name,model in models.items():
        t=time.perf_counter();model.fit(tr.text,tr.label);duration=time.perf_counter()-t
        sp=scores(model,tr.text);svp=scores(model,va.text)
        rows.append({'model':name,'train_auc':roc_auc_score(tr.label,sp),'validation_auc':roc_auc_score(va.label,svp),'train_seconds':duration})
        pd.DataFrame({'id':va.id,'true_label':va.label,'score':svp}).to_csv(out/(name.split('+')[-1].strip().replace(' ','_')+'_val_scores.csv'),index=False)
        print(name,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(out/'baseline_comparison.csv',index=False)
    if args.tune:
        cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
        tuning=[]
        for name in list(models)[:2]:
            print('Tuning',name,flush=True)
            search=GridSearchCV(models[name],{'model__C':[0.01,0.1,1,10,100]},scoring='roc_auc',cv=cv,n_jobs=1,refit=True)
            t=time.perf_counter();search.fit(tr.text,tr.label);elapsed=time.perf_counter()-t
            tuning.append({'model':name,'best_C':search.best_params_['model__C'],'cv_auc':search.best_score_,'validation_auc':roc_auc_score(va.label,scores(search.best_estimator_,va.text)),'tuning_seconds':elapsed})
            print(tuning[-1],flush=True)
        pd.DataFrame(tuning).to_csv(out/'tuning_comparison.csv',index=False)
    (out/'experiment_config.json').write_text(json.dumps({'seed':SEED,'split':'stratified 8000 train / 2000 validation','tfidf':{'ngram_range':[1,2],'min_df':2,'max_features':100000,'sublinear_tf':True}},indent=2))

if __name__=='__main__':main()
