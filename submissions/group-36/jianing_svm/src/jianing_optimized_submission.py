"""Reproduce the latest selected optimized pipeline and create a separate Kaggle CSV.
Run: python src/jianing_optimized_submission.py
Requires the experiment notebook's full-text embedding cache.
"""
from datetime import datetime,timezone
import json, hashlib, warnings
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from jianing_submission import ROOT,EMB_COLS,prepare_data,test_embeddings

class OptimizedFeatures(TransformerMixin,BaseEstimator):
    def __init__(self,lexical_specs,lexical_names,embedding_weight):
        self.lexical_specs=lexical_specs
        self.lexical_names=lexical_names
        self.embedding_weight=embedding_weight
    def fit(self,X,y=None):
        self.extractors_=[]
        for name in self.lexical_names:
            args=dict(self.lexical_specs[name]);args['ngram_range']=tuple(args['ngram_range'])
            extractor=TfidfVectorizer(**args,dtype=np.float64)
            extractor.fit(X['text'])
            self.extractors_.append(extractor)
        return self
    def transform(self,X):
        blocks=[e.transform(X['text']) for e in self.extractors_]
        if self.embedding_weight>0:
            blocks.append(sp.csr_matrix(X[EMB_COLS].to_numpy(dtype=np.float32))*self.embedding_weight)
        return normalize(sp.hstack(blocks,format='csr'),norm='l2',copy=False)

def train_model(X,y,chosen,plan):
    features=OptimizedFeatures(plan['lexical_specs'],chosen['lexical'],chosen['emb_weight'])
    pipeline=Pipeline([('features',features),('svm',LinearSVC(C=chosen['C'],dual='auto',
        max_iter=plan['svm_max_iter'],tol=plan['svm_tol'],random_state=plan['seed']))])
    model=CalibratedClassifierCV(pipeline,method='sigmoid',ensemble=False,n_jobs=1,
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=plan['seed']))
    with threadpool_limits(limits=4),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(X,y)
    assert model.classes_.tolist()==[0,1]
    return model

def main():
    source_id=(ROOT/'results/jianing_optimization/latest_run.txt').read_text().strip()
    source=ROOT/'results/jianing_optimization'/source_id
    selection=json.loads((source/'selected_config.json').read_text())
    chosen,plan=selection['selected'],selection['search_plan']
    assert chosen=={'name':'WordChar_emb025','lexical':['Word12','Char35'],'emb_weight':0.25,'C':1.0}, 'Selection changed; review the submission configuration.'
    train,test,sample,X,split,spec=prepare_data()
    assert plan['data_sha256']==spec['data_sha256']
    assert plan['embedding_config']==spec['MiniLM_chunked']
    split_path=ROOT/'data/splits/seed42_stratified_8000_2000.csv'
    assert hashlib.sha256(split_path.read_bytes()).hexdigest()==plan['split_sha256']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'results/jianing_optimized_submission'/stamp;out.mkdir(parents=True,exist_ok=False)
    print('Selected source run:',source_id,flush=True)
    print('Checking sigmoid calibration on the fixed validation split (5 internal folds)...',flush=True)
    byid=X.set_index('id',drop=False);labels=train.set_index('id').label
    train_ids=split.loc[split.partition=='train','id'];val_ids=split.loc[split.partition=='validation','id']
    model=train_model(byid.loc[train_ids],labels.loc[train_ids],chosen,plan)
    prob=model.predict_proba(byid.loc[val_ids])[:,1]
    auc=roc_auc_score(labels.loc[val_ids],prob)
    assert abs(auc-selection['validation_AUROC'])<1e-6, 'Validation ranking changed; inspect before submission.'
    metrics={'AUROC':auc,'log_loss':log_loss(labels.loc[val_ids],prob),
        'Brier':brier_score_loss(labels.loc[val_ids],prob),'rows':len(val_ids)}
    pd.DataFrame({'id':val_ids.to_numpy(),'true_label':labels.loc[val_ids].to_numpy(),
                  'probability_AI':prob}).to_csv(out/'validation_probabilities.csv',index=False)
    (out/'validation_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Validation metrics:',metrics,flush=True)
    del model
    emb=test_embeddings(test,plan['embedding_config'])
    assert emb.shape==(3000,384) and np.isfinite(emb).all()
    Xt=pd.concat([test[['id','text']].reset_index(drop=True),pd.DataFrame(emb,columns=EMB_COLS)],axis=1)
    print('Refitting optimized calibrated pipeline on all 10,000 labeled articles...',flush=True)
    model=train_model(X,train.label,chosen,plan)
    prob=model.predict_proba(Xt)[:,1]
    keyed=pd.Series(prob,index=test.id)
    submission=pd.DataFrame({'id':sample.id,'label':keyed.loc[sample.id].to_numpy()})
    assert submission.shape==(3000,2) and submission.columns.tolist()==['id','label']
    assert submission.id.is_unique and set(submission.id)==set(test.id)
    assert np.isfinite(submission.label).all() and submission.label.between(0,1).all()
    assert submission.label.nunique()>1
    path=ROOT/'submissions'/f'submission_Jianing_WordChar_MiniLM025_SVM_C1_{stamp}.csv'
    assert not path.exists()
    submission.to_csv(path,index=False,float_format='%.12g')
    reread=pd.read_csv(path)
    assert reread.id.tolist()==sample.id.tolist() and reread.label.between(0,1).all()
    extractor=model.calibrated_classifiers_[0].estimator.named_steps['features']
    feature_counts=[len(e.vocabulary_) for e in extractor.extractors_]+[384]
    receipt={'source_optimization_run':source_id,'selected_configuration':selection,
        'calibration':'sigmoid, 5-fold stratified shuffled CV, ensemble=False; vocabularies and IDFs fit inside each fold',
        'seed':plan['seed'],'full_training_rows':len(train),'test_rows':len(test),'validation':metrics,
        'full_training_feature_counts':dict(zip(chosen['lexical']+['MiniLM'],feature_counts)),
        'probability_min':float(prob.min()),'probability_max':float(prob.max()),
        'submission_file':path.name,'submission_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'test_sha256':hashlib.sha256((ROOT/'data/raw/test.csv').read_bytes()).hexdigest(),
        'versions':{k:metadata.version(k) for k in ['numpy','pandas','scikit-learn','scipy','sentence-transformers']},
        'note':'Validation AUROC is not a Kaggle score. Test labels and sample placeholder labels were never used.'}
    (out/'submission_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('Verified submission:',path,flush=True)
    print(submission.head().to_string(index=False),flush=True)
    print('P(AI) range:',prob.min(),prob.max(),flush=True)
if __name__=='__main__':
    main()
