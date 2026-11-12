"""Generate P(AI) using the selected TF + chunked MiniLM + C=1 SVM.
Run from the project root: python src/jianing_submission.py
"""
from pathlib import Path
import os, json, hashlib, time, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
FEATURES = ROOT / 'data/processed/jianing_svm'
SEED = 42
EMB_COLS = [f'emb_{i}' for i in range(384)]

class FusionTransformer(TransformerMixin, BaseEstimator):
    """Vocabulary is fitted inside each calibration fold, never globally."""
    def fit(self, X, y=None):
        self.vectorizer_ = CountVectorizer(lowercase=True, min_df=5, max_df=1.0,
                                           token_pattern=r'(?u)\b\w\w+\b', ngram_range=(1,1))
        self.vectorizer_.fit(X['text'])
        return self
    def transform(self, X):
        tf = normalize(self.vectorizer_.transform(X['text']), norm='l2')
        emb = X[EMB_COLS].to_numpy(dtype=np.float32)
        return normalize(sp.hstack([tf,sp.csr_matrix(emb)],format='csr'),norm='l2')

def classifier():
    pipeline = Pipeline([('features',FusionTransformer()),
        ('svm',LinearSVC(C=1.0,dual='auto',max_iter=50000,tol=1e-5,random_state=SEED))])
    # Out-of-fold scores train the sigmoid; one base pipeline is then refitted
    # on all supplied training rows. No ensemble or validation-label fitting.
    return CalibratedClassifierCV(pipeline,method='sigmoid',
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED),ensemble=False,n_jobs=1)

def fit_checked(X,y):
    m=classifier()
    with threadpool_limits(limits=4),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        m.fit(X,y)
    assert m.classes_.tolist()==[0,1]
    return m

def prepare_data():
    train=pd.read_csv(ROOT/'data/raw/train.csv')
    test=pd.read_csv(ROOT/'data/raw/test.csv')
    sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv')
    assert len(train)==10000 and len(test)==len(sample)==3000
    assert train.id.is_unique and test.id.is_unique and sample.id.is_unique
    assert set(test.id)==set(sample.id) and set(sample.columns)=={'id','label'}
    assert not train.isna().any().any() and not test.isna().any().any()
    spec=json.loads((FEATURES/'feature_config.json').read_text())
    data_hash=hashlib.sha256((ROOT/'data/raw/train.csv').read_bytes()).hexdigest()
    assert spec['data_sha256']==data_hash, 'Data changed; rerun the experiment notebook first.'
    expected_tf={'lowercase':True,'min_df':5,'max_df':1.0,'token_pattern':r'(?u)\b\w\w+\b','ngram_range':[1,1],'norm':'l2'}
    assert spec['TF']==expected_tf
    assert spec['TF_MiniLM_fused']['TF_weight']==spec['TF_MiniLM_fused']['MiniLM_weight']==1
    cache=json.loads((FEATURES/'minilm_cache.json').read_text())
    assert cache['config']==spec['MiniLM_chunked']
    blocks=[]
    for part in ['train','validation']:
        ids=pd.read_csv(FEATURES/f'{part}_rows.csv')
        emb=np.load(FEATURES/f'minilm_{part}.npy',allow_pickle=False)
        assert emb.shape==(len(ids),384) and np.isfinite(emb).all()
        assert np.array_equal(train.set_index('id').loc[ids.id,'label'].to_numpy(),ids.label.to_numpy())
        blocks.append(pd.DataFrame(emb,index=ids.id,columns=EMB_COLS))
    embeddings=pd.concat(blocks)
    assert embeddings.index.is_unique and set(embeddings.index)==set(train.id)
    X=train[['id','text']].set_index('id').join(embeddings).reset_index()
    assert X.id.tolist()==train.id.tolist() and not X.isna().any().any()
    split=pd.read_csv(ROOT/'data/splits/seed42_stratified_8000_2000.csv')
    return train,test,sample,X,split,spec

def validate(train,X,split):
    idx=X.set_index('id',drop=False)
    a=split.loc[split.partition=='train','id'];b=split.loc[split.partition=='validation','id']
    y=train.set_index('id').label
    print('Checking calibrated pipeline on the fixed 8,000 / 2,000 split...',flush=True)
    model=fit_checked(idx.loc[a],y.loc[a])
    prob=model.predict_proba(idx.loc[b])[:,1]
    metrics={'validation_rows':len(b),'AUROC':roc_auc_score(y.loc[b],prob),
             'log_loss':log_loss(y.loc[b],prob),'Brier':brier_score_loss(y.loc[b],prob)}
    print(metrics,flush=True)
    out=ROOT/'results/jianing_submission';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'id':b.to_numpy(),'label':y.loc[b].to_numpy(),'probability_AI':prob}).to_csv(out/'validation_probabilities.csv',index=False)
    (out/'validation_metrics.json').write_text(json.dumps(metrics,indent=2))
    return metrics

def test_embeddings(test,cfg):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4);torch.manual_seed(SEED)
    key=hashlib.sha256(json.dumps({'test_sha256':hashlib.sha256((ROOT/'data/raw/test.csv').read_bytes()).hexdigest(),'config':cfg},sort_keys=True).encode()).hexdigest()
    meta=FEATURES/'submission_test_embedding_meta.json';array=FEATURES/'submission_test_embeddings.npy'
    if meta.exists() and array.exists() and json.loads(meta.read_text())['key']==key:
        return np.load(array,allow_pickle=False)
    encoder=SentenceTransformer(cfg['model'],revision=cfg['revision'],device='cpu',cache_folder=str(ROOT/'.cache/huggingface/hub'))
    encoder.max_seq_length=cfg['max_sequence_length'];encoder.eval();t=encoder.tokenizer
    capacity=encoder.max_seq_length-t.num_special_tokens_to_add(pair=False);assert capacity==254
    tokens=t(test.text.tolist(),add_special_tokens=False,truncation=False,padding=False,verbose=False)['input_ids']
    chunks=[];owners=[];weights=[];coverage=[]
    for i,ids in enumerate(tokens):
        assert len(ids)>0
        pieces=[ids[j:j+capacity] for j in range(0,len(ids),capacity)]
        assert sum(pieces,[])==ids
        coverage.append({'id':int(test.id.iloc[i]),'content_tokens':len(ids),'chunks':len(pieces),'dropped_tokens':0})
        for piece in pieces:
            seq=[t.cls_token_id]+piece+[t.sep_token_id]
            assert len(seq)<=encoder.max_seq_length
            chunks.append({'input_ids':seq,'attention_mask':[1]*len(seq)})
            owners.append(i);weights.append(len(piece))
    print(f'Encoding all test text: {len(chunks)} chunks; no token truncation.',flush=True)
    sums=np.zeros((len(test),384),dtype=np.float64)
    owners=np.asarray(owners);weights=np.asarray(weights)
    for start in range(0,len(chunks),cfg['batch_size']):
        end=min(start+cfg['batch_size'],len(chunks))
        batch=t.pad(chunks[start:end],padding=True,return_tensors='pt')
        with torch.inference_mode():
            emb=torch.nn.functional.normalize(encoder(batch)['sentence_embedding'],dim=1).cpu().numpy()
        np.add.at(sums,owners[start:end],emb*weights[start:end,None])
    result=normalize(sums/np.array([len(x) for x in tokens])[:,None]).astype(np.float32)
    assert result.shape==(3000,384) and np.isfinite(result).all()
    np.save(array,result);pd.DataFrame(coverage).to_csv(FEATURES/'submission_test_chunks.csv',index=False)
    meta.write_text(json.dumps({'key':key,'config':cfg,'rows':len(test)},indent=2))
    return result

def generate(train,test,sample,X,spec,metrics):
    emb=test_embeddings(test,spec['MiniLM_chunked'])
    Xt=pd.concat([test[['id','text']].reset_index(drop=True),pd.DataFrame(emb,columns=EMB_COLS)],axis=1)
    print('Refitting the calibrated C=1 pipeline on all 10,000 labeled articles...',flush=True)
    model=fit_checked(X,train.label)
    prob=model.predict_proba(Xt)[:,1]
    predictions=pd.Series(prob,index=test.id)
    submission=pd.DataFrame({'id':sample.id,'label':predictions.loc[sample.id].to_numpy()})
    assert list(submission.columns)==['id','label'] and len(submission)==3000
    assert submission.id.is_unique and set(submission.id)==set(test.id)
    assert np.isfinite(submission.label).all() and submission.label.between(0,1).all()
    assert submission.label.nunique()>1
    out=ROOT/'submissions';out.mkdir(exist_ok=True)
    path=out/'submission_Jianing_TF_MiniLM_SVM_C1.csv'
    submission.to_csv(path,index=False,float_format='%.12g')
    reread=pd.read_csv(path)
    assert reread.id.tolist()==sample.id.tolist() and reread.label.between(0,1).all()
    receipt={'method':'L2 normalized TF + full-text chunked MiniLM, equal-weight fusion, LinearSVC C=1',
             'calibration':'sigmoid, 5-fold shuffled stratified CV seed=42, ensemble=False; vocabulary refit inside each fold',
             'full_training_rows':10000,'test_rows':3000,'validation':metrics,
             'feature_spec':spec,'probability_min':float(prob.min()),'probability_max':float(prob.max()),
             'submission_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'test_sha256':hashlib.sha256((ROOT/'data/raw/test.csv').read_bytes()).hexdigest(),
             'full_model_features':len(model.calibrated_classifiers_[0].estimator.named_steps['features'].vectorizer_.vocabulary_)+384,
             'note':'Validation scores are not Kaggle scores. Sample labels were not used for training.'}
    (ROOT/'results/jianing_submission/submission_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('Verified submission:',path,flush=True)
    print(submission.head().to_string(index=False),flush=True)
    print('P(AI) range:',prob.min(),prob.max(),flush=True)
    return path

if __name__=='__main__':
    train,test,sample,X,split,spec=prepare_data()
    metrics=validate(train,X,split)
    generate(train,test,sample,X,spec,metrics)
