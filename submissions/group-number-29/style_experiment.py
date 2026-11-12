"""Compare word TF-IDF with 20 simple style measurements on training-side folds."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from train import load_data, split_data, SEED

NAMES=('word_pairs','style','word_and_style')
FEATURES=['log_words','log_chars','mean_word_length','std_word_length','unique_word_ratio',
          'repeated_bigram_ratio','mean_sentence_words','std_sentence_words','log_sentences',
          'log_paragraphs','mean_line_words','std_line_words','short_line_ratio','blank_line_ratio',
          'uppercase_ratio','digit_ratio','comma_ratio','semicolon_ratio','question_ratio','exclamation_ratio']


def style_features(texts):
    rows=[]
    for text in texts:
        words=re.findall(r"\b[\w']+\b",text.lower());n=max(len(words),1);chars=max(len(text),1)
        lengths=[len(w) for w in words] or [0]
        sentences=[s for s in re.split(r'[.!?]+',text) if s.strip()]
        sl=[len(re.findall(r"\b[\w']+\b",s)) for s in sentences] or [0]
        lines=text.split('\n');ll=[len(s.split()) for s in lines if s.strip()] or [0]
        bigrams=list(zip(words,words[1:]))
        rows.append([np.log1p(len(words)),np.log1p(len(text)),np.mean(lengths),np.std(lengths),
                     len(set(words))/n,(len(bigrams)-len(set(bigrams)))/max(len(bigrams),1),
                     np.mean(sl),np.std(sl),np.log1p(len(sentences)),
                     np.log1p(len([p for p in re.split(r'\n\s*\n',text) if p.strip()])),
                     np.mean(ll),np.std(ll),np.mean(np.array(ll)<=8),
                     sum(not s.strip() for s in lines)/len(lines),
                     sum(c.isupper() for c in text)/chars,sum(c.isdigit() for c in text)/chars,
                     text.count(',')/chars,text.count(';')/chars,text.count('?')/chars,text.count('!')/chars])
    return np.asarray(rows,dtype=float)


def build_model(name):
    words=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000,sublinear_tf=True)
    style=Pipeline([('measure',FunctionTransformer(style_features)),('scale',StandardScaler())])
    features=words if name=='word_pairs' else style
    if name=='word_and_style':features=FeatureUnion([('words',words),('style',style)])
    return Pipeline([('features',features),('classifier',LogisticRegression(C=10,solver='liblinear',
                      max_iter=2000,random_state=SEED))])


def fit_score(name,training,validation):
    model=build_model(name)
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        start=time.perf_counter();model.fit(training.text,training.label);seconds=time.perf_counter()-start
        probs=model.predict_proba(validation.text)[:,list(model.classes_).index(1)]
    assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
    return model,probs,seconds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,default=Path('outputs/style_experiment'))
    args=p.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):p.error('Choose an empty output directory.')
    data=load_data(args.train,labeled=True);digest=hashlib.sha256(args.train.read_bytes()).hexdigest()
    split=Path(__file__).with_name('validation_ids.json')
    training,validation=split_data(data,split,digest);training=training.reset_index(drop=True)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    # Clusters approximate topics, not true source domains. Their vocabulary is not used as model features.
    with threadpool_limits(limits=1):
        grouping=TfidfVectorizer(stop_words='english',min_df=5,max_df=.8,max_features=10000)
        groups=MiniBatchKMeans(n_clusters=12,n_init=10,random_state=SEED,batch_size=1024).fit_predict(grouping.fit_transform(training.text))
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED).split(training.text,training.label,groups))
    manifest=training[['id','label']].copy();manifest['cluster']=groups;manifest['fold']=-1
    for i,(a,b) in enumerate(folds):
        assert not set(groups[a]) & set(groups[b])
        manifest.loc[b,'fold']=i
    manifest.to_csv(args.out_dir/'folds.csv',index=False)
    metadata=dict(seed=SEED,train_sha256=digest,split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),features=FEATURES,
                  selection='Highest mean grouped AUROC; fixed C=10; original holdout evaluated after selection')
    (args.out_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    results=[]
    for name in NAMES:
        for i,(a,b) in enumerate(folds):
            print(f'{name}: grouped fold {i}',flush=True)
            m,probs,seconds=fit_score(name,training.iloc[a],training.iloc[b])
            row=dict(model=name,fold=i,auroc=roc_auc_score(training.iloc[b].label,probs),fit_seconds=seconds,
                     train_rows=len(a),validation_rows=len(b))
            results.append(row);pd.DataFrame(results).to_csv(args.out_dir/'cv_results.csv',index=False)
            pred=training.iloc[b][['id','label']].copy();pred['probability_ai']=probs
            pred.to_csv(args.out_dir/f'{name}_fold{i}.csv',index=False)
            print(json.dumps(row),flush=True)
    means=pd.DataFrame(results).groupby('model').auroc.mean();winner=means.idxmax()
    summary=dict(winner=winner,grouped_auroc=means.to_dict(),holdout=[])
    for name in NAMES:
        m,probs,seconds=fit_score(name,training,validation)
        with threadpool_limits(limits=1):
            train_probs=m.predict_proba(training.text)[:,1]
            flat=m.predict_proba(validation.text.str.replace(r'\s+',' ',regex=True))[:,1]
        pred=validation[['id','label']].copy();pred['probability_ai']=probs;pred['flat_probability_ai']=flat
        pred.to_csv(args.out_dir/f'{name}_validation.csv',index=False)
        row=dict(model=name,train_auroc=roc_auc_score(training.label,train_probs),validation_auroc=roc_auc_score(validation.label,probs),
                 flat_validation_auroc=roc_auc_score(validation.label,flat),fit_seconds=seconds)
        summary['holdout'].append(row);print(json.dumps(row),flush=True)
        joblib.dump(m,args.out_dir/f'{name}_model.joblib')
    final=build_model(winner)
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        final.fit(data.text,data.label)
    joblib.dump(final,args.out_dir/'final_model.joblib')
    summary['final_training_rows']=len(data)
    (args.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    from style_experiment import main as run
    run()
