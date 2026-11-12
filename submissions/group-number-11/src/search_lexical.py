"""Whitespace-invariant sparse models; vocabularies fit on training only."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from advanced_features import canonical
from train import ROOT,SEED,classifier,probability


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
    art=ROOT/'artifacts/optimization';out=ROOT/'results/optimization'
    df=pd.read_csv(args.data/'train.csv');y=df.label.to_numpy();texts=df.text.map(canonical)
    tr,va=train_test_split(np.arange(len(df)),test_size=2000,stratify=y,random_state=SEED)
    results=[];models={};predictions={};vectorizers={}
    for analyzer,ngram in [('char',(2,5)),('char_wb',(3,5))]:
        start=time.perf_counter();v=TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,lowercase=True,
            min_df=3,max_features=250000,sublinear_tf=True,dtype=np.float32)
        xt=v.fit_transform(texts.iloc[tr]);xv=v.transform(texts.iloc[va]);elapsed=time.perf_counter()-start
        vectorizers[analyzer]=v
        for kind,c in [('lr',1),('lr',4),('lr',16),('svm',1),('svm',4)]:
            start=time.perf_counter();m=classifier(kind,c);m.fit(xt,y[tr]);p=probability(m,xv)
            name=f'flat_{analyzer}_{kind}_{c}';r={'name':name,'config':{'analyzer':analyzer,'ngram':ngram,'kind':kind,'C':c},
                'val_auc':roc_auc_score(y[va],p),'train_auc':roc_auc_score(y[tr],probability(m,xt)),
                'seconds':time.perf_counter()-start+elapsed}
            print(json.dumps(r),flush=True);results.append(r);models[name]=m;predictions[name]={k:p for k in ['raw','flat','single','double']}
    (out/'lexical.json').write_text(json.dumps(results,indent=2))
    joblib.dump({'records':results,'models':models,'vectorizers':vectorizers,'predictions':predictions},art/'lexical.joblib',compress=3)


if __name__=='__main__':main()
