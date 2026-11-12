"""Reproducible uncertainty, subgroup, duplicate audit, and report figures."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from features import normalize
from train import ROOT, SEED, probability


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
    out=ROOT/'results'; report=ROOT/'report';report.mkdir(exist_ok=True)
    df=pd.read_csv(args.data/'train.csv');v=pd.read_csv(out/'validation_predictions.csv')
    cache=joblib.load(ROOT/'artifacts/validation.joblib');tr=cache['train'];va=cache['val']
    selection=json.loads((out/'selection.json').read_text())
    # Bootstrap stratified by observed validation class, conditional on this fitted model.
    rng=np.random.default_rng(SEED);y=v.label.to_numpy();p=v.probability.to_numpy()
    indexes=[np.flatnonzero(y==k) for k in [0,1]]
    samples=[roc_auc_score(y[idx],p[idx]) for idx in
        [np.concatenate([rng.choice(i,len(i),replace=True) for i in indexes]) for _ in range(1000)]]
    summary={'val_auc':roc_auc_score(y,p),'bootstrap_95_ci':np.quantile(samples,[.025,.975]).tolist(),
             'confusion_matrix_at_0.5':confusion_matrix(y,p>=.5).tolist()}
    tp=np.zeros(len(tr))
    for kind,weight in zip(['word','char','style'],selection['selected']['weights']):
        if not weight:continue
        model=cache['models'][selection['best'][kind]]
        x=cache['style'][tr] if kind=='style' else cache['matrices'][kind][1]
        if isinstance(model,tuple): model,ratio=model;x=x.multiply(ratio)
        tp+=weight*probability(model,x)
    summary['train_auc']=roc_auc_score(df.label.iloc[tr],tp)
    groups=[]
    for lower,upper in [(0,150),(150,250),(250,100000)]:
        sub=v[(v.words>=lower)&(v.words<upper)]
        groups.append({'bin':f'{lower} <= words < {upper}' if upper<100000 else 'words >= 250',
            'n':len(sub),'human':int((sub.label==0).sum()),'ai':int((sub.label==1).sum()),
            'auc':roc_auc_score(sub.label,sub.probability)})
    summary['length_bins']=groups
    # Observable text similarity; only training-fit TF-IDF used. Not a source-label audit.
    _,xt,xv,_=cache['matrices']['word']
    nn=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute',n_jobs=4).fit(xt)
    distance,index=nn.kneighbors(xv);sim=1-distance[:,0]
    pd.DataFrame({'validation_id':df.id.iloc[va].to_numpy(),'nearest_training_id':df.id.iloc[tr[index[:,0]]].to_numpy(),
                  'cosine_similarity':sim}).to_csv(out/'nearest_neighbors.csv',index=False)
    summary['near_duplicate_audit']={'criterion':'word TF-IDF cosine similarity >= 0.90',
        'count':int((sim>=.9).sum()),'max_similarity':float(sim.max()),
        'auc_excluding_high_similarity':roc_auc_score(y[sim<.9],p[sim<.9])}
    errors=[]
    for typ,label,order in [('FP',0,False),('FN',1,True)]:
        subset=v[(v.label==label)&((v.probability>=.5) if label==0 else (v.probability<.5))]
        for _,r in subset.sort_values('probability',ascending=order).head(5).iterrows():
            text=df.set_index('id').loc[int(r.id),'text']
            errors.append({'id':int(r.id),'type':typ,'probability':r.probability,
                'words':int(r.words),'excerpt':normalize(text)[:800]})
    (ROOT/'artifacts/errors_for_review.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False))
    (out/'analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.prop_cycle':plt.cycler(color=['#236B82','#C46B36','#19334C'])})
    curve=pd.read_json(out/'learning_curve.json')
    fig,ax=plt.subplots(figsize=(6.6,3.0),layout='constrained')
    ax.plot(curve.n,curve.train_auc,'o-',label='Training');ax.plot(curve.n,curve.val_auc,'o-',label='Validation')
    ax.set(xlabel='Labeled training examples',ylabel='AUROC',ylim=(.75,1.01));ax.legend();ax.grid(alpha=.18)
    fig.savefig(report/'learning_curve.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
    fpr,tpr,_=roc_curve(y,p);axes[0].plot(fpr,tpr,label=f'AUC = {summary["val_auc"]:.4f}')
    axes[0].plot([0,1],[0,1],color='gray',linestyle='--',linewidth=.7);axes[0].set(xlabel='False positive rate',ylabel='True positive rate');axes[0].legend()
    for label,name in [(0,'Human'),(1,'AI')]:axes[1].hist(p[y==label],bins=20,alpha=.55,label=name)
    axes[1].set(xlabel='Predicted P(AI)',ylabel='Validation passages');axes[1].legend()
    fig.savefig(report/'diagnostics.pdf');plt.close(fig)


if __name__=='__main__':main()
