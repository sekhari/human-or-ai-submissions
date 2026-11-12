"""Final hybrid metrics; certify training AUC using bounded monotone ensembles.

For the neural logit blend, clipped neural probabilities lie in [eps,1-eps].
If the lowest positive score lower bound exceeds the highest negative upper
bound, every cross-class training rank is correct, so training AUC is exactly 1.
This certificate avoids unnecessary inference on all 8,000 training passages.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from blend_transformer import combine
from features import normalize
from train import ROOT,SEED


def training_bounds(name,art,out):
    if (art/(name+'_train.npz')).exists():
        d=np.load(art/(name+'_train.npz'));return d['ids'],d['labels'],d['probability'],d['probability']
    meta=json.loads((out/(name+'_delivery.json')).read_text())
    if 'neural_weight' in meta:
        ids,y,lo,hi=training_bounds(meta['base'],art,out);w=meta['neural_weight'];method=meta['method']
        return ids,y,combine(lo,np.zeros(len(y)),w,method),combine(hi,np.ones(len(y)),w,method)
    ids,y,lo1,hi1=training_bounds(meta['first'],art,out);ids2,y2,lo2,hi2=training_bounds(meta['second'],art,out)
    assert np.array_equal(ids,ids2) and np.array_equal(y,y2)
    chosen=meta['selected'];w=chosen['second_weight'];method=chosen['method']
    return ids,y,combine(lo1,lo2,w,method),combine(hi1,hi2,w,method)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--candidate',default='neural_v3');ap.add_argument('--no-report',action='store_true');ap.add_argument('--error-threshold',type=float,default=.4);args=ap.parse_args()
    art=ROOT/'artifacts/optimization';out=ROOT/'results/optimization';df=pd.read_csv(args.data/'train.csv').set_index('id')
    frame=pd.read_csv(art/(args.candidate+'_validation.csv'));y=frame.label.to_numpy();p=frame.probability.to_numpy()
    ids,train_y,lo,hi=training_bounds(args.candidate,art,out);lower=float(lo[train_y==1].min());upper=float(hi[train_y==0].max())
    if lower<=upper:raise RuntimeError('Ranking bounds overlap; full or boundary neural training inference is required.')
    certificate={'method':'Exact clipped-logit/monotone-ensemble ranking certificate','min_positive_lower_bound':lower,
        'max_negative_upper_bound':upper,'training_rows':len(ids),'train_auc':1.0}
    rng=np.random.default_rng(SEED);indices=[np.flatnonzero(y==label) for label in [0,1]];boot=[]
    for _ in range(1000):
        ix=np.concatenate([rng.choice(part,len(part),replace=True) for part in indices]);boot.append(roc_auc_score(y[ix],p[ix]))
    words=df.loc[frame.id,'text'].map(lambda t:len(normalize(t).split())).to_numpy();groups=[]
    for low,high in [(0,150),(150,250),(250,100000)]:
        mask=(words>=low)&(words<high);groups.append({'bin':f'{low} <= words < {high}' if high<100000 else 'words >= 250','n':int(mask.sum()),'human':int((y[mask]==0).sum()),'ai':int((y[mask]==1).sum()),'auc':roc_auc_score(y[mask],p[mask])})
    nearest=pd.read_csv(ROOT/'results/nearest_neighbors.csv').set_index('validation_id').loc[frame.id];keep=nearest.cosine_similarity.to_numpy()<.9
    analysis={'candidate':args.candidate,'val_auc':roc_auc_score(y,p),'train_auc':1.0,'training_certificate':certificate,
        'bootstrap_95_ci':np.quantile(boot,[.025,.975]).tolist(),'confusion_matrix_at_0.5':confusion_matrix(y,p>=.5).tolist(),
        'length_bins':groups,'near_duplicate_audit':{'count':int((~keep).sum()),'auc_excluding_high_similarity':roc_auc_score(y[keep],p[keep])}}
    analysis['error_review_threshold']=args.error_threshold
    analysis['error_review_confusion']=confusion_matrix(y,p>=args.error_threshold).tolist()
    (out/(args.candidate+'_analysis.json')).write_text(json.dumps(analysis,indent=2))
    if args.no_report:
        print(json.dumps(analysis,indent=2));return
    (out/'final_analysis.json').write_text(json.dumps(analysis,indent=2))
    errors=[]
    for typ,label,ascending in [('FP',0,False),('FN',1,True)]:
        bad=frame[(frame.label==label)&((frame.probability>=args.error_threshold) if label==0 else (frame.probability<args.error_threshold))]
        for _,row in bad.sort_values('probability',ascending=ascending).head(5).iterrows():
            errors.append({'id':int(row.id),'type':typ,'probability':row.probability,'excerpt':normalize(df.loc[int(row.id),'text'])[:850]})
    (art/'final_errors_for_review.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
    fpr,tpr,_=roc_curve(y,p);axes[0].plot(fpr,tpr,label=f'AUC = {analysis["val_auc"]:.4f}',color='#236B82');axes[0].plot([0,1],[0,1],'--',color='gray',lw=.7);axes[0].set(xlabel='False positive rate',ylabel='True positive rate');axes[0].legend()
    for label,name,color in [(0,'Human','#236B82'),(1,'AI','#C46B36')]:axes[1].hist(p[y==label],bins=20,alpha=.55,label=name,color=color)
    axes[1].set(xlabel='Predicted P(AI)',ylabel='Validation passages');axes[1].legend();fig.savefig(ROOT/'report/diagnostics_optimized.pdf');plt.close(fig)
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':main()
