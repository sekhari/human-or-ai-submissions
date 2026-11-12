"""Generalization/error analysis for the selected language-feature model."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix
from advanced_features import load_lm_matrix
from predict_optimized import conservative_blend
from train import ROOT,SEED,probability
from features import normalize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--candidate',default='language_v2');ap.add_argument('--search',default='language_small');ap.add_argument('--train-cache-only',action='store_true');args=ap.parse_args()
    art=ROOT/'artifacts/optimization';out=ROOT/'results/optimization';df=pd.read_csv(args.data/'train.csv')
    frame=pd.read_csv(art/(args.candidate+'_validation.csv'));y=frame.label.to_numpy();p=frame.probability.to_numpy()
    selected=json.loads((out/(args.candidate+'.json')).read_text());cache=joblib.load(art/(args.search+'.joblib'))
    tr=cache['tr'];va=cache['va'];config=selected['config'];rich=joblib.load(art/'rich_train.joblib');lm=load_lm_matrix(art,'train',config.get('lm_dirs',['lm']))
    model=cache['models'][selected['selected']['source']];view='flat' if config['view']=='canonical' else 'raw'
    train_p=model.predict_proba(np.column_stack([rich[view][tr],lm[tr]]))[:,1]
    original=joblib.load(ROOT/'artifacts/validation.joblib');base_train=np.zeros(len(tr));base_selection=json.loads((ROOT/'results/selection.json').read_text())
    for kind,w in zip(['word','char','style'],base_selection['selected']['weights']):
        fitted=original['models'][base_selection['best'][kind]];x=original['style'][tr] if kind=='style' else original['matrices'][kind][1]
        base_train+=w*probability(fitted,x)
    blend=selected['selected'];train_p=conservative_blend(train_p,base_train,df.text.iloc[tr].tolist(),blend['baseline_weight'],blend['method'])
    np.savez_compressed(art/(args.candidate+'_train.npz'),ids=df.id.iloc[tr].to_numpy(),labels=df.label.iloc[tr].to_numpy(),probability=train_p)
    if args.train_cache_only:
        print('Saved controlled training predictions',args.candidate);return
    rng=np.random.default_rng(SEED);classes=[np.flatnonzero(y==k) for k in [0,1]];bootstrap=[]
    for _ in range(1000):
        ids=np.concatenate([rng.choice(c,len(c),replace=True) for c in classes]);bootstrap.append(roc_auc_score(y[ids],p[ids]))
    words=df.set_index('id').loc[frame.id,'text'].map(lambda t:len(normalize(t).split())).to_numpy();groups=[]
    for low,high in [(0,150),(150,250),(250,100000)]:
        mask=(words>=low)&(words<high);groups.append({'bin':f'{low} <= words < {high}' if high<100000 else 'words >= 250',
            'n':int(mask.sum()),'human':int((y[mask]==0).sum()),'ai':int((y[mask]==1).sum()),'auc':roc_auc_score(y[mask],p[mask])})
    nearest=pd.read_csv(ROOT/'results/nearest_neighbors.csv').set_index('validation_id').loc[frame.id]
    keep=nearest.cosine_similarity.to_numpy()<.9
    analysis={'candidate':args.candidate,'val_auc':roc_auc_score(y,p),'train_auc':roc_auc_score(df.label.iloc[tr],train_p),
        'bootstrap_95_ci':np.quantile(bootstrap,[.025,.975]).tolist(),'confusion_matrix_at_0.5':confusion_matrix(y,p>=.5).tolist(),
        'length_bins':groups,'near_duplicate_audit':{'count':int((~keep).sum()),'auc_excluding_high_similarity':roc_auc_score(y[keep],p[keep])}}
    errors=[]
    for typ,label,ascending in [('FP',0,False),('FN',1,True)]:
        bad=frame[(frame.label==label)&((frame.probability>=.5) if label==0 else (frame.probability<.5))]
        for _,row in bad.sort_values('probability',ascending=ascending).head(5).iterrows():
            text=df.set_index('id').loc[int(row.id),'text'];errors.append({'id':int(row.id),'type':typ,'probability':row.probability,
                'excerpt':normalize(text)[:850]})
    (art/'final_errors_for_review.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False))
    (out/'final_analysis.json').write_text(json.dumps(analysis,indent=2));print(json.dumps(analysis,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),layout='constrained');fpr,tpr,_=roc_curve(y,p)
    axes[0].plot(fpr,tpr,color='#236B82',label=f'AUC = {analysis["val_auc"]:.4f}');axes[0].plot([0,1],[0,1],'--',color='gray',lw=.7)
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate');axes[0].legend()
    for label,name,color in [(0,'Human','#236B82'),(1,'AI','#C46B36')]:axes[1].hist(p[y==label],bins=20,alpha=.55,label=name,color=color)
    axes[1].set(xlabel='Predicted P(AI)',ylabel='Validation passages');axes[1].legend();fig.savefig(ROOT/'report/diagnostics_optimized.pdf');plt.close(fig)


if __name__=='__main__':main()
