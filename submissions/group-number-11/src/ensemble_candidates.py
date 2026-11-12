"""Finite held-out selection of blends between two reproducible candidates."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from features import normalize
from blend_transformer import combine
from train import ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--first',default='neural_v3');ap.add_argument('--second',default='language_medium');ap.add_argument('--name',default='medium_v4');args=ap.parse_args()
    art=ROOT/'artifacts/optimization';out=ROOT/'results/optimization';a=pd.read_csv(art/(args.first+'_validation.csv'));b=pd.read_csv(art/(args.second+'_validation.csv'))
    assert a.id.equals(b.id) and a.label.equals(b.label)
    df=pd.read_csv(args.data/'train.csv').set_index('id');hard=df.loc[a.id,'text'].map(lambda t:'\n\n' not in normalize(t)).to_numpy();y=a.label.to_numpy();rows=[]
    base_auc=roc_auc_score(y,a.probability)
    for method in ['plain','logit']:
        for weight in [.25,.5,.75]:
            pred=combine(a.probability.to_numpy(),b.probability.to_numpy(),weight,method)
            auc=roc_auc_score(y,pred);sub=roc_auc_score(y[hard],pred[hard]);rows.append({'method':method,'second_weight':weight,'val_auc':auc,'blank0_auc':sub,'objective':.75*sub+.25*auc,'eligible':bool(auc>=base_auc-.00005)})
    chosen=max([r for r in rows if r['eligible']],key=lambda r:r['objective'])
    pred=combine(a.probability.to_numpy(),b.probability.to_numpy(),chosen['second_weight'],chosen['method'])
    validation=a[['id','label']].copy();validation['probability']=pred;validation.to_csv(art/(args.name+'_validation.csv'),index=False)
    ta=pd.read_csv(ROOT/'output'/('submission_'+args.first+'.csv'));tb=pd.read_csv(ROOT/'output'/('submission_'+args.second+'.csv'));assert ta.id.equals(tb.id)
    result=ta[['id']].copy();result['label']=combine(ta.label.to_numpy(),tb.label.to_numpy(),chosen['second_weight'],chosen['method'])
    sample=pd.read_csv(args.data/'sample_submission.csv');assert result.id.equals(sample.id) and result.label.between(0,1).all() and np.isfinite(result.label).all()
    path=ROOT/'output'/('submission_'+args.name+'.csv');result.to_csv(path,index=False)
    delivery={'candidate':args.name,'first':args.first,'second':args.second,'selected':chosen,'grid':rows,
        'file':path.name,'public_auc':None,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(result),
        'rule':'Retain >=25% of the currently scored best; native AUC within .00005 of parent, then maximize .75*blank0 AUC + .25*overall.'}
    (out/(args.name+'_delivery.json')).write_text(json.dumps(delivery,indent=2));print(json.dumps(chosen,indent=2),flush=True)


if __name__=='__main__':main()
