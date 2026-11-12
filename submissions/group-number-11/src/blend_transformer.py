"""Blend held-out neural predictions with the scored language_v2 detector."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from train import ROOT


def combine(a,b,weight,method):
    if method=='logit':return expit((1-weight)*logit(np.clip(a,1e-6,1-1e-6))+weight*logit(np.clip(b,1e-6,1-1e-6)))
    return (1-weight)*a+weight*b


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--weight',type=float,default=.1);ap.add_argument('--method',choices=['plain','logit'],default='logit')
    ap.add_argument('--neural',default='roberta_v1');ap.add_argument('--base',default='language_v2');ap.add_argument('--name',default='neural_v3');args=ap.parse_args()
    art=ROOT/'artifacts/optimization';out=ROOT/'results/optimization'
    va=pd.read_csv(art/(args.base+'_validation.csv'));n=np.load(art/args.neural/'validation.npz');assert np.array_equal(va.id.to_numpy(),n['ids'])
    pred=combine(va.probability.to_numpy(),n['probability'],args.weight,args.method)
    frame=va[['id','label']].copy();frame['probability']=pred;frame.to_csv(art/(args.name+'_validation.csv'),index=False)
    base=pd.read_csv(ROOT/'output'/('submission_'+args.base+'.csv'));test=np.load(art/args.neural/'test.npz')
    lookup=pd.Series(np.arange(len(test['ids'])),index=test['ids']);npred=test['probability'][lookup.loc[base.id].to_numpy()]
    p=combine(base.label.to_numpy(),npred,args.weight,args.method)
    result=base[['id']].copy();result['label']=p;sample=pd.read_csv(args.data/'sample_submission.csv')
    assert len(result)==3000 and result.id.equals(sample.id) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    target=ROOT/'output'/('submission_'+args.name+'.csv');result.to_csv(target,index=False)
    metadata={'candidate':args.name,'file':target.name,'base':args.base,'neural':args.neural,
        'neural_weight':args.weight,'method':args.method,'val_auc':roc_auc_score(va.label,pred),
        'public_auc':None,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'rows':len(result),
        'neural_training_rows':8000,'note':'The neural model uses only the original training partition; tree components are refit on all 10,000 labels after validation selection.'}
    (out/(args.name+'_delivery.json')).write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
