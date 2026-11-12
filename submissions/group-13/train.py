"""Train selected baselines, the nine production members, or the OOF stack."""
import argparse
from pathlib import Path
import numpy as np
from common import read_data,split_data,write_json,metrics,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['logistic','svm','gru','ensemble','stack'],required=True)
    p.add_argument('--full',action='store_true',help='Use all supplied labels; no holdout score')
    p.add_argument('--seed',type=int,default=42,help='Baseline initialization and random-split seed; transformer seeds are frozen in configs/models.json')
    p.add_argument('--pooling',choices=['mean','attention'],default='attention');p.add_argument('--strong-dropout',action='store_true')
    p.add_argument('--epochs',type=int);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--cache-dir',type=Path);p.add_argument('--offline',action='store_true')
    p.add_argument('--micro-batch',type=int,help='Optional transformer memory override; changes numerical reproduction')
    p.add_argument('--limit',type=int,help='Balanced training subset for smoke tests only; not a scored reproduction')
    p.add_argument('--keep-fold-models',action='store_true');a=p.parse_args()
    d=read_data(a.train,True)
    if a.model=='stack' and (a.full or a.limit):p.error('Stack uses the frozen four full-data partitions; omit --full/--limit')
    if a.full and a.model=='gru' and not a.epochs:p.error('Specify --epochs from a prior validation run for a full GRU refit')
    if a.limit:
        if a.limit<10 or a.limit%2:p.error('--limit must be even and >=10')
        d=d.groupby('label',group_keys=False).sample(n=a.limit//2,random_state=a.seed)
    tr,va=(d,None) if a.full or a.model=='stack' else split_data(d,a.seed)
    if a.out.exists():p.error('Output directory exists; choose a fresh --out to protect prior artifacts')
    a.out.mkdir(parents=True);tr[['id','label']].to_csv(a.out/'training_ids.csv',index=False)
    if va is not None:va[['id','label']].to_csv(a.out/'validation_ids.csv',index=False)
    write_json(a.out/'run.json',dict(model=a.model,source_sha256=sha(a.train),training_rows=len(tr),validation_rows=0 if va is None else len(va),
        seed=a.seed,full=a.full,smoke_limit=a.limit))
    if a.model in ['logistic','svm']:
        import classical
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):m=classical.fit(a.model,tr,a.out,a.seed)
        if va is not None:pred=m.predict_proba(va.text)[:,1]
    else:
        import torch
        torch.set_num_threads(a.threads)
        device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
        if a.model in ['ensemble','stack'] and a.device=='auto' and device=='cpu':p.error('No GPU detected; --device cpu deliberately allows very slow execution')
        if a.model=='gru':
            import scratch
            scratch.fit(tr,va,a.out,device,a.seed,a.pooling,a.epochs or 20,a.strong_dropout)
            if va is not None:pred=scratch.predict(a.out,va.text,device)
        elif a.model=='stack':
            import stack_fit
            stack_fit.fit(tr,a.out,device,a.cache_dir,a.offline,a.keep_fold_models)
        else:
            from transformer_models import train_members,predict_members
            from ensembles import families,combine,RECIPES
            train_members(tr,a.out,device,a.cache_dir,a.offline,a.micro_batch)
            if va is not None:
                c=families(predict_members(a.out,va,device,a.cache_dir,a.offline))
                # Frozen stack coefficients were trained on all supplied labels;
                # applying them here would leak this holdout's labels.
                safe_recipes=[r for r in RECIPES if r not in ['stack','midpoint']]
                write_json(a.out/'recipe_metrics.json',{r:metrics(va.label,combine(c,r)) for r in safe_recipes})
                pred=combine(c,'best')
    if va is not None:
        import pandas as pd
        pd.DataFrame(dict(id=va.id,label=va.label,probability_AI=pred)).to_csv(a.out/'validation_predictions.csv',index=False)
        result=metrics(va.label,pred);write_json(a.out/'metrics.json',result);print(result)
    print('Artifacts:',a.out)
if __name__=='__main__':main()
