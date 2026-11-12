"""Write id,label probabilities for Kaggle; never upload."""
import argparse
from pathlib import Path
from common import read_data,write_predictions,read_json,write_json,sha
from ensembles import RECIPES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['logistic','svm','gru','ensemble'],required=True)
    p.add_argument('--recipe',choices=RECIPES,default='best');p.add_argument('--stack-config',type=Path)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--cache-dir',type=Path);p.add_argument('--offline',action='store_true');p.add_argument('--allow-small-test',action='store_true',help='Smoke testing only')
    a=p.parse_args();d=read_data(a.test)
    if len(d)!=3000 and not a.allow_small_test:p.error('Expected 3000 competition test rows; use --allow-small-test only for a smoke test')
    run=read_json(a.artifacts/'run.json')
    if run['model']!=a.model:p.error('Artifact model type differs from --model')
    if not run['full']:print('NOTE: these artifacts were fitted with a validation holdout; use --full training for final submissions.')
    if a.model in ['logistic','svm']:
        from classical import load
        pred=load(a.artifacts).predict_proba(d.text)[:,1]
    else:
        import torch
        torch.set_num_threads(a.threads);device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
        if a.model=='gru':
            from scratch import predict
            pred=predict(a.artifacts,d.text,device)
        else:
            from transformer_models import predict_members
            from ensembles import families,combine
            pred=combine(families(predict_members(a.artifacts,d,device,a.cache_dir,a.offline)),a.recipe,read_json(a.stack_config) if a.stack_config else None)
    write_predictions(a.out,d,pred)
    write_json(a.out.with_suffix('.manifest.json'),dict(rows=len(d),model=a.model,recipe=a.recipe if a.model=='ensemble' else None,
        input_sha256=sha(a.test),submission_sha256=sha(a.out),uploaded=False))
    print('Wrote',a.out)
if __name__=='__main__':main()
