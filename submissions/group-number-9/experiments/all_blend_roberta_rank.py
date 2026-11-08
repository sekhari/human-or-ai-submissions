"""Portable reproduction of the public-0.99132, predeclared 10% rank blend.

Default: regenerate exact uploaded scores using committed component predictions.
Optional train-neural/score-neural stages rebuild the RoBERTa component locally.
No checkpoint or competition text is distributed with this addition.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ARTIFACTS=ROOT/'results/all_blend_roberta_rank'
CACHE=ROOT/'.cache/rank10_roberta'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blend(args,config):
    from src.rank_blend import blend_rank_reference
    baseline=pd.read_csv(args.baseline)
    # Round-trip parsing preserves the original float64 probabilities and ties.
    neural=pd.read_csv(args.neural,float_precision='round_trip')
    if list(baseline.columns)!=['id','label'] or list(neural.columns)!=['id','probability']:
        raise ValueError('Expected id,label baseline and id,probability neural scores')
    if len(baseline)!=3000:
        raise ValueError('Expected exactly 3000 test predictions')
    result=blend_rank_reference(baseline.id.to_numpy(),baseline.id.to_numpy(),baseline.label.to_numpy(),
        neural.id.to_numpy(),neural.probability.to_numpy(),config['neural_weight'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.verify_original:
        assert sha(args.baseline)==config['reference_csv_sha256'], 'Baseline differs from the submitted component'
        assert sha(args.neural)==config['component_predictions_sha256'], 'Neural artifact differs from original'
    result.to_csv(args.out,index=False)
    digest=sha(args.out)
    if args.verify_original:
        assert digest==config['expected_submission_sha256'], 'Output differs from the original uploaded CSV'
    print(json.dumps({'output':str(args.out),'rows':len(result),'sha256':digest,
        'matches_original_upload':digest==config['expected_submission_sha256'],
        'offline_validation_auroc':None},indent=2))


def neural(args,config):
    import numpy as np
    import torch
    from src.deberta import device_for
    from src.roberta_experiment import frame_hash,load_model,predict,tokenizer,train_epoch
    device=device_for(args.device)
    torch.set_num_threads(4)
    if args.stage=='train-neural':
        from huggingface_hub import snapshot_download
        snapshot_download(config['model'],revision=config['revision'],token=False,
            cache_dir=ROOT/'.cache/huggingface',allow_patterns=['config.json','model.safetensors',
            'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'])
        frame=pd.read_csv(args.train)[['id','text','label']]
        frame['soft_target']=frame.pop('label').astype(float)
        if len(frame)!=10000 or frame_hash(frame)!=config['original_training_frame_sha256']:
            raise ValueError('Training rows/order/content do not match the original full fit')
        # Horizon stays at TWO epochs; run ONLY epoch 1 (625 optimizer updates).
        result=train_epoch(frame,config['training'],CACHE,1,device,
                           lambda row:print(json.dumps(row),flush=True))
        assert result['updates']==625 and result['pseudo_rows_available']==0
        print(json.dumps(result),flush=True)
    elif args.stage=='score-neural':
        test=pd.read_csv(args.test)[['id','text']]
        if len(test)!=3000 or not test.id.is_unique:raise ValueError('Expected 3000 unique test IDs')
        model=load_model(checkpoint=CACHE/'epoch_1').to(device)
        p=predict(model,test,tokenizer(),device,config['training']['eval_batch_size'])
        assert np.isfinite(p).all()
        args.neural.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'id':test.id,'probability':p}).to_csv(args.neural,index=False,float_format='%.17g')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['blend','train-neural','score-neural','retrain'],default='blend')
    parser.add_argument('--baseline',type=Path,default=ROOT/'submissions/improve_all_blend.csv')
    parser.add_argument('--neural',type=Path,default=ARTIFACTS/'roberta_test_predictions.csv')
    parser.add_argument('--out',type=Path,default=ROOT/'submissions/all_blend_roberta_rank10_10000.csv')
    parser.add_argument('--train',type=Path,default=ROOT/'Data/train.csv')
    parser.add_argument('--test',type=Path,default=ROOT/'Data/test.csv')
    parser.add_argument('--device',choices=['auto','mps','cpu','cuda'],default='auto')
    parser.add_argument('--verify-original',action='store_true')
    args=parser.parse_args()
    config=json.loads((ARTIFACTS/'config.json').read_text())
    if args.stage=='blend':blend(args,config)
    elif args.stage=='retrain':
        if args.verify_original:raise ValueError('Retraining is not guaranteed bitwise identical across hardware')
        # Fresh GPU process for training and inference; never overwrite the committed score artifact.
        fresh=CACHE/'retrained_test_predictions.csv'
        for stage in ['train-neural','score-neural','blend']:
            subprocess.run([sys.executable,'-m','experiments.all_blend_roberta_rank',
                '--stage',stage,'--baseline',str(args.baseline),'--neural',str(fresh),
                '--train',str(args.train),'--test',str(args.test),'--device',args.device,
                '--out',str(args.out)],cwd=ROOT,check=True)
    else:
        if args.stage=='score-neural' and args.neural.resolve()==(ARTIFACTS/'roberta_test_predictions.csv').resolve():
            raise ValueError('Use --neural .cache/rank10_roberta/retrained_test_predictions.csv to preserve original predictions')
        neural(args,config)


if __name__=='__main__':main()
