"""Train the fixed four-method robustness candidate using supplied data only."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.encoder import train_encoder,predict as legacy_predict,DEVICE
from src.encoder_v2 import fit_encoder,predict_encoder
from src.robust_ensemble import RobustEnsemble
from src.lm_features import MODELS,LARGE_REVISION,extract,combine
from src.recipes import BASELINE_WEIGHTS,CURRENT_WEIGHTS

TRAINING_SHA256='325e345faf88fb1c4203278098c282e40319a87b4bf17e094d2f235f51d0c07e'


def atomic_array(path,**arrays):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as handle:np.savez(handle,**arrays)
    temporary.replace(path)


def partitions(labels):
    dev,audit=train_test_split(np.arange(len(labels)),test_size=2000,stratify=labels,random_state=3737)
    fit,meta=train_test_split(dev,test_size=1600,stratify=labels[dev],random_state=9407)
    cal,selection=train_test_split(meta,test_size=800,stratify=labels[meta],random_state=9413)
    return fit,cal,selection,audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--small-features',type=Path);p.add_argument('--large-features',type=Path)
    p.add_argument('--validation',action='store_true',help='Fit 6400 rows and reserve the frozen 2000-row audit instead of final refit')
    p.add_argument('--recipe',choices=('current','baseline'),default=None,
                   help='Defaults to current for final fitting, baseline for reproducing the original validation audit')
    a=p.parse_args();started=time.perf_counter()
    recipe=a.recipe or ('baseline' if a.validation else 'current')
    inference_weights=CURRENT_WEIGHTS if recipe=='current' else BASELINE_WEIGHTS
    tr=pd.read_csv(a.data/'train.csv',keep_default_na=False)
    digest=hashlib.sha256(tr[['id','text','label']].to_csv(index=False).encode()).hexdigest()
    if digest!=TRAINING_SHA256:raise ValueError('Use the original competition train.csv to reproduce this validated recipe.')
    t=tr.text.to_numpy();y=tr.label.to_numpy();base_fit,cal,_,_=partitions(y)
    fit=base_fit if a.validation else np.setdiff1d(np.arange(len(tr)),cal)
    stage='validation' if a.validation else 'final'
    source=Path(__file__).parent
    signature=hashlib.sha256(Path(__file__).read_bytes()+b''.join(path.read_bytes() for path in sorted((source/'src').glob('*.py')))).hexdigest()
    config={'training_sha256':digest,'source_sha256':signature,'weights':inference_weights.tolist(),'inference_recipe':recipe,
            'fit_rows':len(fit),'calibration_rows':len(cal),'encoder_epochs':2,'recipe':'four-method-robustness-v1','stage':stage}
    a.output.mkdir(parents=True,exist_ok=True);manifest=a.output/'training_config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:
        raise ValueError('This output folder belongs to another recipe. Choose a fresh output folder.')
    manifest.write_text(json.dumps(config,indent=2)+'\n')
    atomic_array(a.output/'partitions.npz',fit=fit,calibration=cal)
    matrices=[]
    for model_id,supplied,name in zip(MODELS,(a.small_features,a.large_features),('small','large')):
        feature_path=supplied or a.output/f'{name}_features.npz'
        if feature_path.exists():
            with np.load(feature_path) as stored:
                if str(stored['digest'])!=digest:raise ValueError('Feature cache dataset mismatch.')
                values=stored['features']
                if 'model' in stored and str(stored['model'])!=model_id:raise ValueError('Feature model mismatch.')
                if name=='large' and ('revision' not in stored or str(stored['revision'])!=LARGE_REVISION):
                    raise ValueError('Large-model feature revision mismatch.')
        else:
            if supplied:raise FileNotFoundError(feature_path)
            values=extract(t,model_id,batch=8,checkpoint=a.output/f'{name}_partial.npz')
            metadata={'model':model_id,'revision':LARGE_REVISION if name=='large' else 'gpt2'}
            atomic_array(feature_path,features=values,digest=digest,**metadata)
        if values.shape!=(len(tr),15) or not np.isfinite(values).all():raise ValueError('Invalid language-model feature cache.')
        matrices.append(values)
        print(f'{model_id} features ready',flush=True)
    features=combine(*matrices)
    classical_path=a.output/'classical.joblib'
    if classical_path.exists():classical=joblib.load(classical_path)
    else:
        classical=RobustEnsemble.fit_classical(t[fit],y[fit],features[fit]);joblib.dump(classical,classical_path)
    print('Reference and language-feature classifiers ready',flush=True)
    deb_dir=a.output/'deberta'
    if not (deb_dir/'training_complete.json').exists():fit_encoder(t[fit],y[fit],deb_dir)
    distil_dir=a.output/'distilroberta'
    if not (distil_dir/'training_complete.json').exists():
        model,tok=train_encoder(list(t[fit]),y[fit],log_every=100)
        model.save_pretrained(distil_dir);tok.save_pretrained(distil_dir)
        (distil_dir/'training_complete.json').write_text(json.dumps({'epochs':2,'rows':len(fit),'seed':3737})+'\n')
        del model,tok;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    # Calibration rows are excluded from every supervised component fit.
    deberta=predict_encoder(t[cal],deb_dir)
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    model=AutoModelForSequenceClassification.from_pretrained(distil_dir,token=False,attn_implementation='eager').to(DEVICE)
    distil=legacy_predict(model,AutoTokenizer.from_pretrained(distil_dir,token=False),list(t[cal]))
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    ensemble=RobustEnsemble(classical,weights=inference_weights).fit_calibration(t[cal],y[cal],features[cal],distil,deberta)
    joblib.dump({'ensemble':ensemble,'deberta_dir':deb_dir.name,'distil_dir':distil_dir.name,
                 'training_sha256':digest,'stage':stage,'fit_rows':len(fit),'calibration_rows':len(cal),
                 'lm_models':list(MODELS),'lm_feature_layout':'small,large,large-minus-small','inference_recipe':recipe},
                a.output/'model.joblib')
    config['elapsed_minutes']=(time.perf_counter()-started)/60
    config['python']=sys.version
    config['torch']=torch.__version__
    config['device']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else DEVICE
    if torch.cuda.is_available():config['peak_gpu_gib']=torch.cuda.max_memory_allocated()/2**30
    (a.output/'training_complete.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(config,indent=2),flush=True)

if __name__=='__main__':main()
