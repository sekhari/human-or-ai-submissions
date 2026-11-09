"""Score any new id,text CSV with fixed models; never submit to Kaggle."""
import argparse
import gc
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from src.encoder import DEVICE,predict as legacy_predict
from src.encoder_v2 import predict_encoder
from src.lm_features import MODELS,extract,combine


def predict_rows(bundle,directory,texts,offline_root=None,batch=8):
    if bundle.get('lm_models')!=list(MODELS):raise ValueError('Expected the validated dual-LM model bundle.')
    features=combine(*(extract(texts,model_id,batch,offline_root) for model_id in MODELS))
    deberta=predict_encoder(texts,directory/bundle['deberta_dir'],batch_size=batch)
    distil_dir=directory/bundle['distil_dir']
    model=AutoModelForSequenceClassification.from_pretrained(distil_dir,token=False,attn_implementation='eager').to(DEVICE)
    tok=AutoTokenizer.from_pretrained(distil_dir,token=False)
    distil=legacy_predict(model,tok,list(texts),batch=batch)
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    result=bundle['ensemble'].score(texts,features,distil,deberta)
    if not np.isfinite(result).all() or np.any((result<0)|(result>1)):raise ValueError('Invalid scores.')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--offline-root',type=Path,help='Optional directory containing gpt2 and gpt2-large pretrained folders')
    p.add_argument('--batch',type=int,default=8)
    a=p.parse_args()
    if a.batch<1:raise ValueError('Batch size must be positive.')
    frame=pd.read_csv(a.input,keep_default_na=False)
    if not {'id','text'}.issubset(frame.columns) or not len(frame):raise ValueError('Expected a nonempty CSV with id and text columns.')
    if frame.id.duplicated().any():raise ValueError('Input IDs must be unique.')
    if not frame.text.map(lambda x:isinstance(x,str) and bool(x.strip())).all():raise ValueError('Every passage must contain text.')
    bundle=joblib.load(a.model/'model.joblib')
    result=predict_rows(bundle,a.model,frame.text.tolist(),a.offline_root,a.batch)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'id':frame.id,'label':result}).to_csv(a.output,index=False)
    print(json.dumps({'rows':len(frame),'output':str(a.output),'weights':bundle['ensemble'].weights.tolist(),
                      'fit_rows':bundle.get('fit_rows'),'stage':bundle.get('stage')}),flush=True)

if __name__=='__main__':main()
