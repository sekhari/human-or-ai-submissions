"""Verify complete inference on different batch sizes and input order."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from predict_robust import predict_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    texts=pd.read_csv(a.input,keep_default_na=False).text.head(8).tolist()
    bundle=joblib.load(a.model/'model.joblib')
    singular=predict_rows(bundle,a.model,texts,batch=1)
    together=predict_rows(bundle,a.model,texts,batch=8)
    reversed_scores=predict_rows(bundle,a.model,texts[::-1],batch=8)[::-1]
    difference=max(float(np.max(np.abs(singular-together))),float(np.max(np.abs(together-reversed_scores))))
    if not np.allclose(singular,together,atol=1e-4,rtol=1e-4) or not np.allclose(together,reversed_scores,atol=1e-4,rtol=1e-4):
        raise AssertionError(f'Inference depends on batch/order beyond numerical tolerance: {difference}')
    result={'passed':True,'rows':len(texts),'maximum_score_difference':difference,
            'model_sha256':hashlib.sha256((a.model/'model.joblib').read_bytes()).hexdigest()}
    (a.model/'inference_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
