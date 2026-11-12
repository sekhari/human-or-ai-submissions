"""Generate probabilities in exactly the sample-submission order."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from features import style_features
from train import ROOT, probability


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--test',type=Path,required=True)
    ap.add_argument('--sample',type=Path,required=True)
    ap.add_argument('--model',type=Path,default=ROOT/'artifacts/final.joblib')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();df=pd.read_csv(args.test);sample=pd.read_csv(args.sample)
    if df.id.duplicated().any() or sample.id.duplicated().any() or set(df.id)!=set(sample.id):
        raise ValueError('IDs must be unique and match the sample submission.')
    p=np.zeros(len(df))
    for component in joblib.load(args.model):
        if component['kind']=='style': x=style_features(df.text)
        else:
            x=component['vectorizer'].transform(df.text)
            if component['ratio'] is not None:x=x.multiply(component['ratio'])
        p+=component['weight']*probability(component['model'],x)
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('Invalid probabilities')
    result=sample[['id']].merge(pd.DataFrame({'id':df.id,'label':p}),on='id',validate='one_to_one',sort=False)
    args.out.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.out,index=False)
    print(f'Wrote {len(result)} rows to {args.out}; P(AI) range {p.min():.6f}..{p.max():.6f}')


if __name__=='__main__':main()
