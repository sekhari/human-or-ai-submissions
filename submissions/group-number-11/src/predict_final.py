"""Write the frozen final blend from the model predictions produced by training.

Run reproduce_final.sh first to train/infer all components. This entrypoint does
not retune weights or read training/validation labels; it uses recorded weights.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from blend_transformer import combine
from train import ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'output/submission.csv');args=ap.parse_args()
    art=ROOT/'artifacts/optimization';out=ROOT/'results/optimization'
    sample=pd.read_csv(args.data/'sample_submission.csv')
    language=pd.read_csv(ROOT/'output/submission_language_v2.csv')
    medium=pd.read_csv(ROOT/'output/submission_language_medium.csv')
    neural=np.load(art/'roberta_v1/test.npz')
    assert language.id.equals(sample.id) and medium.id.equals(sample.id)
    lookup=pd.Series(np.arange(len(neural['ids'])),index=neural['ids'])
    p_neural=neural['probability'][lookup.loc[sample.id].to_numpy()]
    config=json.loads((out/'neural_v3_delivery.json').read_text())
    p=combine(language.label.to_numpy(),p_neural,config['neural_weight'],config['method'])
    # Match the scored pipeline's intermediate CSV round-trip exactly.
    intermediate=ROOT/'output/submission_neural_v3.csv'
    pd.DataFrame({'id':sample.id,'label':p}).to_csv(intermediate,index=False)
    p=pd.read_csv(intermediate).label.to_numpy()
    final=json.loads((out/'medium_v4_delivery.json').read_text())['selected']
    p=combine(p,medium.label.to_numpy(),final['second_weight'],final['method'])
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and sample.id.is_unique
    args.out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'id':sample.id,'label':p}).to_csv(args.out,index=False)
    print(f'Wrote {len(sample)} final probabilities to {args.out}')


if __name__=='__main__':main()
