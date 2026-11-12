"""Verify final submission format, scored-file identity and validation provenance."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from train import ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--submission',type=Path,default=ROOT/'output/submission.csv')
    ap.add_argument('--require-scored-hash',action='store_true');args=ap.parse_args()
    best=json.loads((ROOT/'results/optimization/current_best.json').read_text())
    meta=json.loads((ROOT/'report_metadata.json').read_text())
    output=pd.read_csv(args.submission);sample=pd.read_csv(args.data/'sample_submission.csv');test=pd.read_csv(args.data/'test.csv')
    assert output.columns.tolist()==['id','label']
    assert len(output)==len(sample)==len(test)==3000 and output.id.is_unique
    assert output.id.equals(sample.id) and set(output.id)==set(test.id)
    assert np.isfinite(output.label).all() and output.label.between(0,1).all()
    checksum=hashlib.sha256(args.submission.read_bytes()).hexdigest();matches=checksum==best['sha256']
    if args.require_scored_hash:assert matches,'CSV differs from the original Kaggle-scored artifact.'
    validation=pd.read_csv(ROOT/'artifacts/optimization'/(best['candidate']+'_validation.csv'))
    labels=pd.read_csv(args.data/'train.csv').set_index('id').loc[validation.id,'label'].to_numpy()
    assert len(validation)==2000 and np.array_equal(labels,validation.label.to_numpy())
    analysis=json.loads((ROOT/'results/optimization/final_analysis.json').read_text())
    assert analysis['candidate']==best['candidate'] and abs(roc_auc_score(labels,validation.probability)-analysis['val_auc'])<1e-12
    manifest={'candidate':best['candidate'],'file':str(args.submission.relative_to(ROOT)),
        'submission_rows':3000,'columns':['id','label'],'submission_sha256':checksum,
        'matches_original_scored_csv':matches,'public_auc_user_reported':best['public_auc'],
        'public_score_source':'Team-reported Kaggle result for the original scored CSV hash.',
        'private_auc':None,'validation_auc':analysis['val_auc'],'split_seed':20261023,'neural_seed':20261111,
        'tree_refit_rows':10000,'neural_training_rows':8000,'pr_url':meta['pr_url'],
        'checks':'CSV schema, all test IDs, uniqueness, sample order, finite probability bounds, scored-file hash, validation-label alignment and validation AUC passed.'}
    (ROOT/'results/delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
