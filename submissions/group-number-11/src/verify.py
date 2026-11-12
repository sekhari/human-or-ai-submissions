"""Verify the final handoff against supplied data and selected validation model."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from train import ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
    test=pd.read_csv(args.data/'test.csv');sample=pd.read_csv(args.data/'sample_submission.csv')
    submission_path=ROOT/'output/submission_baseline.csv';submission=pd.read_csv(submission_path)
    assert submission.columns.tolist()==sample.columns.tolist()==['id','label']
    assert len(submission)==len(test)==3000
    assert submission.id.equals(sample.id)
    assert not submission.id.duplicated().any()
    assert set(submission.id)==set(test.id)
    assert np.isfinite(submission.label).all() and submission.label.between(0,1).all()
    train=pd.read_csv(args.data/'train.csv');split=pd.read_csv(ROOT/'results/split.csv')
    assert split.id.is_unique and set(split.id)==set(train.id)
    assert (split.partition=='validation').sum()==2000
    assert (split.partition=='training').sum()==8000
    val=pd.read_csv(ROOT/'results/validation_predictions.csv')
    assert set(val.id)==set(split.loc[split.partition=='validation','id'])
    assert np.array_equal(train.set_index('id').loc[val.id,'label'].to_numpy(),val.label.to_numpy())
    chosen=json.loads((ROOT/'results/selection.json').read_text())['selected']
    assert abs(roc_auc_score(val.label,val.probability)-chosen['val_auc'])<1e-10
    components=joblib.load(ROOT/'artifacts/final.joblib')
    assert abs(sum(c['weight'] for c in components)-1)<1e-12
    audit=json.loads((ROOT/'results/audit.json').read_text())
    for name,checksum in audit['data_sha256'].items():
        assert hashlib.sha256((args.data/name).read_bytes()).hexdigest()==checksum
    metadata=json.loads((ROOT/'report_metadata.json').read_text())
    manifest={'submission_rows':len(submission),'columns':submission.columns.tolist(),
        'submission_sha256':hashlib.sha256(submission_path.read_bytes()).hexdigest(),
        'model_sha256':hashlib.sha256((ROOT/'artifacts/final.joblib').read_bytes()).hexdigest(),
        'validation_auc':chosen['val_auc'],'kaggle_uploaded':metadata.get('kaggle_uploaded_by_user',False),
        'public_auc_user_reported':metadata.get('new_public_auc'),
        'prior_public_auc_user_reported':0.92433,
        'checks':'IDs, sample order, finite [0,1] probabilities, data hashes, fixed split, validation labels/AUC and blend weights passed.'}
    (ROOT/'results/baseline_delivery_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
