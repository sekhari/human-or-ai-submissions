"""Recompute saved metrics and validate split/selection/submission integrity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[2]
    parser.add_argument('--data',type=Path,default=root/'data')
    parser.add_argument('--out',type=Path,default=root/'results/yuchen/catboost_v1')
    args=parser.parse_args()
    train=pd.read_csv(args.data/'train.csv').set_index('id')
    test=pd.read_csv(args.data/'test.csv')
    split=pd.read_csv(args.out/'split.csv')
    assert split.id.is_unique and set(split.id)==set(train.index)
    counts=split.role.value_counts().to_dict()
    assert counts=={'inner_fit':6400,'outer_validation':2000,'inner_tune':1600},counts
    validation=split.loc[split.role=='outer_validation','id'].to_numpy()
    records=json.loads((args.out/'metrics.json').read_text())
    for record in records:
        d=pd.read_csv(args.out/f'val_{record["model"]}.csv')
        np.testing.assert_array_equal(d.id,validation)
        np.testing.assert_array_equal(d.label,train.loc[d.id,'label'])
        assert np.isfinite(d.pred).all() and d.pred.between(0,1).all()
        assert abs(roc_auc_score(d.label,d.pred)-record['auc'])<1e-12
        assert abs(log_loss(d.label,d.pred)-record['logloss'])<1e-10
    for path in args.out.glob('cat_*_selection.json'):
        selected=json.loads(path.read_text())
        if 'grid' not in selected: continue
        assert selected['inner_auc']==max(x['inner_auc'] for x in selected['grid'])
    best=json.loads((args.out/'best_style.json').read_text())
    choices=[json.loads((args.out/f'cat_{rep}_selection.json').read_text()) for rep in ['style','extended']]
    assert best['inner_auc']==max(x['inner_auc'] for x in choices)
    for path in args.out.glob('submission_*.csv'):
        d=pd.read_csv(path)
        assert list(d.columns)==['id','label']
        assert len(d)==3000 and d.id.is_unique
        np.testing.assert_array_equal(d.id,test.id)
        assert np.isfinite(d.label).all() and d.label.between(0,1).all()
    for record in records:
        if not record['model'].startswith('cat_pseudo_t'): continue
        d=pd.read_csv(args.out/f'{record["model"]}_selection.csv')
        config=record['config']
        assert int(d.selected.sum())==config['selected']
        p=d.loc[d.selected,'teacher_prob']
        assert ((p>=config['threshold'])|(p<=1-config['threshold'])).all()
        assert int((p>=0.5).sum())==config['pseudo_positive']
    print(json.dumps({'status':'passed','validation_predictions_checked':len(records),
                      'submissions_checked':len(list(args.out.glob('submission_*.csv'))),
                      'split':counts},indent=2))


if __name__=='__main__': main()
