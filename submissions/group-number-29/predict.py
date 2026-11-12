"""Load the trained pipeline and write AI probabilities in sample submission order."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train import load_data


def make_submission(model, test, sample):
    if list(sample.columns) != ['id', 'label'] or sample['id'].isna().any():
        raise ValueError('Sample submission must have id,label columns and valid IDs')
    if sample['id'].duplicated().any() or set(sample['id']) != set(test['id']):
        raise ValueError('Sample IDs must match the test IDs exactly, without duplicates')
    if set(model.classes_) != {0, 1}:
        raise ValueError('Expected a model trained on human=0 and AI=1')
    ai_column = list(model.classes_).index(1)
    probabilities = model.predict_proba(test['text'])[:, ai_column]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Predictions must be finite probabilities between 0 and 1')
    # Map by ID, since the sample file can use a different row order than test.csv.
    by_id = pd.Series(probabilities, index=test['id'])
    submission = sample[['id']].copy()
    submission['label'] = submission['id'].map(by_id)
    return submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('outputs/baseline/model.joblib'))
    parser.add_argument('--out', type=Path, default=Path('outputs/baseline/submission.csv'))
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output file already exists; choose another --out path')
    test = load_data(args.test)
    if len(test) != 3000:
        raise ValueError('Expected 3,000 test passages')
    sample = pd.read_csv(args.sample, dtype={'id': str}, keep_default_na=False)
    # Load only model files created by you or a trusted teammate.
    model = joblib.load(args.model)
    submission = make_submission(model, test, sample)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f'Saved {len(submission)} checked predictions to {args.out}')


if __name__ == '__main__':
    main()
