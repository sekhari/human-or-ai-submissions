"""Write the observed Kaggle contract: one id,label probability per supplied test ID."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from data_utils import load_data
from experiment import THREADS, probability_ai


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--sample-submission', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('artifacts/submission.csv'))
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f'Refusing to overwrite {args.out}; choose a new output path')
    data = load_data(args.test, labeled=False, expected_rows=3000)
    template = pd.read_csv(args.sample_submission, dtype={'id': str})
    if list(template.columns) != ['id', 'label'] or len(template) != 3000:
        parser.error('Sample submission must have exactly columns id,label and 3,000 rows')
    if template.id.isna().any() or template.id.duplicated().any():
        parser.error('Sample submission IDs must be nonmissing and unique')
    if set(template.id) != set(data.id.astype(str)):
        parser.error('Sample submission and test must contain exactly the same IDs')
    saved = joblib.load(args.model)
    if saved.get('artifact_version') != 1 or saved.get('schema') != {'id': 'id', 'text': 'text', 'label': 'label'}:
        parser.error('Unsupported model artifact or data schema')
    if saved.get('fit_scope') != 'all_labeled' or saved.get('train_rows') != 10000:
        parser.error('Kaggle predictions require a model refit on all 10,000 labeled examples with train.py --full-data')
    # Match the authoritative template order even if the test file was reordered.
    ordered = data.set_index('id').loc[template.id]
    with threadpool_limits(limits=THREADS):
        probability = probability_ai(saved['model'], ordered.text)
    submission = pd.DataFrame({'id': template.id, 'label': probability})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    readback = pd.read_csv(args.out, dtype={'id': str})
    if len(readback) != 3000 or readback.id.tolist() != template.id.tolist() or not readback.label.between(0, 1).all():
        raise RuntimeError('Written submission failed ID/probability verification')
    print(f'Wrote and verified {len(readback)} rows with columns id,label: {args.out}')


if __name__ == '__main__':
    main()
