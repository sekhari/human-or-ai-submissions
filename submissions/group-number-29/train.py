"""Train a word TF-IDF + logistic regression baseline on a fixed holdout split."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

SEED = 42


def load_data(path, labeled=False):
    # Strings preserve IDs such as "001" and text such as "NA".
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = ['id', 'text', 'label'] if labeled else ['id', 'text']
    if list(data.columns) != expected or data.empty:
        raise ValueError(f'Expected nonempty data with columns {expected}')
    if data.isna().any().any() or data['id'].str.strip().eq('').any():
        raise ValueError('Missing values or empty IDs')
    if data['id'].duplicated().any() or data['text'].str.strip().eq('').any():
        raise ValueError('Duplicate IDs or empty passages')
    if labeled:
        if not data['label'].isin(['0', '1']).all():
            raise ValueError('Labels must be 0 (human) or 1 (AI)')
        data['label'] = data['label'].astype(int)
    return data


def split_data(data, path, fingerprint):
    # Normalize only for duplicate checks; keep the original text for modeling.
    normalized = data['text'].str.normalize('NFKC').str.casefold().str.split().str.join(' ')
    if normalized.duplicated().any():
        raise ValueError('Duplicate passages need review before splitting')
    if len(data) != 10000 or data['label'].value_counts().to_dict() != {0: 5000, 1: 5000}:
        raise ValueError('Expected 10,000 training rows, with 5,000 per class')
    data = data.sort_values('id').reset_index(drop=True)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['train_sha256'] != fingerprint or saved['seed'] != SEED:
            raise ValueError('Training file or split seed changed; check the saved split')
    else:
        _, validation = train_test_split(data, test_size=2000, random_state=SEED,
                                         stratify=data['label'])
        saved = {'seed': SEED, 'train_sha256': fingerprint,
                 'validation_ids': sorted(validation['id'].tolist())}
    ids = saved['validation_ids']
    if len(ids) != 2000 or len(set(ids)) != 2000 or not set(ids) <= set(data['id']):
        raise ValueError('Saved split must contain 2,000 unique training IDs')
    is_validation = data['id'].isin(ids)
    training, validation = data[~is_validation].copy(), data[is_validation].copy()
    if validation['label'].value_counts().to_dict() != {0: 1000, 1: 1000}:
        raise ValueError('Validation set must contain 1,000 examples of each class')
    if not path.exists():
        path.write_text(json.dumps(saved, indent=2) + '\n')
    return training, validation


def build_model():
    # Pipeline fits both vocabulary and IDF only on the training partition.
    return Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 1), min_df=2,
                                 max_df=1.0, max_features=100000, sublinear_tf=True,
                                 smooth_idf=True, norm='l2', stop_words=None,
                                 token_pattern=r'(?u)\b\w\w+\b')),
        ('classifier', LogisticRegression(C=1.0, penalty='l2', solver='liblinear',
                                          max_iter=2000, tol=0.0001, random_state=SEED)),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=Path(__file__).with_name('validation_ids.json'))
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/baseline'))
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError('Output directory is not empty; choose a new --out-dir')
    data = load_data(args.train, labeled=True)
    fingerprint = hashlib.sha256(args.train.read_bytes()).hexdigest()
    training, validation = split_data(data, args.split, fingerprint)
    model = build_model()
    start = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(training['text'], training['label'])
    seconds = time.perf_counter() - start
    ai_column = list(model.classes_).index(1)
    training_probs = model.predict_proba(training['text'])[:, ai_column]
    validation_probs = model.predict_proba(validation['text'])[:, ai_column]
    # Keep the settings and code revision alongside each local experiment.
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    metrics = {
        'train_rows': len(training), 'validation_rows': len(validation), 'seed': SEED,
        'train_auroc': roc_auc_score(training['label'], training_probs),
        'validation_auroc': roc_auc_score(validation['label'], validation_probs),
        'fit_seconds': seconds, 'vocabulary_size': len(model['tfidf'].vocabulary_),
        'train_sha256': fingerprint, 'split_sha256': hashlib.sha256(args.split.read_bytes()).hexdigest(),
        'code_commit': commit, 'working_tree_dirty': dirty, 'python': platform.python_version(),
        'tfidf_settings': model['tfidf'].get_params(),
        'classifier_settings': model['classifier'].get_params(),
    }
    # dtype is a Python class, so record its name for JSON output.
    metrics['tfidf_settings']['dtype'] = str(metrics['tfidf_settings']['dtype'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / 'model.joblib')
    predictions = validation[['id', 'label']].copy()
    predictions['probability_ai'] = validation_probs
    predictions.to_csv(args.out_dir / 'validation_predictions.csv', index=False)
    (args.out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    print(f"Training AUROC:   {metrics['train_auroc']:.6f}")
    print(f"Validation AUROC: {metrics['validation_auroc']:.6f}")
    print(f'Fit time: {seconds:.3f}s; saved model and results to {args.out_dir}')


if __name__ == '__main__':
    main()
