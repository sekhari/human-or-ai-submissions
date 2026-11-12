"""Q5: compare word and character features on the shared validation split."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from train import SEED, build_model, load_data, split_data

CANDIDATES = ('word', 'word_bigrams', 'character', 'combined')


def build_candidate(name):
    if name not in CANDIDATES:
        raise ValueError(f'Unknown candidate: {name}')
    baseline = build_model().set_params(classifier__C=10.0)
    if name == 'word':
        return baseline
    words = baseline['tfidf'].set_params(ngram_range=(1, 2))
    chars = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), lowercase=True,
                           min_df=2, max_features=100000, sublinear_tf=True,
                           smooth_idf=True, norm='l2')
    if name == 'word_bigrams':
        features = words
    elif name == 'character':
        features = chars
    else:
        # Equal-weight blocks, then unit length to keep the classifier scale comparable.
        features = Pipeline([('union', FeatureUnion([('words', words), ('chars', chars)])),
                             ('normalize', Normalizer())])
    return Pipeline([('features', features), ('classifier', baseline['classifier'])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--tune', action='store_true', help='Small two-stage character-model search')
    parser.add_argument('--split', type=Path, default=Path(__file__).with_name('validation_ids.json'))
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/q5'))
    args = parser.parse_args()
    if args.tune and args.out_dir == Path('outputs/q5'):
        args.out_dir = Path('outputs/q5_tuning')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Output directory is not empty; choose a new --out-dir')
    data = load_data(args.train, labeled=True)
    fingerprint = hashlib.sha256(args.train.read_bytes()).hexdigest()
    training, validation = split_data(data, args.split, fingerprint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        'seed': SEED, 'train_sha256': fingerprint,
        'split_sha256': hashlib.sha256(args.split.read_bytes()).hexdigest(),
        'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'platform': platform.platform(), 'train_rows': len(training),
        'validation_rows': len(validation), 'native_threads': 1,
        'base_C': 10.0, 'candidates': 'two-stage character search' if args.tune else list(CANDIDATES),
        'tuning': args.tune,
        'tuning_protocol': 'C in [0.1,1,3,10,30] at char 3-5 lowercase; then 2-5, 3-6, and case-preserving 3-5 at best C',
    }
    (args.out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    results = []
    best_score = -1
    runs = [(name, None) for name in CANDIDATES]
    if args.tune:
        runs = [(f'char_C{c:g}', {'classifier__C': c}) for c in (0.1, 1.0, 3.0, 10.0, 30.0)]
    for name, settings in runs:
        print(f'Fitting {name}...', flush=True)
        model = build_candidate('character' if args.tune else name)
        if settings:
            model.set_params(**settings)
        start = time.perf_counter()
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(training['text'], training['label'])
            fit_seconds = time.perf_counter() - start
            ai_column = list(model.classes_).index(1)
            train_probs = model.predict_proba(training['text'])[:, ai_column]
            val_probs = model.predict_proba(validation['text'])[:, ai_column]
        if not np.isfinite(val_probs).all() or not ((val_probs >= 0) & (val_probs <= 1)).all():
            raise ValueError('Invalid validation probabilities')
        row = {'candidate': name, 'train_auroc': roc_auc_score(training['label'], train_probs),
               'validation_auroc': roc_auc_score(validation['label'], val_probs),
               'fit_seconds': fit_seconds,
               'fit_and_score_seconds': time.perf_counter() - start,
               'features': model['classifier'].coef_.shape[1],
               'settings': settings or {}}
        predictions = validation[['id', 'label']].copy()
        predictions['probability_ai'] = val_probs
        predictions.to_csv(args.out_dir / f'{name}_validation_predictions.csv', index=False)
        # Save baseline and current winner for local analysis / the existing predict.py.
        if name == 'word':
            joblib.dump(model, args.out_dir / 'baseline_model.joblib')
        if row['validation_auroc'] > best_score:
            best_score = row['validation_auroc']
            joblib.dump(model, args.out_dir / 'best_model.joblib')
            (args.out_dir / 'best_candidate.json').write_text(json.dumps(row, indent=2) + '\n')
        results.append(row)
        pd.DataFrame(results).to_csv(args.out_dir / 'results.csv', index=False)
        print(json.dumps(row), flush=True)
        if args.tune and len(results) == 5:
            # Freeze C before checking just three representation changes.
            chosen_c = max(results, key=lambda r: r['validation_auroc'])['settings']['classifier__C']
            runs.extend([
                ('char_2_5', {'classifier__C': chosen_c, 'features__ngram_range': (2, 5)}),
                ('char_3_6', {'classifier__C': chosen_c, 'features__ngram_range': (3, 6)}),
                ('char_keep_case', {'classifier__C': chosen_c, 'features__lowercase': False}),
            ])


if __name__ == '__main__':
    main()
