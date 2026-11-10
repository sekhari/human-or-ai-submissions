"""Group 38 Model B: character TF-IDF + linear SVM (Mia).
Run from the team folder; uses Constance's exact shared split when available.
"""
import argparse
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42


def load_and_split(zip_path):
    # Keep the team's loading and splitting code as the source of truth.
    from data_utils import load_data, split_data
    train, _, test = load_data(zip_path)
    for name, frame, expected in [('train', train, 10000), ('test', test, 3000)]:
        if len(frame) != expected or not {'id', 'text'}.issubset(frame.columns):
            raise ValueError(f'Unexpected {name} shape/columns: {frame.shape}')
        if frame['id'].duplicated().any() or frame[['id', 'text']].isna().any().any():
            raise ValueError(f'{name} contains missing values or duplicate IDs')
        if not frame['text'].map(lambda x: isinstance(x, str)).all():
            raise ValueError(f'{name} contains non-string text')
    if set(train['label'].unique()) != {0, 1}:
        raise ValueError('Expected binary labels 0=human, 1=AI')
    fit, val = split_data(train)
    if (len(fit), len(val)) != (8000, 2000):
        raise ValueError('Expected the shared 8000/2000 split')
    if set(fit.id) & set(val.id):
        raise ValueError('Training and validation IDs overlap')
    overlap = len(set(fit.text) & set(val.text))
    if overlap:
        warnings.warn(f'{overlap} exact texts overlap across the shared split; disclose this.')
    return train, test, fit, val, overlap


def vectorizer(ngram):
    return TfidfVectorizer(analyzer='char', ngram_range=tuple(ngram),
                           lowercase=True, min_df=2, max_features=100000,
                           sublinear_tf=True, dtype=np.float32)


def svm(c):
    return LinearSVC(C=float(c), dual=True, random_state=SEED,
                     max_iter=10000, tol=1e-4)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train, test, fit, val, overlap = load_and_split(args.data)
    pd.DataFrame({'id': fit.id, 'partition': 'train'}).to_csv(out/'train_ids.csv', index=False)
    pd.DataFrame({'id': val.id, 'partition': 'validation'}).to_csv(out/'validation_ids.csv', index=False)
    configs = [(3, 5), (3, 6), (4, 6)]
    c_values = [0.1, 1.0, 10.0, 100.0]
    rows = []
    # Fit each vocabulary/IDF on training text only; reuse it across C values.
    for ngram in configs:
        start = time.perf_counter()
        tfidf = vectorizer(ngram)
        x_fit = tfidf.fit_transform(fit.text)
        feature_seconds = time.perf_counter() - start
        x_val = tfidf.transform(val.text)
        for c in c_values:
            model = svm(c)
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(x_fit, fit.label)
            svm_seconds = time.perf_counter() - start
            row = dict(ngram_min=ngram[0], ngram_max=ngram[1], C=c,
                       features=x_fit.shape[1],
                       train_auroc=roc_auc_score(fit.label, model.decision_function(x_fit)),
                       validation_auroc=roc_auc_score(val.label, model.decision_function(x_val)),
                       feature_fit_seconds=feature_seconds, svm_fit_seconds=svm_seconds,
                       total_fit_seconds=feature_seconds + svm_seconds,
                       converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught))
            rows.append(row)
            pd.DataFrame(rows).to_csv(out/'tuning_results.csv', index=False)
            print(json.dumps(row), flush=True)
    results = pd.DataFrame(rows)
    valid = results[results.converged]
    if valid.empty:
        raise RuntimeError('No configuration converged; increase max_iter before selecting a model')
    # Deterministic tie break: smaller C, then shorter n-grams.
    best = valid.sort_values(['validation_auroc', 'C', 'ngram_max', 'ngram_min'],
                             ascending=[False, True, True, True]).iloc[0]
    selected = {'ngram_range': [int(best.ngram_min), int(best.ngram_max)], 'C': float(best.C)}
    pipeline = Pipeline([('tfidf', vectorizer(selected['ngram_range'])), ('svm', svm(selected['C']))])
    start = time.perf_counter()
    pipeline.fit(fit.text, fit.label)
    refit_seconds = time.perf_counter()-start
    margins = pipeline.decision_function(val.text)
    predictions = pd.DataFrame({'id': val.id, 'label': val.label, 'decision_score': margins,
                                'prediction': (margins >= 0).astype(int)})
    predictions.to_csv(out/'validation_predictions.csv', index=False)
    summary = dict(selected=selected, seed=SEED, train_rows=len(fit), validation_rows=len(val),
                   train_auroc=roc_auc_score(fit.label, pipeline.decision_function(fit.text)),
                   validation_auroc=roc_auc_score(val.label, margins),
                   selected_refit_seconds=refit_seconds, exact_text_overlap=overlap,
                   python=platform.python_version(),
                   versions={p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib']})
    joblib.dump(pipeline, out/'character_svm.joblib')
    # AUROC uses margins directly. Calibration is optional and never fits validation labels.
    if args.probabilities:
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        calibrated = CalibratedClassifierCV(estimator=pipeline, method='sigmoid', cv=folds, ensemble=True)
        start = time.perf_counter()
        calibrated.fit(fit.text, fit.label)
        probs = calibrated.predict_proba(val.text)[:, list(calibrated.classes_).index(1)]
        predictions['probability_ai'] = probs
        predictions.to_csv(out/'validation_predictions.csv', index=False)
        summary['calibrated_validation_auroc'] = roc_auc_score(val.label, probs)
        summary['calibration_fit_seconds'] = time.perf_counter()-start
        # Refit on all labels only after evaluation; each fold has its own TF-IDF pipeline.
        calibrated.fit(train.text, train.label)
        probs_test = calibrated.predict_proba(test.text)[:, list(calibrated.classes_).index(1)]
        submission = pd.DataFrame({'id': test.id, 'label': probs_test})
        if not np.isfinite(probs_test).all() or not submission.label.between(0,1).all():
            raise ValueError('Invalid test probabilities')
        submission.to_csv(out/'submission_character_svm.csv', index=False)
        joblib.dump(calibrated, out/'character_svm_calibrated_full.joblib')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip')
    parser.add_argument('--output', default='results/character_svm')
    parser.add_argument('--probabilities', action='store_true', help='Also evaluate calibrated probabilities and generate test CSV')
    run(parser.parse_args())
