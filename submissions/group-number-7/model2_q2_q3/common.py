"""Shared official-data, reproducibility and evaluation helpers for Model 2 Q2/Q3."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import unicodedata
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from model2.data import read_data, partition
from model2.model import build_calibrated_model, build_pipeline

HERE = Path(__file__).resolve().parent
SIZES = (500, 1000, 2000, 5000, 8000)
SEED = 42
THREADS = 1
CONFIDENCE = 0.95
SIMILARITY = 0.9
HASHES = {'train.csv': '325e345faf88fb1c4203278098c282e40319a87b4bf17e094d2f235f51d0c07e',
          'unlabeled.csv': '42fe881240148a2e06583692ff16ded3df5bd60558820738b2de470680934aae'}
SPLIT_HASH = '15340e28ee050a0023b017abeddd2d3b72b7fc05612b9d299b59d88d31a1cb2b'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {n: importlib.metadata.version(n) for n in
                         ['numpy', 'scipy', 'pandas', 'scikit-learn', 'joblib', 'matplotlib', 'threadpoolctl']}}


def config():
    return json.loads((HERE / 'config/selected.json').read_text())


def source_hashes():
    names = ['common.py', 'run_q2.py', 'run_q3.py', 'verify.py', 'requirements.txt',
             'model2/model.py', 'model2/data.py', 'config/selected.json',
             'config/shared_split.csv', 'EXPERIMENT_PROTOCOL.json']
    return {n: digest(HERE / n) for n in names if (HERE / n).is_file()}


def load_inputs(data_dir, include_unlabeled=False):
    data_dir = Path(data_dir)
    names = ['train.csv'] + (['unlabeled.csv'] if include_unlabeled else [])
    frames = {}
    for name in names:
        require(digest(data_dir / name) == HASHES[name], f'Official input hash differs: {name}')
        frames[name] = read_data(data_dir / name, labeled=name == 'train.csv',
                                 expected_rows=10000 if name == 'train.csv' else 3000)
    split_path = HERE / 'config/shared_split.csv'
    require(digest(split_path) == SPLIT_HASH, 'Canonical split changed.')
    raw = frames['train.csv']
    train, val = partition(raw, pd.read_csv(split_path, dtype=str))
    require(train.label.value_counts().to_dict() == {0: 4000, 1: 4000}, 'Wrong train balance.')
    require(val.label.value_counts().to_dict() == {0: 1000, 1: 1000}, 'Wrong validation balance.')
    expected_train, expected_val = train_test_split(raw, test_size=.2, random_state=SEED, stratify=raw.label)
    require(set(expected_train.id) == set(train.id) and set(expected_val.id) == set(val.id),
            'Model 2 membership differs from the agreed original-order split.')
    # Model 2 retains CSV order within partitions. This also fixes its calibration folds.
    require(not set(train.id) & set(val.id), 'Train/validation overlap.')
    if include_unlabeled:
        u = frames['unlabeled.csv']
        require(not set(u.id) & set(raw.id), 'Labeled/unlabeled ID overlap.')
        return train, val, u
    return train, val


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def near_reference(reference, query):
    """Same diagnostic as teammate: hashed word-trigram cosine, not exhaustive deduplication."""
    hasher = HashingVectorizer(n_features=2**18, analyzer='word', ngram_range=(3, 3),
                              alternate_sign=False, norm='l2', dtype=np.float32)
    xr, xq = hasher.transform(reference.text), hasher.transform(query.text)
    maxima, nearest = [], []
    for start in range(0, len(query), 128):
        scores = (xq[start:start + 128] @ xr.T).toarray()
        positions = scores.argmax(axis=1)
        maxima.extend(scores[np.arange(len(positions)), positions].tolist())
        nearest.extend(reference.id.iloc[positions].tolist())
    return pd.DataFrame({'id': query.id.to_numpy(), 'nearest_reference_id': nearest,
                         'word_trigram_cosine': maxima})


def subsets(train):
    rng = np.random.default_rng(SEED)
    positions = [rng.permutation(np.flatnonzero(train.label.to_numpy() == c)) for c in (0, 1)]
    for size in SIZES:
        selected = np.sort(np.concatenate([p[:size // 2] for p in positions]))
        yield size, train.iloc[selected]


def fit_model(frame, calibrated=True):
    model = build_calibrated_model(config()) if calibrated else build_pipeline(config())
    with threadpool_limits(limits=THREADS), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        started = time.perf_counter()
        model.fit(frame.text, frame.label)
        seconds = time.perf_counter() - started
    require(not any(issubclass(w.category, ConvergenceWarning) for w in caught),
            'SVM did not converge; experiment is incomplete.')
    require(model.classes_.tolist() == [0, 1], 'Expected Human=0 / AI=1.')
    return model, seconds, [str(w.message) for w in caught]


def final_pipeline(model, calibrated=True):
    if not calibrated:
        return model
    require(len(model.calibrated_classifiers_) == 1, 'Expected ensemble=False.')
    return model.calibrated_classifiers_[0].estimator


def evaluate(model, frame, calibrated=True):
    pipe = final_pipeline(model, calibrated)
    with threadpool_limits(limits=THREADS):
        margin = pipe.decision_function(frame.text)
        probability = model.predict_proba(frame.text)[:, 1] if calibrated else None
    require(margin.shape == (len(frame),) and np.isfinite(margin).all(), 'Invalid margins.')
    prediction = pd.DataFrame({'id': frame.id.to_numpy(), 'label': frame.label.to_numpy(), 'margin': margin})
    auc = float(roc_auc_score(frame.label, margin))
    if probability is not None:
        require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
                'Invalid probabilities.')
        prediction['probability'] = probability
        require(abs(float(roc_auc_score(frame.label, probability)) - auc) < 1e-12,
                'Probability/margin rankings differ; inspect before making comparable AUROC claims.')
    return prediction, auc


def new_run(path):
    path = Path(path).resolve()
    require(not path.exists(), f'Output directory already exists: {path}')
    path.mkdir(parents=True)
    return path


def save_fitted(out, model, frames, fit_seconds, fit_warnings, calibrated=True, sensitivity_ids=()):
    joblib.dump(model, out / 'model.joblib', compress=3)
    metrics = {'fit_seconds': fit_seconds, 'warnings': fit_warnings,
               'model_sha256': digest(out / 'model.joblib'),
               'feature_count': len(final_pipeline(model, calibrated)['tfidf'].vocabulary_)}
    for name, frame in frames.items():
        pred, auc = evaluate(model, frame, calibrated)
        pred.to_csv(out / f'{name}_predictions.csv', index=False)
        metrics[f'{name}_rows'] = len(frame)
        metrics[f'{name}_auroc'] = auc
        if name == 'validation':
            retained = pred[~pred.id.isin(sensitivity_ids)]
            metrics['validation_sensitivity_rows'] = len(retained)
            metrics['validation_sensitivity_auroc'] = float(roc_auc_score(retained.label, retained.margin))
    write_json(out / 'metrics.json', metrics)
    return metrics


def run_metadata():
    return {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'model': 'Model 2 selected configuration',
            'config': config(), 'environment': environment(), 'source_sha256': source_hashes(),
            'data_sha256': HASHES, 'split_sha256': SPLIT_HASH, 'thread_limit': THREADS,
            'validation_labels_used_for_training_or_pseudo_selection': False,
            'test_csv_used': False}
