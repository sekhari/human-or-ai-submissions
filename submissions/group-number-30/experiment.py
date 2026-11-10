"""Measured evaluation, artifacts and paired uncertainty on the fixed holdout."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from models import SEED, build_model

SCHEMA = {'id': 'id', 'text': 'text', 'label': 'label'}
THREADS = 2


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def ids_hash(frame):
    return hashlib.sha256(json.dumps(frame.id.astype(str).tolist(), separators=(',', ':')).encode()).hexdigest()


def implementation_hash():
    digest = hashlib.sha256()
    names = ['models.py', 'structure.py', 'experiment.py', 'data_utils.py']
    if (Path(__file__).parent / 'ensemble.py').is_file():
        names.append('ensemble.py')
    for name in names:
        path = Path(__file__).parent / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.is_file() else b'<absent>')
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tagger_resource_identity():
    """Hash installed tagger bytes, with no machine-specific paths or download."""
    import nltk.data
    resource = 'averaged_perceptron_tagger_eng'
    digest = hashlib.sha256()
    files = {}
    for suffix in ('classes', 'tagdict', 'weights'):
        name = f'{resource}.{suffix}.json'
        try:
            pointer = nltk.data.find(f'taggers/{resource}/{name}')
        except LookupError:
            return None
        with pointer.open() as handle:
            contents = handle.read()
        if isinstance(contents, str):
            contents = contents.encode('utf-8')
        files[name] = hashlib.sha256(contents).hexdigest()
        digest.update(name.encode())
        digest.update(contents)
    return {'resource': resource, 'sha256': digest.hexdigest(), 'files': files}


def runtime_identity():
    return {'python': platform.python_version(),
            'versions': {name: importlib.metadata.version(name) for name in
                         ('numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'joblib', 'nltk')},
            'nltk_tagger_resource': tagger_resource_identity()}


def verify_environment(previous, current):
    """Reject mixed dependency/resource environments while allowing path moves."""
    keys = ('python', 'versions', 'nltk_tagger_resource')
    mismatches = [key for key in keys if key not in previous or previous[key] != current[key]]
    if mismatches:
        raise ValueError('Resume environment differs or lacks provenance for: ' + ', '.join(mismatches) +
                         '. Reproduce the recorded environment or use a new results directory.')


def probability_ai(model, texts):
    classes = np.asarray(model.classes_)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(f'Expected classes {{0,1}}, received {classes.tolist()}')
    probability = np.asarray(model.predict_proba(list(texts)))[:, int(np.flatnonzero(classes == 1)[0])]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Model produced invalid AI probabilities')
    return probability


def artifact(model, config, frame, *, fit_scope, seed=SEED):
    return {'artifact_version': 1, 'model': model, 'config': config, 'schema': SCHEMA,
            'fit_scope': fit_scope, 'seed': seed, 'train_rows': len(frame),
            'source_sha256': frame.attrs.get('source_sha256'), 'train_ids_sha256': ids_hash(frame),
            'implementation_sha256': implementation_hash(), 'runtime': runtime_identity()}


def evaluate(config, train, validation, output, name, *, save_model=False, resume=False,
             sensitivity_exclude_ids=None, seed=SEED):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics_file = output / f'{name}.json'
    prediction_file = output / f'{name}.predictions.tsv'
    model_file = output / f'{name}.joblib'
    exclusions = sorted(set(map(str, sensitivity_exclude_ids or [])))
    identity = {'config': config, 'seed': seed, 'source_sha256': train.attrs.get('source_sha256'),
                'train_ids_sha256': ids_hash(train), 'validation_ids_sha256': ids_hash(validation),
                'train_rows': len(train), 'validation_rows': len(validation),
                'sensitivity_excluded_validation_ids': exclusions,
                'implementation_sha256': implementation_hash(),
                'sklearn_version': importlib.metadata.version('scikit-learn'),
                'runtime': runtime_identity()}
    if train.attrs.get('source_sha256') != validation.attrs.get('source_sha256'):
        raise ValueError('Training and validation must originate from the same labeled file')
    if metrics_file.exists():
        previous = json.loads(metrics_file.read_text())
        if not resume:
            raise FileExistsError(f'Refusing to overwrite {metrics_file}; use --resume for an identical recorded run')
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError(f'Existing run {name} has different data/configuration; use a new results directory')
        if not prediction_file.is_file() or (save_model and not model_file.is_file()):
            raise ValueError(f'Existing run {name} is incomplete; use a new results directory')
        predictions = pd.read_csv(prediction_file, sep='\t', dtype={'id': str})
        if list(predictions.columns) != ['id', 'label', 'probability_ai']:
            raise ValueError(f'Existing predictions have an invalid schema for {name}')
        if predictions.id.tolist() != validation.id.astype(str).tolist() or predictions.label.tolist() != validation.label.tolist():
            raise ValueError(f'Existing predictions do not match validation IDs/labels for {name}')
        probability = predictions.probability_ai.to_numpy(dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f'Existing predictions contain invalid probabilities for {name}')
        if not np.isclose(roc_auc_score(predictions.label, probability), previous['validation_auroc'], rtol=0, atol=1e-12):
            raise ValueError(f'Existing predictions disagree with recorded AUROC for {name}')
        if not np.isclose(log_loss(predictions.label, probability, labels=[0, 1]),
                          previous['validation_log_loss'], rtol=0, atol=1e-12):
            raise ValueError(f'Existing predictions disagree with recorded log loss for {name}')
        if previous.get('predictions_sha256') and previous['predictions_sha256'] != _file_sha256(prediction_file):
            raise ValueError(f'Existing prediction file checksum changed for {name}')
        if model_file.is_file():
            if previous.get('model_sha256') and previous['model_sha256'] != _file_sha256(model_file):
                raise ValueError(f'Existing model file checksum changed for {name}')
            saved = joblib.load(model_file)  # Trusted, locally produced artifacts only.
            expected = {'artifact_version': 1, 'schema': SCHEMA, 'fit_scope': 'training_partition',
                        'config': config, 'seed': seed, 'train_rows': len(train),
                        'source_sha256': identity['source_sha256'],
                        'train_ids_sha256': identity['train_ids_sha256'],
                        'implementation_sha256': identity['implementation_sha256']}
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Existing model metadata does not match recorded run {name}')
            if 'runtime' in saved and saved['runtime'] != identity['runtime']:
                raise ValueError(f'Existing model runtime does not match recorded run {name}')
            if 'model' not in saved or set(np.asarray(saved['model'].classes_).tolist()) != {0, 1}:
                raise ValueError(f'Existing model lacks the expected binary classifier for {name}')
        print(f'Reusing verified run {name}', flush=True)
        return previous
    if prediction_file.exists() or model_file.exists():
        raise FileExistsError(f'Partial files already exist for {name}; use a new results directory')
    model = build_model(config, seed=seed)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=THREADS):
        warnings.simplefilter('always')
        start = time.perf_counter()
        model.fit(train.text.tolist(), train.label.to_numpy())
        fit_seconds = time.perf_counter() - start
        pred_start = time.perf_counter()
        train_probability = probability_ai(model, train.text)
        val_probability = probability_ai(model, validation.text)
        predict_seconds = time.perf_counter() - pred_start
    record = {'name': name, **identity,
              'train_auroc': float(roc_auc_score(train.label, train_probability)),
              'validation_auroc': float(roc_auc_score(validation.label, val_probability)),
              'validation_log_loss': float(log_loss(validation.label, val_probability, labels=[0, 1])),
              'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
              'warnings': [f'{item.category.__name__}: {item.message}' for item in caught]}
    if sensitivity_exclude_ids is not None:
        keep = ~validation.id.astype(str).isin(exclusions).to_numpy()
        labels = validation.label.to_numpy()[keep]
        record['sensitivity_rows'] = int(keep.sum())
        record['sensitivity_auroc'] = (float(roc_auc_score(labels, val_probability[keep]))
                                       if len(set(labels.tolist())) == 2 else None)
        record['sensitivity_note'] = ('Excluded validation IDs flagged by the fixed full-training source-overlap audit; '
                                      'this is a diagnostic only, not the tuning metric.')
    if config['kind'] == 'surprisal_lr':
        record['training_score_note'] = ('Classifier fit uses out-of-fold surprisal; reported training AUROC '
            'uses final tag models fit on all training rows, so it is optimistic resubstitution performance.')
    if config['kind'] == 'word_svm':
        record['calibration_note'] = ('Three seeded stratified training-only folds calibrate LinearSVC with sigmoid; '
            'internal folds share vocabulary/IDF fit on the outer training partition. No outer validation rows enter fit.')
    pd.DataFrame({'id': validation.id.astype(str).tolist(), 'label': validation.label.to_numpy(),
                  'probability_ai': val_probability}).to_csv(prediction_file, sep='\t', index=False)
    record['predictions_sha256'] = _file_sha256(prediction_file)
    if save_model:
        temporary = model_file.with_name(model_file.name + '.tmp')
        joblib.dump(artifact(model, config, train, fit_scope='training_partition', seed=seed), temporary, compress=3)
        temporary.replace(model_file)
        record['model_sha256'] = _file_sha256(model_file)
    # Metrics are the completion marker after other artifacts succeed.
    write_json(metrics_file, record)
    print(json.dumps(record), flush=True)
    return record


def paired_bootstrap(labels, baseline, candidate, *, repeats=1000, seed=SEED):
    y, a, b = map(np.asarray, (labels, baseline, candidate))
    if len(y) != len(a) or len(y) != len(b) or set(y.tolist()) != {0, 1}:
        raise ValueError('Paired bootstrap requires aligned binary labels and predictions')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or repeats < 1:
        raise ValueError('Bootstrap predictions must be finite and repeats positive')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    changes = []
    for _ in range(repeats):
        index = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        changes.append(roc_auc_score(y[index], b[index]) - roc_auc_score(y[index], a[index]))
    return {'delta_auroc': float(roc_auc_score(y, b) - roc_auc_score(y, a)),
            'ci_95_percentile': np.quantile(changes, [.025, .975]).tolist(), 'bootstrap_repeats': repeats,
            'seed': seed, 'caution': 'Exploratory paired stratified percentile interval, conditional on fitted models; '
            'does not include training variability or correct for selection/multiple comparisons.'}
