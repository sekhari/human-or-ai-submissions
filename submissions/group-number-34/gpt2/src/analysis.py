"""Numerical metrics and paired diagnostics for saved classifier predictions."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

FORMATS = ('original', 'newlines_only', 'all_whitespace')
LENGTH_BINS = (('Short: <=150 words', 0, 150),
               ('Medium: 151-250 words', 151, 250),
               ('Long: >250 words', 251, None))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def classification_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or labels.shape != probabilities.shape or not len(labels):
        raise ValueError('Labels and probabilities must be nonempty aligned vectors.')
    if not set(labels).issubset({0, 1}):
        raise ValueError('Labels must be 0 or 1.')
    labels = labels.astype(int)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities must be finite values from zero to one.')
    predicted = probabilities >= threshold
    human = labels == 0
    ai = labels == 1
    fp = int((human & predicted).sum())
    fn = int((ai & ~predicted).sum())
    return {'rows': len(labels), 'human_rows': int(human.sum()), 'ai_rows': int(ai.sum()),
            'auc': float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
            'bce': float(log_loss(labels, probabilities, labels=[0, 1])),
            'brier_score': float(np.mean((probabilities - labels) ** 2)),
            'threshold': float(threshold), 'true_negatives': int((human & ~predicted).sum()),
            'false_positives': fp, 'false_negatives': fn, 'true_positives': int((ai & predicted).sum()),
            'errors': fp + fn, 'accuracy': 1.0 - (fp + fn) / len(labels),
            'human_fpr': fp / int(human.sum()) if human.any() else None,
            'ai_fnr': fn / int(ai.sum()) if ai.any() else None}


def auc_interval(labels, probabilities, replicates=1000, seed=42):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if set(labels) != {0, 1}:
        return {'lower': None, 'upper': None, 'replicates': 0, 'seed': seed}
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    values = []
    for _ in range(replicates):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        values.append(roc_auc_score(labels[indices], probabilities[indices]))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {'lower': float(lower), 'upper': float(upper), 'replicates': replicates, 'seed': seed}


def paired_auc_interval(labels, baseline, candidate, replicates=2000, seed=42):
    labels = np.asarray(labels)
    baseline, candidate = np.asarray(baseline), np.asarray(candidate)
    classification_metrics(labels, baseline)
    classification_metrics(labels, candidate)
    if set(labels) != {0, 1}:
        raise ValueError('Paired AUROC comparison requires both label classes.')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    values = []
    for _ in range(replicates):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        values.append(roc_auc_score(labels[indices], candidate[indices])
                      - roc_auc_score(labels[indices], baseline[indices]))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {'lower': float(lower), 'upper': float(upper), 'replicates': replicates, 'seed': seed}


def transform_text(text, variant):
    if variant == 'original':
        return text
    if variant == 'newlines_only':
        return re.sub(r'[\r\n]+', ' ', text)
    if variant == 'all_whitespace':
        return re.sub(r'\s+', ' ', text).strip()
    raise ValueError(f'Unknown text format: {variant}')


def read_predictions(path):
    frame = pd.read_csv(path, dtype={'id': str}, float_precision='round_trip')
    if list(frame.columns) != ['id', 'label', 'probability'] or frame.id.duplicated().any():
        raise ValueError('Prediction CSV columns must be id,label,probability with unique IDs.')
    classification_metrics(frame.label, frame.probability)
    return frame


def aligned_predictions(frame, reference):
    if set(frame.id) != set(reference.id):
        raise ValueError('Prediction files must contain the same IDs.')
    frame = frame.set_index('id').loc[reference.id].reset_index()
    if not np.array_equal(frame.label.to_numpy(), reference.label.to_numpy()):
        raise ValueError('Prediction files must use the same labels.')
    return frame


def length_subgroups(frame, replicates=1000, seed=42):
    counts = frame.text.map(lambda text: len(text.split()))
    records = []
    for index, (name, lower, upper) in enumerate(LENGTH_BINS):
        mask = counts >= lower
        if upper is not None:
            mask &= counts <= upper
        group = frame.loc[mask]
        if group.empty:
            records.append({'subgroup': name, 'rows': 0, 'human_rows': 0, 'ai_rows': 0,
                            'auc': None, 'auc_ci_lower': None, 'auc_ci_upper': None})
            continue
        metrics = classification_metrics(group.label, group.probability)
        interval = auc_interval(group.label, group.probability, replicates, seed + index)
        records.append({'subgroup': name, **metrics, 'auc_ci_lower': interval['lower'],
                        'auc_ci_upper': interval['upper']})
    return pd.DataFrame(records)


def error_cases(frame, threshold):
    cases = frame.copy()
    predicted = cases.probability >= threshold
    cases['error_type'] = np.where((cases.label == 0) & predicted, 'FP',
                                  np.where((cases.label == 1) & ~predicted, 'FN', ''))
    cases = cases.loc[cases.error_type != ''].copy()
    cases['threshold'] = threshold
    cases['error_confidence'] = np.where(cases.error_type == 'FP', cases.probability,
                                        1 - cases.probability)
    return cases.sort_values(['error_type', 'error_confidence', 'id'],
                             ascending=[True, False, True], kind='stable').reset_index(drop=True)


def compare_formats(baseline, candidate, replicates=2000, seed=42):
    reference = candidate['original']
    baseline = {name: aligned_predictions(frame, reference) for name, frame in baseline.items()}
    candidate = {name: aligned_predictions(frame, reference) for name, frame in candidate.items()}
    records = []
    for index, name in enumerate(FORMATS):
        labels = reference.label.to_numpy()
        a, b = baseline[name].probability.to_numpy(), candidate[name].probability.to_numpy()
        ma, mb = classification_metrics(labels, a), classification_metrics(labels, b)
        interval = paired_auc_interval(labels, a, b, replicates, seed + index)
        record = {'format': name, 'rows': len(reference), 'baseline_auc': ma['auc'],
                  'candidate_auc': mb['auc'], 'delta_auc': mb['auc'] - ma['auc'],
                  'delta_auc_ci_lower': interval['lower'], 'delta_auc_ci_upper': interval['upper'],
                  'bootstrap_replicates': replicates}
        for arm, measures, values, original in (
            ('baseline', ma, a, baseline['original'].probability.to_numpy()),
            ('candidate', mb, b, candidate['original'].probability.to_numpy()),
        ):
            for key in ['bce', 'brier_score', 'false_positives', 'false_negatives', 'accuracy']:
                record[f'{arm}_{key}'] = measures[key]
            record[f'{arm}_mean_probability_change'] = float(np.abs(values - original).mean())
            record[f'{arm}_threshold_flips'] = int(((values >= 0.5) != (original >= 0.5)).sum())
        records.append(record)
    return pd.DataFrame(records)
