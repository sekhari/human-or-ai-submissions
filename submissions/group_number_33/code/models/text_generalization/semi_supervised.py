"""Two fixed, single-round pseudo-label experiments using the style HGB31.

Only protocol-training labels are inputs to teacher/student training. Validation
labels are consumed exclusively after fitting for AUROC. Additional examples
come from unlabeled.csv, after exact-text hash exclusion against train.csv's
text column. No test.csv content is read by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from models.text_generalization.common import OUT, ROOT, decision_scores, load_protocol, sha256
from models.text_generalization.style_search import (
    StyleFeatures, build_pipeline, candidate_configs,
)


STYLE_CONFIG = next(config for config in candidate_configs() if config['name'] == 'style_hgb31')
PSEUDO_WEIGHT = 0.25
CONFIGS = [
    {'name': 'semi_style_hgb31_t095', 'threshold': 0.95, 'pseudo_weight': PSEUDO_WEIGHT},
    {'name': 'semi_style_hgb31_t099', 'threshold': 0.99, 'pseudo_weight': PSEUDO_WEIGHT},
]


def text_sha256(text) -> str:
    """Hash exact original text, without whitespace/case normalization."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        text = ''
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def exclude_labeled_text_duplicates(unlabeled: pd.DataFrame, labeled_texts):
    """Remove exact-text collisions; no labels or source IDs are inspected."""
    hashes = {text_sha256(text) for text in labeled_texts}
    collision = unlabeled.text.map(text_sha256).isin(hashes)
    return (
        unlabeled.loc[~collision].reset_index(drop=True),
        unlabeled.loc[collision].reset_index(drop=True),
    )


def load_unlabeled_pool():
    """Reusable final-fit loader: reads only labeled TEXT for deduplication."""
    unlabeled = pd.read_csv(ROOT / 'data/unlabeled.csv', keep_default_na=False)
    labeled_texts = pd.read_csv(ROOT / 'data/train.csv', usecols=['text'], keep_default_na=False).text
    eligible, removed = exclude_labeled_text_duplicates(unlabeled, labeled_texts)
    audit = {
        'unlabeled_sha256': sha256(ROOT / 'data/unlabeled.csv'),
        'labeled_text_rows_checked': len(labeled_texts),
        'unlabeled_rows_before_exclusion': len(unlabeled),
        'excluded_exact_text_matches': len(removed),
        'excluded_ids': removed.id.tolist(),
        'eligible_unlabeled_rows': len(eligible),
        'within_eligible_exact_text_duplicate_count': int(eligible.text.map(text_sha256).duplicated().sum()),
        'deduplication': 'SHA256 of exact original UTF-8 text; labeled text column only; no normalization',
    }
    return eligible, audit


def select_pseudo_labels(teacher, unlabeled_features, threshold: float):
    """Select once using teacher confidence; API has no validation inputs."""
    if not 0.5 < threshold <= 1.0:
        raise ValueError('threshold must be in (0.5, 1]')
    if not np.array_equal(teacher.classes_, [0, 1]):
        raise ValueError('Expected binary classes [0, 1]')
    if len(unlabeled_features):
        p1 = teacher.predict_proba(unlabeled_features)[:, 1]
    else:
        p1 = np.zeros(0, dtype=float)
    confidence = np.maximum(p1, 1.0 - p1)
    selected = np.flatnonzero(confidence >= threshold)
    return {
        'selected_indices': selected,
        'labels': (p1[selected] >= 0.5).astype(np.int64),
        'confidence': confidence[selected],
        'positive_probability': p1[selected],
    }


def fit_self_training_from_features(train_features, train_labels, unlabeled_features,
                                    threshold: float, teacher=None):
    """Fit fixed HGB31 on true examples plus one round of weight-0.25 pseudos.

    The optional teacher must have been fitted only on these training examples.
    This feature-level entrypoint permits cached, stateless style extraction.
    Validation features/labels are deliberately absent from the interface.
    """
    train_labels = np.asarray(train_labels)
    if teacher is None:
        teacher = build_pipeline(STYLE_CONFIG).named_steps['classifier']
        teacher.fit(train_features, train_labels)
    pseudo = select_pseudo_labels(teacher, unlabeled_features, threshold)
    selected = pseudo['selected_indices']
    if len(selected):
        student_features = np.concatenate([train_features, unlabeled_features[selected]], axis=0)
        student_labels = np.concatenate([train_labels, pseudo['labels']])
        weights = np.concatenate([np.ones(len(train_labels)), np.full(len(selected), PSEUDO_WEIGHT)])
    else:
        student_features, student_labels = train_features, train_labels
        weights = np.ones(len(train_labels))
    student = build_pipeline(STYLE_CONFIG).named_steps['classifier']
    student.fit(student_features, student_labels, sample_weight=weights)
    return student, teacher, {**pseudo, 'sample_weight': PSEUDO_WEIGHT, 'rounds': 1}


def fit_self_training(train_texts, train_labels, unlabeled_texts, threshold: float):
    """Reusable raw-text final fit; supply the deduplicated unlabeled pool.

    Returns a fitted ordinary sklearn Pipeline plus the selected-pseudo audit.
    It predicts on raw texts and works with common.decision_scores. The caller
    remains responsible for selecting threshold before outer/test evaluation.
    """
    features = StyleFeatures()
    train_features = features.fit_transform(train_texts)
    unlabeled_features = features.transform(unlabeled_texts)
    with threadpool_limits(limits=2):
        student, _, pseudo = fit_self_training_from_features(
            train_features, train_labels, unlabeled_features, threshold,
        )
    return Pipeline([('style_features', features), ('classifier', student)]), pseudo


def _scores(classifier, features):
    return decision_scores(classifier, features)


def _confidence_summary(pseudo):
    summary = {'selected_count': len(pseudo['labels']), 'classes': {}}
    for label in (0, 1):
        confidence = pseudo['confidence'][pseudo['labels'] == label]
        summary['classes'][str(label)] = {
            'count': len(confidence),
            'confidence_min': float(confidence.min()) if len(confidence) else None,
            'confidence_mean': float(confidence.mean()) if len(confidence) else None,
            'confidence_max': float(confidence.max()) if len(confidence) else None,
        }
    return summary


def run_experiments():
    destination = OUT / 'semi'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'frozen_candidates.json').write_text(json.dumps(CONFIGS, indent=2) + '\n')
    development, protocol = load_protocol()
    unlabeled, pool_audit = load_unlabeled_pool()
    (destination / 'unlabeled_pool_audit.json').write_text(json.dumps(pool_audit, indent=2) + '\n')
    feature_transformer = StyleFeatures()
    # Fixed-rule extraction learns no dataset statistics, so matrices may be
    # cached once. Every teacher/student estimator is fitted on its own split.
    development_features = feature_transformer.transform(development.text)
    unlabeled_features = feature_transformer.transform(unlabeled.text)
    records = {config['name']: {'candidate': config['name'], 'config': config} for config in CONFIGS}
    predictions = {config['name']: {} for config in CONFIGS}
    pseudo_audit = {}
    teacher_audit = {}
    baseline = np.load(OUT / 'style/style_hgb31_predictions.npz', allow_pickle=False)
    with threadpool_limits(limits=2):
        for split_name, split in protocol['protocols'].items():
            train_idx, validation_idx = np.asarray(split['train']), np.asarray(split['validation'])
            train_features, validation_features = development_features[train_idx], development_features[validation_idx]
            train_labels = development.label.iloc[train_idx].to_numpy()
            teacher = build_pipeline(STYLE_CONFIG).named_steps['classifier']
            start = perf_counter()
            teacher.fit(train_features, train_labels)
            teacher_seconds = perf_counter() - start
            teacher_scores = _scores(teacher, validation_features)
            # Confirm the unchanged teacher reproduces the recorded baseline.
            np.testing.assert_array_equal(development.id.iloc[validation_idx], baseline[f'{split_name}_ids'])
            max_score_difference = float(np.max(np.abs(teacher_scores - baseline[f'{split_name}_scores'])))
            np.testing.assert_allclose(teacher_scores, baseline[f'{split_name}_scores'], rtol=1e-10, atol=1e-10)
            teacher_auc = float(roc_auc_score(development.label.iloc[validation_idx], teacher_scores))
            teacher_audit[split_name] = {
                'auroc': teacher_auc, 'fit_seconds': teacher_seconds,
                'baseline_max_absolute_score_difference': max_score_difference,
                'baseline_parity': True, 'true_training_rows': len(train_idx),
            }
            for config in CONFIGS:
                name = config['name']
                start = perf_counter()
                student, _, pseudo = fit_self_training_from_features(
                    train_features, train_labels, unlabeled_features,
                    config['threshold'], teacher=teacher,
                )
                fit_seconds = perf_counter() - start
                # Normalization uses original true-labeled training rows only.
                true_train_scores = _scores(student, train_features)
                validation_scores = _scores(student, validation_features)
                mean, std = float(true_train_scores.mean()), float(true_train_scores.std())
                if std <= 0 or not np.isfinite(validation_scores).all():
                    raise ValueError('Invalid student decision scores')
                validation_labels = development.label.iloc[validation_idx].to_numpy()
                auc = float(roc_auc_score(validation_labels, validation_scores))
                record = records[name]
                record.update({
                    f'{split_name}_auroc': auc,
                    f'{split_name}_fit_seconds': fit_seconds,
                    f'{split_name}_teacher_fit_seconds': teacher_seconds,
                    f'{split_name}_teacher_auroc': teacher_auc,
                    f'{split_name}_delta_vs_teacher': auc - teacher_auc,
                    f'{split_name}_train_score_mean': mean,
                    f'{split_name}_train_score_std': std,
                    f'{split_name}_pseudo_count': len(pseudo['labels']),
                })
                predictions[name].update({
                    f'{split_name}_ids': development.id.iloc[validation_idx].to_numpy(),
                    f'{split_name}_labels': validation_labels,
                    f'{split_name}_scores': validation_scores,
                    f'{split_name}_z': (validation_scores - mean) / std,
                })
                pseudo_audit[f'{name}_{split_name}'] = _confidence_summary(pseudo)
                selected = pseudo['selected_indices']
                pd.DataFrame({
                    'id': unlabeled.id.iloc[selected].to_numpy(),
                    'pseudo_label': pseudo['labels'],
                    'confidence': pseudo['confidence'],
                    'positive_probability': pseudo['positive_probability'],
                    'sample_weight': PSEUDO_WEIGHT,
                }).to_csv(destination / f'{name}_{split_name}_pseudo_labels.csv', index=False)
                print(f'{name} {split_name}: auc={auc:.6f} delta={auc - teacher_auc:+.6f} pseudos={len(selected)}', flush=True)
    baseline.close()
    final_records = list(records.values())
    for record in final_records:
        values = [record[f'{name}_auroc'] for name in protocol['protocols']]
        record['mean_auroc'], record['worst_auroc'] = float(np.mean(values)), float(min(values))
        np.savez_compressed(destination / f"{record['candidate']}_predictions.npz", **predictions[record['candidate']])
    (destination / 'results.json').write_text(json.dumps(final_records, indent=2, allow_nan=False) + '\n')
    pd.DataFrame([{key: value for key, value in record.items() if key != 'config'} for record in final_records]).to_csv(destination / 'scores.csv', index=False)
    (destination / 'teacher_baseline_parity.json').write_text(json.dumps(teacher_audit, indent=2) + '\n')
    (destination / 'pseudo_label_audit.json').write_text(json.dumps(pseudo_audit, indent=2) + '\n')
    (destination / 'provenance.json').write_text(json.dumps({
        'style_config': STYLE_CONFIG,
        'rounds': 1,
        'pseudo_weight': PSEUDO_WEIGHT,
        'thresholds': [0.95, 0.99],
        'normalization': 'Mean/std on original true-labeled protocol training scores only',
        'additional_labels': 'Teacher-generated; unlabeled.csv provides no true labels',
        'validation_label_input_to_training_or_selection': False,
        'outer_holdout_evaluated': False,
        'test_data_read': False,
        'files_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in [Path(__file__), ROOT / 'models/text_generalization/style_search.py', OUT / 'protocol.json']},
    }, indent=2) + '\n')
    return final_records


if __name__ == '__main__':
    run_experiments()
