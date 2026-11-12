"""Fixed layout-by-label reweighting for the original 145-feature HGB31.

This is a response to a documented layout/label confound, evaluated on the two
existing internal protocols and the already-frozen layout stress partition.
Only training texts and labels determine sample weights. Layout IDs are never
added to the feature matrix, and there is no external/outer/test-data access.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .common import OUT, PROTOCOL_FILE, ROOT, decision_scores, load_protocol, run_candidates, sha256
from .layout_stress import PLAN_FILE, layout_group
from .style_search import FEATURE_NAMES, StyleFeatures
from .style_search import build_pipeline as build_style_pipeline
from .style_search import candidate_configs as style_candidate_configs


STYLE_CONFIG = next(config for config in style_candidate_configs() if config['name'] == 'style_hgb31')
DESTINATION = OUT / 'group_robust'


def candidate_configs():
    return [
        {'name': 'group_style_hgb31_p05', 'power': 0.5, 'cap': 10.0},
        {'name': 'group_style_hgb31_p10', 'power': 1.0, 'cap': 10.0},
    ]


def _weight_stats(weights):
    return {
        'min': float(weights.min()), 'mean': float(weights.mean()),
        'max': float(weights.max()), 'std': float(weights.std()),
        'sum': float(weights.sum()),
        'effective_sample_size': float(weights.sum() ** 2 / np.square(weights).sum()),
    }


def balance_layout_weights(train_texts, train_labels, power=0.5, cap=10.0):
    """Return weights and an audit using training inputs exclusively.

    For each present layout/label cell, weight is
    ``(n_train / (2 * present_layout_count) / cell_count) ** power``.
    Clip at ``cap`` first, then divide by the training-row mean. Consequently,
    a normalized weight may exceed ``cap``; cap applies before normalization.
    Missing cells receive no synthetic examples and require no division by 0.
    """
    texts = list(train_texts)
    labels = np.asarray(train_labels)
    if labels.ndim != 1 or len(texts) != len(labels) or not len(labels):
        raise ValueError('Expected equally sized, nonempty training texts and 1-D labels')
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError('Training labels must contain binary classes 0 and 1')
    if not np.isfinite(power) or power <= 0 or not np.isfinite(cap) or cap <= 0:
        raise ValueError('power and cap must be finite and positive')
    groups = np.asarray([layout_group(text) for text in texts], dtype=object)
    group_names = sorted(set(groups))
    counts = Counter(zip(groups, labels.tolist()))
    target = len(labels) / (2 * len(group_names))
    raw = np.asarray([(target / counts[(group, label)]) ** power for group, label in zip(groups, labels)])
    clipped = np.minimum(raw, cap)
    normalization_mean = float(clipped.mean())
    weights = clipped / normalization_mean
    if not np.isfinite(weights).all() or weights.min() <= 0:
        raise ValueError('Invalid training sample weights')
    cells = []
    for group in group_names:
        for label in (0, 1):
            count = counts[(group, label)]
            raw_cell = (target / count) ** power if count else None
            clipped_cell = min(raw_cell, cap) if count else None
            cells.append({
                'layout_group': group, 'label': label, 'count': count,
                'raw_weight': raw_cell, 'clipped_weight': clipped_cell,
                'normalized_weight': clipped_cell / normalization_mean if count else None,
            })
    return weights, {
        'training_rows': len(labels),
        'present_layout_group_count': len(group_names),
        'target_cell_count': target,
        'power': float(power), 'cap_before_normalization': float(cap),
        'normalization_mean': normalization_mean,
        'clipped_training_row_count': int((raw > cap).sum()),
        'raw_weight_stats': _weight_stats(raw),
        'pre_normalization_weight_stats': _weight_stats(clipped),
        'normalized_weight_stats': _weight_stats(weights),
        'cells': cells,
        'validation_labels_used': False,
    }


class GroupBalancedStyleClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable raw-text classifier with train-only layout/label reweighting."""

    def __init__(self, power=0.5, cap=10.0, audit_path=None):
        self.power = power
        self.cap = cap
        self.audit_path = audit_path

    def fit(self, X, y):
        texts = list(X)
        self.sample_weight_, self.weight_audit_ = balance_layout_weights(texts, y, self.power, self.cap)
        self.features_ = StyleFeatures()
        features = self.features_.fit_transform(texts)
        self.classifier_ = build_style_pipeline(STYLE_CONFIG).named_steps['classifier']
        with threadpool_limits(limits=2):
            self.classifier_.fit(features, y, sample_weight=self.sample_weight_)
        self.classes_ = self.classifier_.classes_
        self.n_features_in_ = len(FEATURE_NAMES)
        if self.audit_path is not None:
            path = Path(self.audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.weight_audit_, indent=2, allow_nan=False) + '\n')
        return self

    def decision_function(self, X):
        check_is_fitted(self, 'classifier_')
        with threadpool_limits(limits=2):
            return self.classifier_.decision_function(self.features_.transform(X))

    def predict_proba(self, X):
        check_is_fitted(self, 'classifier_')
        with threadpool_limits(limits=2):
            return self.classifier_.predict_proba(self.features_.transform(X))

    def predict(self, X):
        check_is_fitted(self, 'classifier_')
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]


def build_pipeline(config):
    """Return a reusable, unfitted raw-text estimator with no audit side effects."""
    return GroupBalancedStyleClassifier(power=config['power'], cap=config['cap'])


def run():
    configs = candidate_configs()
    development, protocol = load_protocol()
    layout_plan = json.loads(PLAN_FILE.read_text())
    if layout_plan['source_protocol_sha256'] != sha256(PROTOCOL_FILE):
        raise ValueError('The source protocol changed after layout-stress freeze')
    if layout_plan['train_sha256'] != protocol['train_sha256']:
        raise ValueError('The dataset changed after layout-stress freeze')
    layout_train = np.asarray(layout_plan['train'])
    layout_validation = np.asarray(layout_plan['validation'])
    if set(layout_train) & set(layout_validation) or set(layout_train) | set(layout_validation) != set(range(len(development))):
        raise ValueError('Invalid frozen layout partition')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / 'frozen_candidates.json').write_text(json.dumps(configs, indent=2) + '\n')
    (DESTINATION / 'provenance.json').write_text(json.dumps({
        'motivation': 'Train-only domain reweighting for the observed layout/label confound',
        'style_config': STYLE_CONFIG,
        'feature_count': len(FEATURE_NAMES),
        'layout_group_added_to_feature_matrix': False,
        'layout_group_rule_source': 'models.text_generalization.layout_stress.layout_group',
        'weight_formula': '(n_train / (2 * present_layout_groups) / layout_label_cell_count) ** power',
        'weight_operation_order': 'upper-cap at 10, then divide by training-row mean',
        'protocols': ['random', 'topic', 'layout'],
        'layout_diagnostic_status': 'Post hoc existing frozen stress split; not an independent unseen estimate',
        'outer_holdout_evaluated': False, 'test_data_read': False,
        'files_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in [Path(__file__), ROOT / 'models/text_generalization/style_search.py', ROOT / 'models/text_generalization/layout_stress.py', PROTOCOL_FILE, PLAN_FILE]},
    }, indent=2) + '\n')

    # common.run_candidates preserves the established fitting/scoring contract.
    # Its build order is candidate x protocol; audit paths record training-only
    # cell counts and weight statistics without changing that shared evaluator.
    build_count = Counter()
    split_names = list(protocol['protocols'])

    def audited_builder(config):
        split_name = split_names[build_count[config['name']]]
        build_count[config['name']] += 1
        return build_pipeline(config).set_params(
            audit_path=str(DESTINATION / f"{config['name']}_{split_name}_weight_audit.json"),
        )

    records = run_candidates('group_robust', configs, audited_builder)
    with threadpool_limits(limits=2):
        for record in records:
            config = record['config']
            model = build_pipeline(config).set_params(
                audit_path=str(DESTINATION / f"{config['name']}_layout_weight_audit.json"),
            )
            started = perf_counter()
            model.fit(development.text.iloc[layout_train], development.label.iloc[layout_train])
            fit_seconds = perf_counter() - started
            train_scores = decision_scores(model, development.text.iloc[layout_train])
            scores = decision_scores(model, development.text.iloc[layout_validation])
            mean, std = float(train_scores.mean()), float(train_scores.std())
            if std <= 0:
                raise ValueError('Constant layout-training margins')
            labels = development.label.iloc[layout_validation].to_numpy()
            record.update({
                'layout_auroc': float(roc_auc_score(labels, scores)),
                'layout_fit_seconds': fit_seconds,
                'layout_train_score_mean': mean, 'layout_train_score_std': std,
                'layout_validation_group': layout_plan['selected_validation_group'],
            })
            all_aurocs = [record[f'{name}_auroc'] for name in ['random', 'topic', 'layout']]
            record['all_three_mean_auroc'] = float(np.mean(all_aurocs))
            record['all_three_worst_auroc'] = float(min(all_aurocs))
            path = DESTINATION / f"{config['name']}_predictions.npz"
            with np.load(path, allow_pickle=False) as existing:
                predictions = {key: existing[key] for key in existing.files}
            predictions.update({
                'layout_ids': development.id.iloc[layout_validation].to_numpy(),
                'layout_labels': labels, 'layout_scores': scores,
                'layout_z': (scores - mean) / std,
                'layout_development_indices': layout_validation,
                'layout_original_indices': np.asarray(protocol['development_indices'])[layout_validation],
            })
            np.savez_compressed(path, **predictions)
            (DESTINATION / 'results.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
            pd.DataFrame([{key: value for key, value in row.items() if key != 'config'} for row in records]).to_csv(DESTINATION / 'scores.csv', index=False)
            print(f"{config['name']}: layout={record['layout_auroc']:.6f} all_three_mean={record['all_three_mean_auroc']:.6f}", flush=True)
    return records


if __name__ == '__main__':
    run()
