"""Audit saved split identities and analyze held-out predictions without refitting.

Run train.py first to independently refit the selected configuration. This script
never tunes a model, changes the split, or uses hidden test labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score

from data_utils import file_sha256, get_or_create_split, load_data


def analyze(train_path, split_path, predictions_path, output, day2_record=None):
    data = load_data(train_path, expected_rows=10000)
    train, validation = get_or_create_split(data, split_path)
    pred = pd.read_csv(predictions_path, sep='\t', dtype={'id': str})
    if pred.id.duplicated().any() or set(pred.id) != set(validation.id):
        raise ValueError('Predictions must cover the fixed validation IDs exactly once')
    pred = pred.set_index('id').loc[validation.id].reset_index()
    if pred.label.tolist() != validation.label.tolist():
        raise ValueError('Prediction labels do not match the supplied validation labels')
    p = pred.probability_ai.to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities')
    threshold = 0.5  # Fixed for error reporting, never optimized on this holdout.
    predicted = (p >= threshold).astype(int)
    words = validation.text.str.split().str.len().to_numpy()
    # Simple fixed ranges selected without subgroup outcome inspection.
    bins = [(0, 150, '<150 words'), (150, 250, '150-249 words'), (250, np.inf, '>=250 words')]
    subgroup = []
    for low, high, name in bins:
        mask = (words >= low) & (words < high)
        y = validation.label.to_numpy()[mask]
        subgroup.append({'bin': name, 'count': int(mask.sum()),
                         'human_count': int((y == 0).sum()), 'ai_count': int((y == 1).sum()),
                         'validation_auroc': float(roc_auc_score(y, p[mask])) if len(np.unique(y)) == 2 else None,
                         'error_rate_at_0_5': float((predicted[mask] != y).mean())})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(subgroup).to_csv(output / 'length_bins.csv', index=False)
    rows = validation[['id', 'label']].copy()
    rows['probability_ai'] = p
    rows['predicted_label'] = predicted
    rows['whitespace_words'] = words
    rows['error_type'] = np.where((rows.label == 0) & (predicted == 1), 'FP',
                                  np.where((rows.label == 1) & (predicted == 0), 'FN', 'correct'))
    errors = rows[rows.error_type != 'correct'].copy()
    errors.to_csv(output / 'all_validation_errors.csv', index=False)
    selected = pd.concat([errors[errors.error_type == 'FP'].sort_values(['probability_ai', 'id'], ascending=[False, True]).head(5),
                          errors[errors.error_type == 'FN'].sort_values(['probability_ai', 'id']).head(5)])
    selected.to_csv(output / 'selected_errors.csv', index=False)
    report = {'threshold': threshold, 'positive_class': '1=AI', 'train_rows': len(train),
              'validation_rows': len(validation), 'validation_auroc': float(roc_auc_score(validation.label, p)),
              'validation_log_loss': float(log_loss(validation.label, p)),
              'confusion_matrix_order': 'rows true [human,AI], columns predicted [human,AI]',
              'confusion_matrix': confusion_matrix(validation.label, predicted, labels=[0, 1]).tolist(),
              'length_definition': 'len(text.split()), whitespace-separated words', 'length_bins': subgroup,
              'case_selection': 'Five highest-P(AI) false positives and five lowest-P(AI) false negatives; illustrative, not representative.',
              'source_sha256': file_sha256(train_path), 'split_sha256': file_sha256(split_path),
              'predictions_sha256': file_sha256(predictions_path)}
    if day2_record:
        second = json.loads(Path(day2_record).read_text())
        report['day2_split_audit'] = {
            'training_ids_identical_order': train.id.tolist() == list(map(str, second['split']['train_ids'])),
            'validation_ids_identical_order': validation.id.tolist() == list(map(str, second['split']['validation_ids'])),
            'training_file_hash_matches': report['source_sha256'] == second['data_sha256']['train.csv']}
        if not all(report['day2_split_audit'].values()):
            raise ValueError('Day 2 split/data does not match the saved Day 1 protocol')
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=Path('results/split.json'))
    parser.add_argument('--predictions', type=Path, default=Path('results/verify/selected.predictions.tsv'))
    parser.add_argument('--day2-record', type=Path, default=Path('results/day2/day2_run_record.json'))
    parser.add_argument('--out', type=Path, default=Path('results/day3'))
    args = parser.parse_args()
    analyze(args.train, args.split, args.predictions, args.out, args.day2_record)
