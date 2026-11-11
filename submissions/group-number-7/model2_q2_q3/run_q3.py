"""Q3: one-round confidence pseudo-labeling with fixed Model 2 settings."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from common import (CONFIDENCE, SIMILARITY, THREADS, digest, load_inputs, normalized, near_reference,
                    require, new_run, fit_model, save_fitted, write_json, run_metadata, evaluate)


def run(data_dir, q2_dir, output_dir):
    out = new_run(output_dir)
    train, val, unlabeled = load_inputs(data_dir, include_unlabeled=True)
    q2 = Path(q2_dir)
    q2_meta = json.loads((q2 / 'summary.json').read_text())
    require(q2_meta['status'] == 'TRAINING_COMPLETE', 'Q2 is incomplete.')
    point = q2 / 'n_8000'
    base = json.loads((point / 'metrics.json').read_text())
    require(digest(point / 'model.joblib') == base['model_sha256'], 'Teacher checksum differs.')
    for name, frame in [('train', train), ('validation', val)]:
        saved = pd.read_csv(q2 / f'{name}_ids.csv', dtype={'id': str})
        require(saved.equals(frame[['id', 'label']].reset_index(drop=True)), 'Baseline split differs.')
    teacher = joblib.load(point / 'model.joblib')
    # Neither the teacher nor the student sees held-out validation labels during fitting.
    print('Q3: auditing unlabeled text overlap without using validation labels...', flush=True)
    near = near_reference(val, unlabeled)
    near.to_csv(out / 'unlabeled_validation_overlap.csv', index=False)
    labeled_normalized = set(pd.concat([train.text, val.text]).map(normalized))
    exact = unlabeled.text.map(normalized).isin(labeled_normalized).to_numpy()
    duplicate_u = unlabeled.text.map(normalized).duplicated().to_numpy()
    excluded = exact | duplicate_u | (near.word_trigram_cosine.to_numpy() >= SIMILARITY)
    with threadpool_limits(limits=THREADS):
        probability = teacher.predict_proba(unlabeled.text)[:, 1]
    confidence = np.maximum(probability, 1 - probability)
    selected = (confidence >= CONFIDENCE) & ~excluded
    pseudo_labels = (probability >= .5).astype(int)
    require(selected.any(), 'No eligible pseudo-labels at the prespecified confidence threshold.')
    record = pd.DataFrame({'id': unlabeled.id.to_numpy(), 'teacher_p_ai': probability,
                           'confidence': confidence, 'pseudo_label': pseudo_labels,
                           'excluded_by_text_audit': excluded, 'selected': selected})
    record.to_csv(out / 'pseudo_labels.csv', index=False)
    pseudo = unlabeled.loc[selected].copy()
    pseudo['label'] = pseudo_labels[selected]
    augmented = pd.concat([train, pseudo], ignore_index=True)
    require(not set(augmented.id) & set(val.id), 'Augmented training overlaps validation.')
    provenance = pd.DataFrame({'id': augmented.id, 'label': augmented.label,
                               'label_source': ['official'] * len(train) + ['pseudo'] * len(pseudo)})
    provenance.to_csv(out / 'augmented_train_ids.csv', index=False)
    print(f'Q3: selected {len(pseudo):,}/3,000 unlabeled examples; fitting fresh student...', flush=True)
    student, seconds, caught = fit_model(augmented, calibrated=False)
    sensitive_ids = q2_meta['sensitivity_excluded_validation_ids']
    metrics = save_fitted(out, student, {'train': train, 'validation': val}, seconds, caught,
                          calibrated=False, sensitivity_ids=sensitive_ids)
    # Report the same ranking metric for both pipelines. The student does not need
    # probability calibration for AUROC; do not calibrate using pseudo-labels as truth.
    baseline_pred, baseline_auc = evaluate(teacher, val)
    require(abs(baseline_auc - base['validation_auroc']) < 1e-12, 'Baseline AUROC changed.')
    comparison = [
        {'method': 'Supervised only', 'labeled_training_rows': len(train), 'pseudo_training_rows': 0,
         'validation_auroc': baseline_auc, 'validation_sensitivity_auroc': base['validation_sensitivity_auroc']},
        {'method': 'One-round pseudo-labeling', 'labeled_training_rows': len(train), 'pseudo_training_rows': len(pseudo),
         'validation_auroc': metrics['validation_auroc'],
         'validation_sensitivity_auroc': metrics['validation_sensitivity_auroc']}]
    pd.DataFrame(comparison).to_csv(out / 'comparison.csv', index=False)
    write_json(out / 'summary.json', {**run_metadata(), 'status': 'TRAINING_COMPLETE', 'question': 'Q3',
        'confidence_threshold': CONFIDENCE, 'rounds': 1, 'pseudo_weight': 1.0,
        'unlabeled_rows': len(unlabeled), 'audit_excluded_rows': int(excluded.sum()),
        'selected_pseudo_rows': len(pseudo), 'selected_class_counts': pseudo.label.value_counts().sort_index().to_dict(),
        'unselected_rows': int((~selected).sum()), 'augmented_training_rows': len(augmented),
        'teacher_model_sha256': base['model_sha256'], 'teacher_config': q2_meta['config'],
        'q2_summary_sha256': digest(q2 / 'summary.json'), 'comparison': comparison,
        'delta_validation_auroc': metrics['validation_auroc'] - baseline_auc,
        'sensitivity_excluded_validation_ids': sensitive_ids,
        'student_calibration': 'None: both baseline/student AUROCs use raw SVM margins. Teacher probabilities select pseudo-labels only.',
        'pseudo_label_accuracy': 'Unknown: hidden unlabeled labels were not accessed.',
        'threshold_selected_using_validation': False, 'q3_train_metric_scope': '8,000 genuinely labeled training examples only'})
    print(f"Q3 complete: baseline={baseline_auc:.6f}, pseudo-labeling={metrics['validation_auroc']:.6f}", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--q2-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    run(a.data_dir, a.q2_dir, a.output_dir)
