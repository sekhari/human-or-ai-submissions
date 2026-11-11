"""Independently verify Model 2 Q2/Q3 identities, IDF, predictions and rank-sum AUROC."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from common import (SEED, SIZES, THREADS, CONFIDENCE, SIMILARITY, config, digest, require, load_inputs,
                    normalized, final_pipeline, write_json)
from model2.model import build_pipeline


def auc(y, scores):
    positive = np.asarray(y) == 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    return float((rankdata(scores)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def verify_fitted(point, fit, score_frames, excluded_ids, calibrated=True):
    metrics = json.loads((point / 'metrics.json').read_text())
    require(digest(point / 'model.joblib') == metrics['model_sha256'], 'Model checksum differs.')
    model = joblib.load(point / 'model.joblib')
    require(model.classes_.tolist() == [0, 1], 'Wrong label mapping.')
    require(not metrics['warnings'], 'Inspect training warnings before reporting completion.')
    pipe = final_pipeline(model, calibrated)
    expected = build_pipeline(config())
    for step in ('tfidf', 'svm'):
        require(pipe[step].get_params() == expected[step].get_params(), f'{step} parameters differ.')
    if calibrated:
        require(model.ensemble is False and model.method == 'sigmoid' and model.n_jobs == 1,
                'Calibration configuration changed.')
        require(model.cv.n_splits == 3 and model.cv.shuffle and model.cv.random_state == SEED,
                'Calibration folds changed.')
    vec = pipe['tfidf']
    counter = CountVectorizer(analyzer=vec.build_analyzer(), vocabulary=vec.vocabulary_)
    counts = counter.transform(fit.text)
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    require((df >= config()['min_df']).all(), 'Vocabulary contains a feature unsupported by fit data.')
    require(np.allclose(vec.idf_, np.log((1 + len(fit)) / (1 + df)) + 1, rtol=0, atol=1e-12),
            'Fitted IDF differs from the documented training texts.')
    require(len(vec.vocabulary_) == metrics['feature_count'], 'Feature count differs.')
    checked = {}
    for name, frame in score_frames.items():
        pred = pd.read_csv(point / f'{name}_predictions.csv', dtype={'id': str})
        require(pred.id.tolist() == frame.id.tolist(), 'Prediction order/membership differs.')
        require(pred.label.tolist() == frame.label.tolist(), 'Prediction ground truth differs.')
        with threadpool_limits(limits=THREADS):
            scores = pipe.decision_function(frame.text)
            p = model.predict_proba(frame.text)[:, 1] if calibrated else None
        require(np.isfinite(scores).all() and np.allclose(scores, pred.margin, rtol=0, atol=1e-12),
                'Reloaded decision scores differ.')
        value = auc(frame.label, scores)
        require(abs(value - metrics[f'{name}_auroc']) < 1e-12, 'Independent AUROC differs.')
        require(len(frame) == metrics[f'{name}_rows'], 'Scoring row count differs.')
        if calibrated:
            require(np.allclose(p, pred.probability, rtol=0, atol=1e-12), 'Reloaded probabilities differ.')
            require(abs(auc(frame.label, p) - value) < 1e-12, 'Calibrated ranking AUROC differs.')
        checked[name] = value
        if name == 'validation':
            mask = ~frame.id.isin(excluded_ids)
            sensitive = auc(frame.loc[mask, 'label'], scores[mask.to_numpy()])
            require(abs(sensitive - metrics['validation_sensitivity_auroc']) < 1e-12, 'Sensitivity AUROC differs.')
            require(int(mask.sum()) == metrics['validation_sensitivity_rows'], 'Sensitivity rows differ.')
            checked['sensitivity'] = sensitive
    return checked


def verify(data_dir, q2_dir, q3_dir=None):
    train, val, unlabeled = load_inputs(data_dir, include_unlabeled=True)
    q2 = Path(q2_dir)
    summary = json.loads((q2 / 'summary.json').read_text())
    excluded_ids = summary['sensitivity_excluded_validation_ids']
    overlap = pd.read_csv(q2 / 'validation_overlap.csv', dtype={'id': str})
    require(overlap.id.tolist() == val.id.tolist(), 'Overlap audit order differs.')
    require(overlap.loc[overlap.word_trigram_cosine >= SIMILARITY, 'id'].tolist() == excluded_ids,
            'Sensitivity IDs do not match documented threshold.')
    for name, frame in [('train', train), ('validation', val)]:
        saved = pd.read_csv(q2 / f'{name}_ids.csv', dtype={'id': str})
        require(saved.equals(frame[['id', 'label']].reset_index(drop=True)), 'Canonical ordered split differs.')
    table = pd.read_csv(q2 / 'learning_curve.csv')
    require(table.training_rows.tolist() == list(SIZES), 'Wrong Q2 sizes.')
    rng = np.random.default_rng(SEED)
    indices = [rng.permutation(np.flatnonzero(train.label.to_numpy() == y)) for y in (0, 1)]
    checked_q2, previous = [], set()
    for row in table.to_dict('records'):
        n = int(row['training_rows'])
        point = q2 / f'n_{n}'
        positions = sorted(np.concatenate([a[:n // 2] for a in indices]).tolist())
        subset = train.iloc[positions]
        ids = pd.read_csv(point / 'train_ids.csv', dtype={'id': str})
        require(ids.equals(subset[['id', 'label']].reset_index(drop=True)), 'Nested sampling/seed differs.')
        require(len(ids) == len(set(ids.id)) == n and previous <= set(ids.id), 'Wrong nested subset.')
        require(ids.label.value_counts().to_dict() == {0: n // 2, 1: n // 2}, 'Wrong class balance.')
        require(not set(ids.id) & set(val.id), 'Validation leaked into training.')
        previous = set(ids.id)
        folds = pd.read_csv(point / 'calibration_folds.csv', dtype={'id': str})
        expected_folds = np.zeros(n, dtype=int)
        for i, (_, heldout) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(subset.text, subset.label)):
            expected_folds[heldout] = i
        require(folds.id.tolist() == subset.id.tolist() and
                np.array_equal(folds.calibration_heldout_fold, expected_folds), 'Calibration assignment differs.')
        values = verify_fitted(point, subset, {'train': subset, 'validation': val}, excluded_ids)
        require(abs(values['train'] - row['training_auroc']) < 1e-12 and
                abs(values['validation'] - row['validation_auroc']) < 1e-12, 'Q2 table differs.')
        require(abs(values['train'] - values['validation'] - row['gap']) < 1e-12, 'Q2 gap differs.')
        checked_q2.append({'training_rows': n, **values})
        print(f'PASS Q2 {n:,}: IDs, calibration folds, IDF, reloaded predictions, independent AUROC.', flush=True)
    report = {'status': 'PASS', 'q2': checked_q2,
              'same_original_holdout': True, 'fresh_subset_idf_verified': True,
              'sensitivity_validation_rows': len(val) - len(excluded_ids),
              'reference_comparison': summary['reference_comparison']}
    write_json(q2 / 'verification.json', report)
    if q3_dir is not None:
        q3 = Path(q3_dir)
        m3 = json.loads((q3 / 'summary.json').read_text())
        require(m3['q2_summary_sha256'] == digest(q2 / 'summary.json'), 'Q3 teacher run differs.')
        require(m3['teacher_model_sha256'] == digest(q2 / 'n_8000/model.joblib'), 'Q3 teacher model differs.')
        teacher = joblib.load(q2 / 'n_8000/model.joblib')
        with threadpool_limits(limits=THREADS):
            p = teacher.predict_proba(unlabeled.text)[:, 1]
        near = pd.read_csv(q3 / 'unlabeled_validation_overlap.csv', dtype={'id': str})
        require(near.id.tolist() == unlabeled.id.tolist(), 'Unlabeled overlap audit order differs.')
        exact = unlabeled.text.map(normalized).isin(set(pd.concat([train.text, val.text]).map(normalized))).to_numpy()
        duplicate = unlabeled.text.map(normalized).duplicated().to_numpy()
        blocked = exact | duplicate | (near.word_trigram_cosine.to_numpy() >= SIMILARITY)
        selected = (np.maximum(p, 1 - p) >= CONFIDENCE) & ~blocked
        pseudo_labels = (p >= .5).astype(int)
        saved = pd.read_csv(q3 / 'pseudo_labels.csv', dtype={'id': str})
        require(saved.id.tolist() == unlabeled.id.tolist(), 'Pseudo-label IDs differ.')
        require(np.allclose(saved.teacher_p_ai, p, rtol=0, atol=1e-12), 'Teacher probabilities differ.')
        require(np.array_equal(saved.selected, selected) and np.array_equal(saved.pseudo_label, pseudo_labels),
                'Pseudo-label selection differs.')
        require(np.array_equal(saved.excluded_by_text_audit, blocked), 'Exclusion mask differs.')
        pseudo = unlabeled.loc[selected].copy(); pseudo['label'] = pseudo_labels[selected]
        augmented = pd.concat([train, pseudo], ignore_index=True)
        require(not set(augmented.id) & set(val.id), 'Augmented training leaked validation.')
        ids = pd.read_csv(q3 / 'augmented_train_ids.csv', dtype={'id': str})
        require(ids[['id', 'label']].equals(augmented[['id', 'label']]), 'Augmented labels differ.')
        require(ids.label_source.tolist() == ['official'] * 8000 + ['pseudo'] * len(pseudo), 'Label provenance differs.')
        require(len(pseudo) == m3['selected_pseudo_rows'] and len(augmented) == m3['augmented_training_rows'],
                'Q3 counts differ.')
        values = verify_fitted(q3, augmented, {'train': train, 'validation': val}, excluded_ids, calibrated=False)
        comparison = pd.read_csv(q3 / 'comparison.csv')
        baseline_auc = checked_q2[-1]['validation']
        require(np.allclose(comparison.validation_auroc, [baseline_auc, values['validation']], rtol=0, atol=1e-12),
                'Q3 comparison table differs.')
        require(abs(values['validation'] - baseline_auc - m3['delta_validation_auroc']) < 1e-12, 'Q3 difference differs.')
        report['q3'] = {'status': 'PASS', 'pseudo_rows': len(pseudo), 'values': values,
                         'teacher_confidence_selection_recomputed': True, 'augmented_idf_verified': True,
                         'heldout_ids_excluded': True, 'student_margin_auroc_independently_checked': True}
        write_json(q3 / 'verification.json', report['q3'])
        print('PASS Q3: teacher, selection, fit data, IDF, predictions and independent comparison.', flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--q2-dir', type=Path, required=True)
    p.add_argument('--q3-dir', type=Path)
    a = p.parse_args()
    verify(a.data_dir, a.q2_dir, a.q3_dir)
