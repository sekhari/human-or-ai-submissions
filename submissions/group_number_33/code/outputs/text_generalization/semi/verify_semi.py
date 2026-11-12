"""Data-free checks for pseudo-label isolation and duplicate handling."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.text_generalization.semi_supervised import (
    exclude_labeled_text_duplicates, select_pseudo_labels, fit_self_training_from_features,
)


class FixedTeacher:
    classes_ = np.array([0, 1])
    def predict_proba(self, features):
        p = np.asarray(features)[:, 0]
        return np.column_stack([1 - p, p])


class SemiChecks(unittest.TestCase):
    def test_threshold_is_inclusive_and_class_independent(self):
        pseudo = select_pseudo_labels(FixedTeacher(), np.array([[.01], [.04], [.5], [.96], [.99]]), .95)
        np.testing.assert_array_equal(pseudo['selected_indices'], [0, 1, 3, 4])
        np.testing.assert_array_equal(pseudo['labels'], [0, 0, 1, 1])
        np.testing.assert_allclose(pseudo['confidence'], [.99, .96, .96, .99])
        strict = select_pseudo_labels(FixedTeacher(), np.array([[.01], [.04], [.5], [.96], [.99]]), .99)
        np.testing.assert_array_equal(strict['selected_indices'], [0, 4])

    def test_exact_duplicate_filter_does_not_normalize_text(self):
        unlabeled = pd.DataFrame({'id':[1, 2, 3], 'text':['Same', 'Same ', 'New']})
        kept, removed = exclude_labeled_text_duplicates(unlabeled, ['Same'])
        self.assertEqual(kept.id.tolist(), [2, 3])
        self.assertEqual(removed.id.tolist(), [1])

    def test_validation_label_permutation_does_not_affect_pseudo_labels(self):
        # Training and prediction API accepts no validation labels. These labels
        # are used only by evaluation downstream, and permuting them changes
        # evaluation while leaving pseudo labels identical.
        validation_labels = np.array([0, 0, 1, 1])
        validation_scores = np.array([.1, .2, .8, .9])
        unlabeled = np.array([[.01], [.04], [.96], [.99]])
        before = select_pseudo_labels(FixedTeacher(), unlabeled, .95)
        first_auc = roc_auc_score(validation_labels, validation_scores)
        validation_labels = validation_labels[::-1]
        after = select_pseudo_labels(FixedTeacher(), unlabeled, .95)
        second_auc = roc_auc_score(validation_labels, validation_scores)
        self.assertNotEqual(first_auc, second_auc)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_student_uses_fixed_weights_and_no_round_two(self):
        # Actual HGB behavior, not an estimator mock: deterministic separated toy
        # data produce confident pseudo labels, and fitted students return
        # finite scores even when the unlabeled pool is empty.
        x = np.r_[np.zeros((64, 2)), np.ones((64, 2))]
        y = np.r_[np.zeros(64, dtype=int), np.ones(64, dtype=int)]
        student, teacher, audit = fit_self_training_from_features(x, y, np.array([[0., 0.], [1., 1.]]), .95)
        self.assertEqual(audit['sample_weight'], .25)
        self.assertEqual(audit['rounds'], 1)
        np.testing.assert_array_equal(audit['labels'], [0, 1])
        self.assertTrue(np.isfinite(student.predict_proba(x)).all())
        empty_student, _, empty = fit_self_training_from_features(x, y, np.empty((0, 2)), .95, teacher=teacher)
        self.assertEqual(len(empty['selected_indices']), 0)
        self.assertTrue(np.isfinite(empty_student.predict_proba(x)).all())


if __name__ == '__main__':
    unittest.main()
