import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB

from models.multinomial_nb import multinomial_nb
from preprocess import tfidf_data as shared_tfidf_data

from models.multinomial_nb.multinomial_nb import calculate_metrics


class ClassSmoothingTests(unittest.TestCase):
    def test_sample_weight_cap_is_applied_before_final_normalization(self):
        X = csr_matrix([[2., 0.], [0., 4.]])
        model = multinomial_nb.fit_weighted_nb(X, [0, 1], 1., -1., 0., weight_cap=1.)
        np.testing.assert_allclose(np.exp(model.feature_log_prob_), [[17 / 22, 5 / 22], [5 / 26, 21 / 26]])

    def test_weighted_nb_ignores_zero_vectors_when_normalizing_weights(self):
        X = csr_matrix([[2., 0.], [0., 4.], [0., 0.]])
        model = multinomial_nb.fit_weighted_nb(X, [0, 1, 0], 1.0, -1.0, 0.0)
        np.testing.assert_allclose(np.exp(model.feature_log_prob_), [[11 / 14, 3 / 14], [3 / 14, 11 / 14]])
        np.testing.assert_allclose(model.predict_proba(csr_matrix([[0., 0.]])), [[0.5, 0.5]])

    def test_feature_specific_smoothing_changes_prior_counts_only(self):
        X = csr_matrix([[3., 0.], [0., 4.]])
        original = X.copy()
        model = multinomial_nb.fit_weighted_nb(X, [0, 1], 1.0, 0.0, -1.0)
        np.testing.assert_allclose(np.exp(model.feature_log_prob_), [[29 / 35, 6 / 35], [4 / 21, 17 / 21]])
        self.assertEqual((X != original).nnz, 0)

    def test_equal_class_smoothing_recovers_ordinary_nb(self):
        X = csr_matrix([[2, 0, 1], [0, 4, 2], [1, 3, 0], [5, 0, 1]])
        y = np.array([0, 1, 1, 0])
        for alpha in [1e-12, 0.1, 10.0]:
            with self.subTest(alpha=alpha):
                expected = MultinomialNB(alpha=alpha, force_alpha=True, fit_prior=False).fit(X, y)
                actual = multinomial_nb.fit_class_smoothed_nb(X, y, alpha, alpha)
                np.testing.assert_allclose(actual.predict_proba(X), expected.predict_proba(X), atol=1e-12)

    def test_asymmetric_smoothing_uses_separate_strengths_and_equal_priors(self):
        X = csr_matrix([[2, 0], [1, 0], [0, 4]])
        y = np.array([0, 0, 1])
        model = multinomial_nb.fit_class_smoothed_nb(X, y, 1.0, 2.0)
        # Human counts [3,0] + 1; AI counts [0,4] + 2.
        np.testing.assert_allclose(np.exp(model.feature_log_prob_), [[4 / 5, 1 / 5], [2 / 8, 6 / 8]])
        np.testing.assert_allclose(np.exp(model.class_log_prior_), [0.5, 0.5])


class MetricsTests(unittest.TestCase):
    def test_auroc_uses_probabilities_even_when_all_predictions_are_zero(self):
        metrics = calculate_metrics([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [2, 0]])

    def test_metrics_keep_ai_as_the_positive_class(self):
        metrics = calculate_metrics([0, 1, 0, 1], [0.1, 0.8, 0.6, 0.4])
        self.assertEqual(metrics["auroc"], 0.75)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            calculate_metrics([0, 1], [0.2, float("nan")])
        with self.assertRaises(ValueError):
            calculate_metrics([0, 1], [0.2, 1.1])

    def test_tuning_fits_text_features_only_within_each_training_fold(self):
        texts = pd.Series(
            [f"human diary example item{i}" for i in range(10)]
            + [f"machine generated example item{i}" for i in range(10, 20)]
        )
        labels = pd.Series([0] * 10 + [1] * 10)
        calls = []

        def record_real_featurization(train, validation, third_batch):
            calls.append((set(train), set(validation)))
            return shared_tfidf_data(train, validation, third_batch)

        with patch.object(multinomial_nb, "tfidf_data", side_effect=record_real_featurization):
            result = multinomial_nb.select_alpha_by_cv(texts, labels, [0.1, 1.0], n_splits=2)
        self.assertEqual(len(calls), 2)
        for training_texts, validation_texts in calls:
            self.assertEqual(len(training_texts), 10)
            self.assertEqual(len(validation_texts), 10)
            self.assertFalse(training_texts & validation_texts)
        self.assertEqual(calls[0][1] | calls[1][1], set(texts))
        self.assertEqual(result["selection_metric"], "mean inner-CV AUROC")
        self.assertIn(result["selected_alpha"], [0.1, 1.0])
        self.assertEqual(len(result["candidates"]), 2)
        for candidate in result["candidates"]:
            self.assertEqual(candidate["mean_cv_auroc"], 1.0)

    def test_weighted_search_refits_features_on_inner_training_rows_only(self):
        texts = pd.Series([f"human diary example item{i}" for i in range(10)]
                          + [f"machine generated example item{i}" for i in range(10, 20)])
        labels = pd.Series([0] * 10 + [1] * 10)
        calls = []

        def record_real_featurization(train, validation, third_batch):
            calls.append((set(train), set(validation)))
            return shared_tfidf_data(train, validation, third_batch)

        candidates = [dict(alpha=0.1, sample_power=-1., prior_power=-1., ai_weight=1.),
                      dict(alpha=1., sample_power=0., prior_power=0., ai_weight=1.)]
        with patch.object(multinomial_nb, "tfidf_data", side_effect=record_real_featurization):
            result = multinomial_nb.select_weighted_parameters_by_cv(texts, labels, candidates, n_splits=2)
        self.assertEqual(len(calls), 2)
        for training_texts, validation_texts in calls:
            self.assertEqual(len(training_texts), 10)
            self.assertEqual(len(validation_texts), 10)
            self.assertFalse(training_texts & validation_texts)
        self.assertEqual(calls[0][1] | calls[1][1], set(texts))
        self.assertIn(result["selected_config"], candidates)
        self.assertEqual(result["selected_mean_cv_auroc"], 1.0)
        self.assertEqual(len(result["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
