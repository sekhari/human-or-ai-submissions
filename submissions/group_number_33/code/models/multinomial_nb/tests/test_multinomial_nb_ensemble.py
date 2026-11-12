"""Small numerical contracts for the fixed-feature MNB ensemble."""

import importlib
import importlib.util
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import MultinomialNB


class EnsembleTestCase(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("models.multinomial_nb.multinomial_nb_ensemble"),
                             "The production MNB ensemble module must exist")
        self.module = importlib.import_module("models.multinomial_nb.multinomial_nb_ensemble")

    @staticmethod
    def training_example():
        # Conflicting equal rows prevent a perfect first weak learner.
        X = csr_matrix([[4., 0.], [3., 1.], [2., 2.], [1., 3.],
                        [0., 4.], [2., 2.], [0., 0.]])
        return X, np.array([0, 0, 0, 1, 1, 1, 0])


class MeanWeightNBTests(EnsembleTestCase):
    def test_external_weights_produce_manual_expected_counts_and_probabilities(self):
        X = csr_matrix([[2., 0.], [0., 4.]])
        weights = np.array([1., 3.])
        model = self.module.MeanWeightNB(alpha=1.).fit(X, [0, 1], sample_weight=weights)
        np.testing.assert_allclose(model.model_.feature_count_, [[1., 0.], [0., 6.]])
        np.testing.assert_allclose(np.exp(model.model_.feature_log_prob_), [[2/3, 1/3], [1/8, 7/8]])
        np.testing.assert_allclose(np.exp(model.model_.class_log_prior_), [.25, .75])
        np.testing.assert_array_equal(weights, [1., 3.])

    def test_global_external_weight_scale_does_not_change_model(self):
        X, y = self.training_example()
        weights = np.array([1., 2., 3., 4., 5., 6., 999.])
        first = self.module.MeanWeightNB(alpha=.2, prior_power=-1.).fit(X, y, sample_weight=weights)
        second = self.module.MeanWeightNB(alpha=.2, prior_power=-1.).fit(X, y, sample_weight=weights*1000)
        np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X), atol=1e-14)

    def test_zero_rows_do_not_change_counts_or_class_priors(self):
        X = csr_matrix([[2., 0.], [0., 4.], [0., 0.]])
        model = self.module.MeanWeightNB().fit(X, [0, 1, 0], sample_weight=[1., 1., 1e6])
        np.testing.assert_allclose(model.model_.feature_count_, [[2., 0.], [0., 4.]])
        np.testing.assert_allclose(model.predict_proba(csr_matrix([[0., 0.]])), [[.5, .5]])

    def test_feature_prior_adds_pseudocounts_without_changing_input_features(self):
        X = csr_matrix([[3., 0.], [0., 4.]])
        original = X.copy()
        model = self.module.MeanWeightNB(alpha=1., prior_power=-1., fit_prior=False).fit(X, [0, 1])
        np.testing.assert_allclose(np.exp(model.model_.feature_log_prob_), [[29/35, 6/35], [4/21, 17/21]])
        self.assertEqual((X != original).nnz, 0)
        self.assertEqual(model.n_features_in_, 2)

    def test_invalid_external_weights_are_rejected(self):
        X = csr_matrix([[2., 0.], [0., 4.]])
        for weights in [[1., -1.], [1., np.nan], [1., np.inf], [0., 0.], [0., 1.], [1.]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.module.MeanWeightNB().fit(X, [0, 1], sample_weight=weights)

    def test_invalid_features_and_hyperparameters_are_rejected(self):
        X = csr_matrix([[2., 0.], [0., 4.]])
        for params in [dict(alpha=0.), dict(alpha=np.inf), dict(prior_power=np.nan)]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.module.MeanWeightNB(**params).fit(X, [0, 1])
        with self.assertRaises(ValueError):
            self.module.MeanWeightNB().fit(csr_matrix((2, 2)), [0, 1])
        with self.assertRaises(ValueError):
            self.module.MeanWeightNB().fit(csr_matrix([[1., -1.], [0., 1.]]), [0, 1])
        with self.assertRaises(ValueError):
            self.module.MeanWeightNB(prior_power=-1.).fit(csr_matrix([[1., 0.], [2., 0.]]), [0, 1])

    def test_predict_methods_require_fitting(self):
        model = self.module.MeanWeightNB()
        for method in [model.predict, model.predict_proba, model.predict_log_proba]:
            with self.subTest(method=method.__name__), self.assertRaises(NotFittedError):
                method(csr_matrix([[1., 0.]]))


class NBEnsembleTests(EnsembleTestCase):
    def test_equal_inference_weights_average_base_probabilities(self):
        model, X = self.aggregation_fixture("probability_average")
        model.coefficient_power = 0.
        # First-feature probabilities2/11 and3/11 receive equal weights.
        np.testing.assert_allclose(model.predict_proba(X[:1]), [[17/22, 5/22]])

    def test_temperature_changes_each_probability_before_averaging(self):
        model, X = self.aggregation_fixture("probability_average")
        model.temperature = .5
        # Odds2/9 and3/8 are squared at temperature1/2.
        expected_ai = (4/85 + 3*9/73)/4
        np.testing.assert_allclose(model.predict_proba(X[:1]), [[1-expected_ai, expected_ai]])

    def test_negative_power_is_stable_for_extreme_positive_learner_weights(self):
        model, X = self.aggregation_fixture("probability_average")
        model.coefficient_power = -2.
        model.boost_.estimator_weights_ = np.array([1e-200, 1e200])
        np.testing.assert_allclose(model.predict_proba(X[:1]), [[9/11, 2/11]])

    def test_clone_and_real_adaboost_fit_keep_all_features_and_change_row_weights(self):
        X, y = self.training_example()
        original = X.copy()
        estimator = self.module.NBEnsemble(alpha=.1, learning_rate=.25, n_estimators=6)
        model = clone(estimator).fit(X, y)
        self.assertIsInstance(model.boost_, AdaBoostClassifier)
        self.assertGreater(len(model.boost_.estimators_), 1)
        self.assertEqual(model.get_params(), estimator.get_params())
        self.assertTrue(all(e.n_features_in_ == X.shape[1] for e in model.boost_.estimators_))
        self.assertFalse(np.allclose(model.boost_.estimators_[0].model_.feature_count_,
                                     model.boost_.estimators_[1].model_.feature_count_))
        self.assertEqual((X != original).nnz, 0)
        np.testing.assert_array_equal(model.classes_, [0, 1])
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.)
        np.testing.assert_array_equal(model.predict(X), model.classes_[model.predict_proba(X).argmax(axis=1)])

    def test_initial_length_weights_cap_then_normalize_and_exclude_zero_rows(self):
        X = csr_matrix([[1., 0.], [0., 10.], [0., 10.], [0., 0.]])
        np.testing.assert_allclose(self.module.initial_weights(X, True), [2.4, .3, .3, 0.])
        np.testing.assert_allclose(self.module.initial_weights(X, False), [1., 1., 1., 0.])

    def test_external_weights_are_applied_once_before_boosting(self):
        X = csr_matrix([[1., 0.], [0., 10.], [0., 10.], [0., 0.]])
        model = self.module.NBEnsemble(initial_weighting=True, n_estimators=1, alpha=.1,
                                      fit_prior=False).fit(X, [0, 1, 1, 0], sample_weight=[1., 2., 2., 999.])
        # Initial [2.4,.3,.3,0] times external weights -> [2.4,.6,.6,0],
        # then nonzero-row mean one -> [2,.5,.5,0].
        np.testing.assert_allclose(model.boost_.estimators_[0].model_.feature_count_, [[2., 0.], [0., 10.]])

    def aggregation_fixture(self, mode):
        X = csr_matrix([[2., 0.], [0., 4.]])
        first = MultinomialNB(alpha=1., fit_prior=False).fit(X, [0, 1])
        second = MultinomialNB(alpha=2., fit_prior=False).fit(X, [0, 1])
        model = self.module.NBEnsemble(aggregation=mode)
        model.boost_ = SimpleNamespace(estimators_=[first, second], estimator_weights_=np.array([1., 3., 0.]))
        model.classes_ = np.array([0, 1])
        model.n_features_in_ = 2
        return model, csr_matrix([[1., 0.], [0., 1.]])

    def test_probability_average_matches_hand_calculated_soft_vote(self):
        model, X = self.aggregation_fixture("probability_average")
        expected = np.array([.25, 37/52])
        np.testing.assert_allclose(model.predict_proba(X), np.column_stack([1-expected, expected]))

    def test_log_odds_average_matches_hand_calculated_geometric_pool(self):
        model, X = self.aggregation_fixture("log_odds_average")
        expected = expit(np.array([np.log(2/9)+3*np.log(3/8), np.log(10/3)+3*np.log(9/4)])/4)
        np.testing.assert_allclose(model.predict_proba(X), np.column_stack([1-expected, expected]))

    def test_samme_probability_is_exactly_sklearn_probability(self):
        X, y = self.training_example()
        model = self.module.NBEnsemble(alpha=.1, n_estimators=6, aggregation="samme").fit(X, y)
        np.testing.assert_array_equal(model.predict_proba(X), model.boost_.predict_proba(X))

    def test_expected_zero_weight_warning_is_suppressed_but_other_warnings_survive(self):
        X, y = self.training_example()
        original_fit = AdaBoostClassifier.fit
        def warn_then_fit(estimator, *args, **kwargs):
            warnings.warn("Unrelated numerical warning", RuntimeWarning)
            return original_fit(estimator, *args, **kwargs)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with patch.object(AdaBoostClassifier, "fit", new=warn_then_fit):
                self.module.NBEnsemble(alpha=.1, n_estimators=6).fit(X, y)
        messages = [str(w.message) for w in captured]
        self.assertIn("Unrelated numerical warning", messages)
        self.assertFalse(any("divide by zero encountered in log" in message for message in messages))

    def test_invalid_aggregation_and_weights_are_rejected(self):
        X, y = self.training_example()
        with self.assertRaises(ValueError):
            self.module.NBEnsemble(aggregation="soft_unknown").fit(X, y)
        for weights in [np.zeros(len(y)), [-1.]*len(y), [np.nan]*len(y), [1.]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.module.NBEnsemble().fit(X, y, sample_weight=weights)
        model, V = self.aggregation_fixture("unknown")
        with self.assertRaises(ValueError):
            model.predict_proba(V)

    def test_predict_requires_fitting_and_binary_class_labels(self):
        model = self.module.NBEnsemble()
        with self.assertRaises(NotFittedError):
            model.predict_proba(csr_matrix([[1., 0.]]))
        with self.assertRaises(NotFittedError):
            model.predict(csr_matrix([[1., 0.]]))
        with self.assertRaises(ValueError):
            model.fit(csr_matrix([[1., 0.], [0., 1.]]), [1, 2])


if __name__ == "__main__":
    unittest.main()
