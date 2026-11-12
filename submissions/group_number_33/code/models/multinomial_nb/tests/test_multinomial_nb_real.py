"""Numerical and estimator contracts for custom binary real-boosted MNB."""

import importlib
import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import MultinomialNB


class RealNBTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("models.multinomial_nb.multinomial_nb_real"),
                             "The production real-boost module must exist")
        self.module = importlib.import_module("models.multinomial_nb.multinomial_nb_real")

    def test_hand_computed_binary_update_and_learning_rate(self):
        weights = np.array([.25, .25, .5, 0.])
        labels = np.array([0, 1, 1, 0])
        probabilities = np.array([[.8, .2], [.2, .8], [.8, .2], [.5, .5]])
        np.testing.assert_allclose(self.module.update_weights(weights, labels, probabilities, 1.),
                                   [.1, .1, .8, 0.], rtol=1e-14)
        np.testing.assert_allclose(self.module.update_weights(weights, labels, probabilities, .5),
                                   [1/6, 1/6, 2/3, 0.], rtol=1e-14)
        np.testing.assert_array_equal(weights, [.25, .25, .5, 0.])

    def test_clipping_extreme_probabilities_keeps_normalized_weights_finite(self):
        weights = np.array([.25, .25, .5, 0.])
        probabilities = np.array([[1., 0.], [0., 1.], [1., 0.], [.5, .5]])
        actual = self.module.update_weights(weights, np.array([0, 1, 1, 0]), probabilities, 1.)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual[:3] > 0).all())
        self.assertEqual(actual[-1], 0.)
        self.assertAlmostEqual(actual.sum(), 1.)

    def test_half_error_does_not_stop_real_boosting(self):
        X = csr_matrix([[1., 1.], [1., 1.]])
        model = self.module.RealNBEnsemble(alpha=1., learning_rate=.1, n_estimators=3).fit(X, [0, 1])
        self.assertEqual(len(model.boost_.estimators_), 3)
        np.testing.assert_allclose(model.boost_.estimator_errors_, [.5, .5, .5])
        np.testing.assert_allclose(model.predict_proba(X), [[.5, .5], [.5, .5]])

    def test_perfect_weighted_training_accuracy_stops_at_one_learner(self):
        X = csr_matrix([[2., 0.], [0., 2.]])
        model = self.module.RealNBEnsemble(alpha=1., n_estimators=3).fit(X, [0, 1])
        self.assertEqual(len(model.boost_.estimators_), 1)
        np.testing.assert_allclose(model.boost_.estimator_errors_, [0.])

    def test_clone_fit_and_predict_preserve_features_and_runner_contract(self):
        X = csr_matrix([[4., 0.], [3., 1.], [2., 2.], [1., 3.], [0., 4.], [2., 2.], [0., 0.]])
        y = np.array([0, 0, 0, 1, 1, 1, 0])
        original = X.copy()
        estimator = self.module.RealNBEnsemble(alpha=.1, learning_rate=.25, n_estimators=6)
        model = clone(estimator).fit(X, y)
        self.assertEqual(estimator.get_params(), model.get_params())
        self.assertEqual(len(model.boost_.estimators_), 6)
        self.assertTrue(all(learner.n_features_in_ == 2 for learner in model.boost_.estimators_))
        np.testing.assert_array_equal(model.boost_.estimator_weights_, np.ones(6))
        self.assertEqual(model.coefficient_power, 0.)
        self.assertEqual(model.temperature, 1.)
        self.assertEqual(model.training_algorithm, "SAMME.R-style")
        self.assertEqual((X != original).nnz, 0)
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.)
        np.testing.assert_array_equal(model.predict(X), model.classes_[model.predict_proba(X).argmax(axis=1)])

    def test_external_weights_are_honored_and_zero_feature_rows_are_excluded(self):
        X = csr_matrix([[2., 0.], [0., 4.], [0., 0.]])
        weights = np.array([1., 3., 999.])
        first = self.module.RealNBEnsemble(alpha=1., n_estimators=2).fit(X, [0, 1, 0], sample_weight=weights)
        scaled = self.module.RealNBEnsemble(alpha=1., n_estimators=2).fit(X, [0, 1, 0], sample_weight=weights*100)
        np.testing.assert_allclose(first.boost_.estimators_[0].model_.feature_count_, [[1., 0.], [0., 6.]])
        np.testing.assert_allclose(first.predict_proba(X), scaled.predict_proba(X), atol=1e-14)
        np.testing.assert_array_equal(weights, [1., 3., 999.])

    def test_both_prediction_rules_match_hand_calculated_equal_pools(self):
        X = csr_matrix([[2., 0.], [0., 4.]])
        learners = [MultinomialNB(alpha=a, fit_prior=False).fit(X, [0, 1]) for a in [1., 2.]]
        model = self.module.RealNBEnsemble(aggregation="equal_probability_average")
        model.boost_ = SimpleNamespace(estimators_=learners, estimator_weights_=np.ones(2))
        model.classes_ = np.array([0, 1])
        model.n_features_in_ = 2
        V = csr_matrix([[1., 0.], [0., 1.]])
        expected = np.array([5/22, 19/26])
        np.testing.assert_allclose(model.predict_proba(V), np.column_stack([1-expected, expected]))
        model.aggregation = "real_log_odds"
        expected = expit(np.array([np.log(2/9)+np.log(3/8), np.log(10/3)+np.log(9/4)])/2)
        np.testing.assert_allclose(model.predict_proba(V), np.column_stack([1-expected, expected]))

    def test_invalid_weights_parameters_and_unfitted_prediction_are_rejected(self):
        X = csr_matrix([[2., 0.], [0., 4.]])
        model = self.module.RealNBEnsemble()
        with self.assertRaises(NotFittedError):
            model.predict_proba(X)
        with self.assertRaises(NotFittedError):
            model.predict(X)
        for weights in [[-1., 1.], [np.nan, 1.], [0., 0.], [0., 1.], [1.]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.module.RealNBEnsemble().fit(X, [0, 1], sample_weight=weights)
        for parameters in [dict(alpha=0), dict(learning_rate=0), dict(n_estimators=0),
                           dict(n_estimators=1.5), dict(aggregation="samme")]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.module.RealNBEnsemble(**parameters).fit(X, [0, 1])
        with self.assertRaises(ValueError):
            model.fit(csr_matrix((2, 2)), [0, 1])


if __name__ == "__main__":
    unittest.main()
