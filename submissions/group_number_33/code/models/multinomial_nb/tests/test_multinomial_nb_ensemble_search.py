import unittest
from unittest.mock import patch

import pandas as pd

from models.multinomial_nb import multinomial_nb_ensemble as ensemble
from preprocess import tfidf_data


class EnsembleSelectionTests(unittest.TestCase):
    def test_inference_search_uses_only_inner_training_features_and_handles_early_stop(self):
        texts = pd.Series([f"human diary sample{i}" for i in range(10)]
                          + [f"machine generated sample{i}" for i in range(10, 20)])
        labels = pd.Series([0]*10+[1]*10)
        calls = []

        def record_transform(train, val, third):
            calls.append((set(train), set(val)))
            return tfidf_data(train, val, third)

        fitting = [dict(alpha=1., prior_power=0., fit_prior=True,
                        initial_weighting=False, learning_rate=.25)]
        inference = [dict(aggregation="probability_average", coefficient_power=0., temperature=.3),
                     dict(aggregation="log_odds_average", coefficient_power=1., temperature=1.)]
        self.assertTrue(hasattr(ensemble, "select_ensemble_inference_by_cv"))
        with patch.object(ensemble, "tfidf_data", side_effect=record_transform):
            r = ensemble.select_ensemble_inference_by_cv(texts, labels, fitting, [1, 3], inference, n_splits=2)
        self.assertEqual(len(calls), 2)
        for tr, va in calls:
            self.assertEqual(len(tr), 10)
            self.assertEqual(len(va), 10)
            self.assertFalse(tr & va)
        self.assertEqual(calls[0][1] | calls[1][1], set(texts))
        self.assertEqual(len(r["candidates"]), 4)
        self.assertEqual(r["selected_mean_cv_auroc"], 1.)
        for candidate in r["candidates"]:
            self.assertEqual(candidate["fitted_estimators_by_fold"], [1, 1])

    def test_search_refits_features_inside_training_folds_and_records_early_stops(self):
        texts = pd.Series([f"human diary example item{i}" for i in range(10)]
                          + [f"machine generated example item{i}" for i in range(10, 20)])
        labels = pd.Series([0] * 10 + [1] * 10)
        calls = []

        def record_real_transform(train, validation, third):
            calls.append((set(train), set(validation)))
            return tfidf_data(train, validation, third)

        config = dict(alpha=1., prior_power=0., fit_prior=True,
                      initial_weighting=False, learning_rate=.5)
        self.assertTrue(hasattr(ensemble, "select_ensemble_by_cv"))
        with patch.object(ensemble, "tfidf_data", side_effect=record_real_transform):
            result = ensemble.select_ensemble_by_cv(texts, labels, [config], [1, 3],
                                                   ["samme", "probability_average", "log_odds_average"], n_splits=2)
        self.assertEqual(len(calls), 2)
        for train, validation in calls:
            self.assertEqual(len(train), 10)
            self.assertEqual(len(validation), 10)
            self.assertFalse(train & validation)
        self.assertEqual(calls[0][1] | calls[1][1], set(texts))
        self.assertEqual(len(result["candidates"]), 6)
        self.assertEqual(result["selected_mean_cv_auroc"], 1.0)
        for candidate in result["candidates"]:
            self.assertEqual(candidate["fitted_estimators_by_fold"], [1, 1])
            self.assertEqual(candidate["mean_cv_auroc"], 1.0)
