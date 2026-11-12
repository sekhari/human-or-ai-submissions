"""Numerical and provenance contracts for the fixed-feature NB score blend."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import MultinomialNB


class NBScoreBlendTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("models.multinomial_nb.multinomial_nb_blend"),
                             "The standalone blend implementation must exist")
        self.module = importlib.import_module("models.multinomial_nb.multinomial_nb_blend")

    @staticmethod
    def training_example():
        return (csr_matrix([[4., 0.], [3., 1.], [2., 2.], [1., 3.],
                            [0., 4.], [2., 2.], [0., 0.]]),
                np.array([0, 0, 0, 1, 1, 1, 0]))

    def test_logits_clip_extremes_and_training_scale_uses_population_sd(self):
        np.testing.assert_allclose(self.module.probability_logits([.25, .5, .75]),
                                   [-np.log(3), 0., np.log(3)], rtol=0, atol=1e-14)
        self.assertTrue(np.isfinite(self.module.probability_logits([0., 1.])).all())
        self.assertAlmostEqual(self.module.training_logit_std([.5, .9]), np.log(9)/2)
        # A constant training score cannot define a finite scaling divisor.
        with self.assertRaises(ValueError):
            self.module.training_logit_std([.5, .5])

    def test_hand_computed_scaled_probability_pool_does_not_center_logits(self):
        X = csr_matrix([[2., 0.], [0., 2.]])
        model = self.module.NBScoreBlend(blend_weight=.2, alternate_score_mode="probability_logit", alternate_logit_cap=None)
        model.components_ = {
            "main": MultinomialNB(alpha=1., fit_prior=False).fit(X, [0, 1]),
            "weighted_single": MultinomialNB(alpha=3., fit_prior=False).fit(X, [0, 1]),
        }
        model.logit_scales_ = {"main": np.log(3)/2, "weighted_single": np.log(5/3)}
        model.classes_ = np.array([0, 1])
        model.n_features_in_ = 2
        V = csr_matrix([[1., 0.], [0., 1.], [0., 0.]])
        # Main logits become [-2,2,0], alternate [-1,1,0], blend [-1.8,1.8,0].
        expected = [0.14185106490048777, 0.8581489350995123, .5]
        np.testing.assert_allclose(model.predict_proba(V)[:, 1], expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(model.predict_proba(V).sum(axis=1), 1.)
        np.testing.assert_array_equal(model.predict(V), [0, 1, 0])

    def test_clone_fit_keeps_features_and_uses_fixed_nb_component_config(self):
        from models.multinomial_nb.multinomial_nb import fit_weighted_nb
        X, y = self.training_example()
        original = X.copy()
        estimator = self.module.NBScoreBlend(alpha=3., n_estimators=3)
        model = clone(estimator).fit(X, y)
        self.assertEqual(model.get_params(), estimator.get_params())
        self.assertFalse(hasattr(estimator, "components_"))
        self.assertEqual(model.n_features_in_, 2)
        self.assertEqual(len(model.components_["main"].boost_.estimators_), 3)
        expected = fit_weighted_nb(X, y, alpha=.3, sample_power=1., prior_power=-96.,
                                   ai_weight=1., weight_cap=2.)
        np.testing.assert_allclose(model.components_["weighted_single"].predict_proba(X),
                                   expected.predict_proba(X), rtol=0, atol=1e-14)
        for attr in ["data", "indices", "indptr"]:
            np.testing.assert_array_equal(getattr(X, attr), getattr(original, attr))
        np.testing.assert_array_equal(model.classes_, [0, 1])

    def test_scales_are_fitted_on_training_predictions_and_never_evaluation_batch(self):
        X, y = self.training_example()
        model = self.module.NBScoreBlend(alpha=3., n_estimators=3, alternate_score_mode="raw_log_odds", alternate_logit_cap=.25).fit(X, y)
        for name, component in model.components_.items():
            if name == "main":
                p = np.clip(component.predict_proba(X)[:, 1], np.finfo(float).eps, 1-np.finfo(float).eps)
                expected = np.std(np.log(p)-np.log1p(-p), ddof=0)
            else:
                lp = component.predict_log_proba(X)
                expected = np.std(np.clip(lp[:, 1]-lp[:, 0], -.25, .25), ddof=0)
            self.assertAlmostEqual(model.logit_scales_[name], expected)
        original = model.logit_scales_.copy()
        V = csr_matrix([[1., 0.], [0., 1.], [40., .1]])
        together = model.predict_proba(V)
        separately = np.vstack([model.predict_proba(V[i]) for i in range(V.shape[0])])
        np.testing.assert_allclose(together, separately, rtol=0, atol=1e-14)
        self.assertEqual(model.logit_scales_, original)

    def test_nested_metadata_records_actual_counts_and_saves_smoothing_vector(self):
        X = csr_matrix([[2., 0.], [0., 2.]])
        model = self.module.NBScoreBlend(alpha=1., n_estimators=8, blend_weight=.2,
                                       alternate_score_mode="raw_log_odds", alternate_logit_cap=16.).fit(X, [0, 1])
        with tempfile.TemporaryDirectory() as directory:
            details = model.get_experiment_details(Path(directory))
            json.dumps(details, allow_nan=False)
            self.assertEqual(details["ensemble_fitted_estimators"], 2)
            self.assertEqual(details["n_nb_learners_total"], 2)
            self.assertEqual(details["blend_weights"], {"main": .8, "weighted_single": .2})
            components = details["ensemble_components"]
            self.assertEqual(components["main"]["n_estimators_fitted"], 1)
            self.assertEqual(components["main"]["parameters"]["n_estimators"], 8)
            self.assertEqual(components["weighted_single"]["n_estimators_fitted"], 1)
            vector = np.load(Path(directory) / components["weighted_single"]["smoothing_vector_file"])
            np.testing.assert_array_equal(vector, model.components_["weighted_single"].alpha)
            self.assertNotIn("ensemble_estimator_weights", details)
            self.assertEqual(details["logit_scaling"]["ddof"], 0)
            self.assertFalse(details["logit_scaling"]["centered"])
            self.assertIsNotNone(details["logit_scaling"]["probability_clip"]["main"])
            self.assertIsNone(details["logit_scaling"]["probability_clip"]["weighted_single"])
            self.assertEqual(components["weighted_single"]["raw_logit_cap"], 16.)
            self.assertEqual(components["weighted_single"]["training_score_std"], model.logit_scales_["weighted_single"])

    def test_invalid_parameters_features_and_unfitted_calls_are_rejected(self):
        X, y = self.training_example()
        for parameters in [dict(alpha=0), dict(alpha=np.nan), dict(learning_rate=0),
                           dict(n_estimators=0), dict(n_estimators=1.5), dict(n_estimators=True),
                           dict(blend_weight=-.1), dict(blend_weight=1.1), dict(blend_weight=np.inf),
                           dict(alternate_score_mode="unknown"), dict(alternate_score_mode="raw_log_odds", alternate_logit_cap=0),
                           dict(alternate_score_mode="raw_log_odds", alternate_logit_cap=-1),
                           dict(alternate_score_mode="raw_log_odds", alternate_logit_cap=np.nan),
                           dict(alternate_score_mode="raw_log_odds", alternate_logit_cap=np.inf),
                           dict(alternate_score_mode="probability_logit", alternate_logit_cap=16.)]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.module.NBScoreBlend(**parameters).fit(X, y)
        with self.assertRaises(ValueError):
            self.module.NBScoreBlend(n_estimators=1).fit(csr_matrix([[1., -1.], [0., 1.]]), [0, 1])
        with self.assertRaises(ValueError):
            self.module.NBScoreBlend(n_estimators=1).fit(csr_matrix([[1., 0.], [0., 1.]]), [1, 2])
        for method in [self.module.NBScoreBlend().predict_proba, self.module.NBScoreBlend().predict]:
            with self.assertRaises(NotFittedError):
                method(X)
        model = self.module.NBScoreBlend(alpha=3., n_estimators=3).fit(X, y)
        with self.assertRaises(ValueError):
            model.predict_proba(csr_matrix([[1., 0., 2.]]))

    def test_missing_historical_reference_fails_before_cv(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            with patch.object(self.module, "SELECTION_HISTORY", missing):
                with self.assertRaisesRegex(FileNotFoundError, "Historical"):
                    self.module.load_selection_history(self.module.NBScoreBlend())

    def test_recovered_history_timing_is_validated_and_labeled_before_cv(self):
        # This matches the recovered experiment schema, which has no elapsed_seconds.
        history = dict(main_config=dict(alpha=1500., learning_rate=1., n_estimators=1600,
                                        aggregation="equal_probability_average", random_state=12),
                       weighted_single_config=dict(alpha=.3, sample_power=1., prior_power=-96.,
                                                   ai_weight=1., weight_cap=2.),
                       seed=12, n_folds=5, elapsed_scoring_seconds_reconstructed=12.5,
                       recovery_note="Metadata reconstructed from saved scoring artifacts.")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path = directory / "candidate_results.csv"
            metadata_path = directory / "metadata.json"
            csv_path.write_text("component,alternate_weight,role,mean_cv_auroc,fold_1_auroc,fold_2_auroc,fold_3_auroc,fold_4_auroc,fold_5_auroc\n"
                                "weighted_single,0.25,blend,0.75,0.75,0.75,0.75,0.75,0.75\n")
            with patch.object(self.module, "SELECTION_HISTORY", csv_path), \
                    patch.object(self.module, "SELECTION_METADATA", metadata_path):
                metadata_path.write_text(json.dumps(history))
                with self.assertRaisesRegex(ValueError, "score"):
                    self.module.load_selection_history(self.module.NBScoreBlend(alternate_score_mode="raw_log_odds", alternate_logit_cap=16.))
                legacy = self.module.NBScoreBlend(blend_weight=.25, alternate_score_mode="probability_logit", alternate_logit_cap=None)
                loaded, _, _ = self.module.load_selection_history(legacy)
                timing = loaded["verified_search_timing"]
                self.assertEqual(timing["seconds"], 12.5)
                self.assertTrue(timing["reconstructed"])
                self.assertEqual(timing["source_field"], "elapsed_scoring_seconds_reconstructed")
                self.assertIn("Reconstructed", timing["note"])
                for bad_value in [-1., float("nan"), None]:
                    broken = history.copy()
                    if bad_value is None:
                        del broken["elapsed_scoring_seconds_reconstructed"]
                    else:
                        broken["elapsed_scoring_seconds_reconstructed"] = bad_value
                    metadata_path.write_text(json.dumps(broken))
                    with self.subTest(bad_value=bad_value), self.assertRaises(ValueError):
                        self.module.load_selection_history(legacy)

    def test_history_filters_full_alternate_config_and_keeps_json_safe_plain_comparisons(self):
        import pandas as pd
        alternate = self.module.ALT_CONFIG.copy()
        history = dict(main_config=dict(alpha=1500., learning_rate=1., n_estimators=1600,
                                        aggregation="equal_probability_average", random_state=12),
                       weighted_single_config=alternate, seed=12, n_folds=5, elapsed_seconds=4.)
        rows = []
        for component, score, alpha_factor in [("weighted_single", .8, 1.),
                                                ("weighted_single", .7, 10.),
                                                ("plain_single", .6, 1.)]:
            row = dict(component=component, alternate_weight=.3, role="blend", mean_cv_auroc=score,
                       **{f"fold_{fold}_auroc": score for fold in range(1, 6)})
            row.update({f"alternate_{name}": value for name, value in alternate.items()})
            row["alternate_alpha"] *= alpha_factor
            if component == "plain_single":
                for name in ["prior_power", "sample_power", "ai_weight", "weight_cap"]:
                    row[f"alternate_{name}"] = None
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path = directory / "candidate_results.csv"
            metadata_path = directory / "metadata.json"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            metadata_path.write_text(json.dumps(history))
            with patch.object(self.module, "SELECTION_HISTORY", csv_path), \
                    patch.object(self.module, "SELECTION_METADATA", metadata_path):
                _, candidates, selected = self.module.load_selection_history(
                    self.module.NBScoreBlend(blend_weight=.3, alternate_score_mode="probability_logit", alternate_logit_cap=None))
            self.assertEqual(selected["alternate_alpha"], alternate["alpha"])
            self.assertEqual(selected["mean_cv_auroc"], .8)
            self.assertEqual(len(candidates), 3)
            json.dumps(candidates, allow_nan=False)
            plain = next(row for row in candidates if row["component"] == "plain_single")
            self.assertIsNone(plain["alternate_prior_power"])

    def test_history_disambiguates_main_stages_and_other_main_parameters(self):
        import pandas as pd
        alternate = self.module.ALT_CONFIG.copy()
        history = dict(main_config=dict(alpha=1500., learning_rate=1., n_estimators=1600,
                                        aggregation="equal_probability_average", random_state=12),
                       weighted_single_config=alternate, seed=12, n_folds=5, elapsed_seconds=4.)
        rows = []
        for alpha, rate, stages, score in [(1500., 1., 1600, .8), (1500., 1., 800, .7),
                                            (150., 1., 1600, .7), (1500., .5, 1600, .7)]:
            rows.append(dict(component="weighted_single", alternate_weight=.25, role="blend",
                             main_alpha=alpha, main_learning_rate=rate, main_n_estimators=stages,
                             mean_cv_auroc=score,
                             **{f"alternate_{name}": value for name, value in alternate.items()},
                             **{f"fold_{fold}_auroc": score for fold in range(1, 6)}))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path, metadata_path = directory / "candidate_results.csv", directory / "metadata.json"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            metadata_path.write_text(json.dumps(history))
            with patch.object(self.module, "SELECTION_HISTORY", csv_path), \
                    patch.object(self.module, "SELECTION_METADATA", metadata_path):
                _, candidates, selected = self.module.load_selection_history(
                    self.module.NBScoreBlend(blend_weight=.25, alternate_score_mode="probability_logit", alternate_logit_cap=None))
            self.assertEqual(len(candidates), 4)
            self.assertEqual(selected["main_n_estimators"], 1600)
            self.assertEqual(selected["main_alpha"], 1500.)
            self.assertEqual(selected["main_learning_rate"], 1.)
            self.assertEqual(selected["mean_cv_auroc"], .8)

    def test_raw_log_odds_preserve_saturated_probability_information_and_optional_cap(self):
        X = csr_matrix([[20., 0.], [0., 20.]])
        component = MultinomialNB(alpha=1., fit_prior=False).fit(X, [0, 1])
        V = csr_matrix([[0., 100.], [100., 0.]])
        self.assertEqual(component.predict_proba(V)[0, 1], 1.)
        raw = self.module.NBScoreBlend(alternate_score_mode="raw_log_odds", alternate_logit_cap=None).alternate_scores(component, V)
        np.testing.assert_allclose(raw, [100*np.log(21), -100*np.log(21)], rtol=0, atol=1e-12)
        clipped = self.module.NBScoreBlend(alternate_score_mode="raw_log_odds", alternate_logit_cap=16.)
        np.testing.assert_allclose(clipped.alternate_scores(component, V), [16., -16.])
        legacy = self.module.NBScoreBlend(alternate_score_mode="probability_logit", alternate_logit_cap=None)
        bounded = legacy.alternate_scores(component, V)
        self.assertLess(bounded[0], raw[0])
        self.assertAlmostEqual(bounded[0], np.log((1-np.finfo(float).eps)/np.finfo(float).eps))

    def test_raw_alternate_cap_does_not_clip_main_component(self):
        X = csr_matrix([[20., 0.], [0., 20.]])
        component = MultinomialNB(alpha=1., fit_prior=False).fit(X, [0, 1])
        model = self.module.NBScoreBlend(blend_weight=.2, alternate_score_mode="raw_log_odds", alternate_logit_cap=16.)
        model.components_ = {"main": component, "weighted_single": component}
        model.classes_, model.n_features_in_ = np.array([0, 1]), 2
        model.logit_scales_ = {"main": 10., "weighted_single": 16.}
        V = csr_matrix([[0., 100.], [100., 0.]])
        main_bound = np.log((1-np.finfo(float).eps)/np.finfo(float).eps)
        expected = expit(np.array([1., -1.])*(.8*main_bound/10+.2))
        np.testing.assert_allclose(model.predict_proba(V)[:, 1], expected, rtol=0, atol=1e-14)

    def test_history_matches_score_mode_and_nullable_cap(self):
        import pandas as pd
        history = dict(main_config=dict(alpha=1500., learning_rate=1., n_estimators=1600,
                                        aggregation="equal_probability_average", random_state=12),
                       weighted_single_config=self.module.ALT_CONFIG.copy(), seed=12, n_folds=5,
                       elapsed_seconds=4., alternate_score_config=dict(mode="raw_log_odds", logit_cap=16.))
        rows = []
        for mode, cap in [("raw_log_odds", 16.), ("raw_log_odds", None),
                          ("raw_log_odds", 8.), ("probability_logit", None)]:
            score = .8 if mode == "raw_log_odds" and cap == 16. else .7
            rows.append(dict(component="weighted_single", role="blend", alternate_weight=.2,
                             alternate_score_mode=mode, alternate_logit_cap=cap, mean_cv_auroc=score,
                             **{f"fold_{fold}_auroc": score for fold in range(1, 6)}))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path, metadata_path = directory / "candidate_results.csv", directory / "metadata.json"
            with patch.object(self.module, "SELECTION_HISTORY", csv_path), \
                    patch.object(self.module, "SELECTION_METADATA", metadata_path):
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                metadata_path.write_text(json.dumps(history))
                _, candidates, selected = self.module.load_selection_history(
                    self.module.NBScoreBlend(blend_weight=.2, alternate_score_mode="raw_log_odds", alternate_logit_cap=16.))
                self.assertEqual(selected["alternate_score_mode"], "raw_log_odds")
                self.assertEqual(selected["alternate_logit_cap"], 16.)
                json.dumps(candidates, allow_nan=False)
                history["alternate_score_config"]["logit_cap"] = None
                for row in rows:
                    score = .8 if row["alternate_score_mode"] == "raw_log_odds" and row["alternate_logit_cap"] is None else .7
                    row.update(mean_cv_auroc=score, **{f"fold_{fold}_auroc": score for fold in range(1, 6)})
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                metadata_path.write_text(json.dumps(history))
                _, _, selected = self.module.load_selection_history(
                    self.module.NBScoreBlend(blend_weight=.2, alternate_score_mode="raw_log_odds", alternate_logit_cap=None))
                self.assertIsNone(selected["alternate_logit_cap"])

    def test_default_alternate_score_retains_legacy_probability_logit_behavior(self):
        X = csr_matrix([[20., 0.], [0., 20.]])
        component = MultinomialNB(alpha=1., fit_prior=False).fit(X, [0, 1])
        V = csr_matrix([[0., 100.], [100., 0.]])
        bound = np.log((1-np.finfo(float).eps)/np.finfo(float).eps)
        np.testing.assert_allclose(self.module.NBScoreBlend().alternate_scores(component, V),
                                   [bound, -bound], rtol=0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
