"""Regression checks for original-ID boundaries, weights and calibration.

The full suite performs exactly seven tiny synthetic SVC fits, with at most
twelve original documents per fit, and no competition-data fitting.
"""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from . import crop_model
from .crop_model import CropAugmentedClassifier, text_sha256, weighted_training_rows
from .run_q5 import load_plan, make_model
from .test_q5 import vectorizers


def tiny_reference():
    plan = deepcopy(load_plan())
    plan["models"]["wc120"].update(word_max_features=40, char_max_features=50)
    plan["word_representation"]["min_df"] = 1
    plan["char_representation"]["min_df"] = 1
    return plan


def synthetic_documents():
    texts, labels = [], []
    for i in range(12):
        label = i % 2
        phrase = ("human diaries recall gardens neighbors summer memory " if label == 0 else
                  "automated outlines explain processes structured steps analysis ")
        texts.append(f"uniqueoriginal{i} " + phrase * 27)
        labels.append(label)
    return texts, np.asarray(labels)


class CropBoundaryTests(unittest.TestCase):
    def test_weight_layout_conserves_each_parent_and_input_order_without_fitting(self):
        docs = ["alpha " * 180, "short document", "beta\n" * 150]
        labels = np.array([0, 1, 1])
        expanded, y, weights, parents = weighted_training_rows(docs, labels)
        self.assertEqual(expanded[:3], docs)
        np.testing.assert_array_equal(parents, [0, 1, 2, 0, 2])
        np.testing.assert_array_equal(y, [0, 1, 1, 0, 1])
        np.testing.assert_allclose(weights, [.9, 1, .9, .1, .1])
        np.testing.assert_allclose(np.bincount(parents, weights=weights), [1, 1, 1])
        self.assertEqual([len(text.split()) for text in expanded[3:]], [100, 100])
        self.assertNotIn("\n", expanded[-1])
        baseline = weighted_training_rows(docs, labels, crop_weight=0)
        self.assertEqual(baseline[0], docs)
        np.testing.assert_array_equal(baseline[2], np.ones(3))
        self.assertEqual(len(baseline[0]), 3)

    def test_external_weights_are_conserved_and_invalid_parameters_are_rejected(self):
        docs = ["word " * 150, "short"]
        _, _, weights, parents = weighted_training_rows(docs, [0, 1], original_sample_weight=[2, 3])
        np.testing.assert_allclose(np.bincount(parents, weights=weights), [2, 3])
        for bad in [-.1, 1.1, float("nan"), True]:
            with self.assertRaises(ValueError):
                weighted_training_rows(docs, [0, 1], crop_weight=bad)
        with self.assertRaises(ValueError):
            weighted_training_rows(docs, [0, 1], min_words=50, crop_words=100)
        with self.assertRaises(ValueError):
            weighted_training_rows(docs, [0])

    def test_zero_crop_matches_direct_reference_pipeline_and_clone(self):
        docs, labels = synthetic_documents()
        plan = tiny_reference()
        wrapper = CropAugmentedClassifier(crop_weight=0, reference_plan=plan)
        copied = clone(wrapper)
        self.assertEqual(copied.get_params(), wrapper.get_params())
        calls = []
        original_fit = LinearSVC.fit

        def counted(estimator, X, y, **kwargs):
            calls.append(X.shape[0])
            return original_fit(estimator, X, y, **kwargs)

        with patch.object(LinearSVC, "fit", counted):
            copied.fit(docs, labels)
            direct = make_model("wc120", plan, calibrated=False).fit(docs, labels)
        self.assertEqual(calls, [12, 12])
        np.testing.assert_allclose(copied.decision_function(docs), direct.decision_function(docs), atol=1e-12, rtol=0)
        self.assertEqual(copied.training_weight_receipt_["n_crops"], 0)
        self.assertEqual(copied.fitting_text_hashes_, tuple(map(text_sha256, docs)))
        self.assertEqual(plan, tiny_reference())

    def test_augmentation_fits_features_on_originals_and_preserves_vocab_at_prediction(self):
        docs, labels = synthetic_documents()
        seen = []
        real_transformer = crop_model.feature_transformer

        def tracking(*args):
            transformer = real_transformer(*args)
            real_fit = transformer.fit_transform

            def tracked_fit(texts, y=None, **kwargs):
                texts = list(texts)
                seen.append(texts)
                return real_fit(texts, y, **kwargs)

            transformer.fit_transform = tracked_fit
            return transformer

        with patch.object(crop_model, "feature_transformer", tracking):
            fitted = CropAugmentedClassifier(.1, tiny_reference()).fit(docs, labels)
        self.assertEqual(seen, [docs])
        self.assertEqual(fitted.training_weight_receipt_["n_crops"], 12)
        self.assertEqual(fitted.training_weight_receipt_["n_training_rows"], 24)
        self.assertAlmostEqual(fitted.training_weight_receipt_["sample_weight_sum"], 12)
        np.testing.assert_allclose(np.bincount(fitted.training_row_parent_indices_,
                                             weights=fitted.training_sample_weight_), np.ones(12))
        leaves = vectorizers(fitted.transformer_)
        before = [dict(leaf.vocabulary_) for leaf in leaves]
        fitted.decision_function(["heldonlysentinel not present in training"])
        self.assertEqual(before, [leaf.vocabulary_ for leaf in leaves])

    def test_calibration_splits_originals_before_creating_variants_and_reload(self):
        docs, labels = synthetic_documents()
        cv = StratifiedKFold(3, shuffle=True, random_state=242)
        seen = []
        real_fit = CropAugmentedClassifier.fit
        svc_calls = []
        real_svc_fit = LinearSVC.fit

        def tracked_fit(estimator, X, y, sample_weight=None):
            texts = list(X)
            result = real_fit(estimator, texts, y, sample_weight)
            seen.append((frozenset(texts), estimator.fitting_text_hashes_,
                         dict(estimator.training_weight_receipt_)))
            return result

        def counted_svc(estimator, X, y, **kwargs):
            svc_calls.append(X.shape[0])
            return real_svc_fit(estimator, X, y, **kwargs)

        calibrator = CalibratedClassifierCV(
            CropAugmentedClassifier(.1, tiny_reference()), method="sigmoid", cv=cv,
            ensemble=False, n_jobs=1,
        )
        with patch.object(CropAugmentedClassifier, "fit", tracked_fit), patch.object(LinearSVC, "fit", counted_svc):
            calibrator.fit(docs, labels)
        self.assertEqual(sorted(svc_calls), [16, 16, 16, 24])
        expected = [frozenset(docs[i] for i in fit) for fit, _ in cv.split(docs, labels)]
        self.assertCountEqual([entry[0] for entry in seen], expected + [frozenset(docs)])
        for fitted_texts, hashes, receipt in seen:
            self.assertEqual(set(hashes), {text_sha256(text) for text in fitted_texts})
            self.assertEqual(receipt["n_originals"], len(fitted_texts))
            self.assertEqual(receipt["n_crops"], len(fitted_texts))
            self.assertAlmostEqual(receipt["sample_weight_sum"], len(fitted_texts))
        probabilities = calibrator.predict_proba(docs)
        np.testing.assert_array_equal(calibrator.classes_, [0, 1])
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.joblib"
            joblib.dump(calibrator, path)
            np.testing.assert_array_equal(joblib.load(path).predict_proba(docs), probabilities)


if __name__ == "__main__":
    unittest.main()
