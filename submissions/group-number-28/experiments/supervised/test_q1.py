"""Targeted modeling checks; shared standard-library tests remain independent."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from . import run_q1 as q1


class Q1Checks(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((q1.HERE / "q1_plan.json").read_text())

    def test_inner_features_fit_only_inner_training_rows(self):
        rows = [{"id": str(i), "text": f"document{i} " + ("human prose" if i % 2 == 0 else "machine generated"),
                 "label": str(i % 2)} for i in range(30)]
        seen = []
        actual = q1.vectorizer

        def tracking_vectorizer(plan):
            obj = actual(plan)
            fit_transform = obj.fit_transform

            def tracked(documents, y=None):
                documents = list(documents)
                seen.append(set(documents))
                return fit_transform(documents, y)

            obj.fit_transform = tracked
            return obj

        with tempfile.TemporaryDirectory() as tmp, patch.object(q1, "vectorizer", tracking_vectorizer), patch("builtins.print"):
            chosen = q1.tune(self.plan, rows, Path(tmp))
            observed = json.loads((Path(tmp) / "cv_progress.json").read_text())
        splitter = StratifiedKFold(3, shuffle=True, random_state=self.plan["seeds"]["inner_cv"])
        y = [int(r["label"]) for r in rows]
        self.assertEqual(len(seen), 3)
        for fitted_texts, (fit, held) in zip(seen, splitter.split(rows, y), strict=True):
            self.assertEqual(fitted_texts, {rows[i]["text"] for i in fit})
            self.assertTrue(fitted_texts.isdisjoint({rows[i]["text"] for i in held}))
        self.assertEqual(len(observed), 27)
        self.assertEqual(set(chosen), set(q1.FAMILIES))

    def test_calibration_contains_entire_feature_pipeline(self):
        model = q1.make_model(self.plan, "svm", 1.0)
        self.assertIsInstance(model, CalibratedClassifierCV)
        self.assertIsInstance(model.estimator, Pipeline)
        self.assertIn("tfidf", model.estimator.named_steps)
        self.assertFalse(model.ensemble)
        self.assertEqual(model.cv.random_state, 242)

    def test_ai_probability_respects_class_order(self):
        class ReverseClasses:
            classes_ = np.array([1, 0])

            def predict_proba(self, text):
                return np.array([[0.8, 0.2], [0.1, 0.9]])

        np.testing.assert_array_equal(q1.predict_ai(ReverseClasses(), ["a", "b"]), [0.8, 0.1])

    def test_existing_study_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q1, "ROOT", Path(tmp)):
            study = Path(tmp) / "results/supervised/existing_study"
            study.mkdir(parents=True)
            (study / "keep").write_text("original")
            with self.assertRaises(FileExistsError):
                q1.run("existing")
            self.assertEqual((study / "keep").read_text(), "original")


if __name__ == "__main__":
    unittest.main()
