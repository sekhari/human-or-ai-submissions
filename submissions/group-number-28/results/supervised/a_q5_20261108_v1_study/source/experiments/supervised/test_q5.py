"""Guard normalization and feature-learning boundaries in the frozen Q5 study."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from . import run_q5 as q5


def vectorizers(transformer):
    """Find TF-IDF leaves without relying on their pipeline step names."""
    if isinstance(transformer, TfidfVectorizer):
        return [transformer]
    children = getattr(transformer, "steps", None)
    if children is None:
        children = getattr(transformer, "transformer_list", [])
    return [leaf for _, child in children for leaf in vectorizers(child)]


class Q5Checks(unittest.TestCase):
    def setUp(self):
        self.plan = q5.load_plan()
        self.text = [
            "Human stories describe a quiet garden and its changing seasons.",
            "Human stories describe a quiet garden and a familiar neighbor.",
            "Machine generated summaries explain the process in clear steps.",
            "Machine generated summaries explain the process in several steps.",
            "A familiar neighbor watches the quiet garden during spring.",
            "The process uses clear steps and machine generated summaries.",
        ]

    def test_all_representations_keep_nonempty_rows_at_unit_norm(self):
        dimensions = {}
        for key in q5.MODEL_KEYS:
            transformer = q5.feature_transformer(key, self.plan)
            matrix = transformer.fit_transform(self.text)
            norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
            np.testing.assert_allclose(norms, np.ones(len(self.text)), atol=1e-12, rtol=0)
            dimensions[key] = matrix.shape[1]
        self.assertEqual(dimensions["wc120"], dimensions["w60"] + dimensions["c60"])

    def test_calibration_wraps_the_complete_feature_pipeline(self):
        for key in q5.MODEL_KEYS:
            model = q5.make_model(key, self.plan, calibrated=True)
            self.assertIsInstance(model, CalibratedClassifierCV)
            self.assertIsInstance(model.estimator, Pipeline)
            self.assertIn("features", model.estimator.named_steps)
            self.assertEqual(model.estimator.named_steps["classifier"].C, 1)
            self.assertFalse(model.ensemble)
            self.assertEqual(model.method, "sigmoid")
            self.assertEqual(model.cv.random_state, 242)
            self.assertEqual(model.cv.n_splits, 3)

    def test_transforming_held_text_cannot_expand_training_vocabularies(self):
        marker = "heldonlysentinelneverinthetrainingdocuments"
        for key in q5.MODEL_KEYS:
            transformer = q5.feature_transformer(key, self.plan)
            transformer.fit(self.text)
            leaves = vectorizers(transformer)
            before = [dict(leaf.vocabulary_) for leaf in leaves]
            transformer.transform([marker + " " + marker])
            self.assertEqual(before, [leaf.vocabulary_ for leaf in leaves])
            for leaf in leaves:
                analyzer = leaf.build_analyzer()
                allowed = {term for document in self.text for term in analyzer(document)}
                self.assertTrue(set(leaf.vocabulary_) <= allowed)
                if leaf.analyzer == "word":
                    self.assertNotIn(marker, leaf.vocabulary_)

    def test_fold_learning_boundaries_and_resume_reuses_completed_cv(self):
        rows = [{"id": str(i), "text": f"record{i} " + self.text[i % len(self.text)],
                 "label": str(i % 2)} for i in range(36)]
        seen = []
        original = q5.feature_transformer

        def tracking(key, plan):
            transformer = original(key, plan)
            method = transformer.fit_transform

            def fit_transform(documents, y=None, **kwargs):
                documents = list(documents)
                seen.append((key, set(documents)))
                return method(documents, y, **kwargs)

            transformer.fit_transform = fit_transform
            return transformer

        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            study = Path(directory)
            with patch.object(q5, "feature_transformer", tracking):
                first = q5.run_cv(study, self.plan, rows)
            saved = json.loads((study / "cv_progress.json").read_text())
            with patch.object(q5, "fit_record", side_effect=AssertionError("Completed fold was refitted")):
                second = q5.run_cv(study, self.plan, rows)
        self.assertEqual(first, second)
        self.assertEqual(len(seen), 12)
        self.assertEqual(len(saved), 12)
        folds = list(StratifiedKFold(3, shuffle=True, random_state=142).split(rows, [r["label"] for r in rows]))
        for position, (key, documents) in enumerate(seen):
            fit, held = folds[position % 3]
            self.assertEqual(key, q5.MODEL_KEYS[position // 3])
            self.assertEqual(documents, {rows[i]["text"] for i in fit})
            self.assertTrue(documents.isdisjoint({rows[i]["text"] for i in held}))


if __name__ == "__main__":
    unittest.main()
