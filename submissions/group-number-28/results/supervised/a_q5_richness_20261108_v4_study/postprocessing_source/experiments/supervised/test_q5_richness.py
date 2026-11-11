"""Check feature definitions and learning boundaries in the Q5 ablation."""
import math
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .richness_features import LengthFeatures, RichnessFeatures, tokenize


class RichnessFeatureChecks(unittest.TestCase):
    def test_frozen_tokens_and_length_are_document_local(self):
        documents = ["CAFÉ naïve 中文 don't don’t alpha2 beta_ 123.", ""]
        original = deepcopy(documents)
        self.assertEqual(tokenize(documents[0]), ["café", "naïve", "中文", "don", "t", "don", "t"])
        expected = np.asarray([[math.log1p(7), math.log1p(len(documents[0]))], [0, 0]])
        extractor = LengthFeatures()
        self.assertIs(extractor.fit(documents, [0, 1]), extractor)
        np.testing.assert_array_equal(extractor.transform(documents), expected)
        np.testing.assert_array_equal(extractor.transform(documents),
                                      np.concatenate([extractor.transform([d]) for d in documents]))
        self.assertEqual(documents, original)
        self.assertEqual(extractor.transform([]).shape, (0, 2))
        self.assertEqual(len(extractor.get_feature_names_out()), 2)

    def test_richness_exact_formulas_and_short_documents(self):
        extractor = RichnessFeatures(window_size=3)
        actual = extractor.fit_transform(["a b a c", "a b a b", "a", "", "a a a a"])
        np.testing.assert_allclose(actual[0], [5 / 6, 0.75, 0.0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(actual[1], [2 / 3, 0.5, 1 / 3], rtol=0, atol=1e-15)
        np.testing.assert_allclose(actual[2:], [[1, 0, 0], [0, 0, 0], [1 / 3, 0, 2 / 3]], rtol=0, atol=1e-15)
        self.assertEqual(actual.dtype, np.float64)
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(extractor.transform([]).shape, (0, 3))
        np.testing.assert_array_equal(RichnessFeatures().transform(["a b a c"]), [[0.75, 0.75, 0]])

    def test_sliding_windows_use_every_position(self):
        document = " ".join(["a"] * 49 + ["b", "c"])
        self.assertEqual(RichnessFeatures().transform([document])[0, 0], 5 / 100)
        documents = [document, "a b a b", "Unique words remain distinct"]
        extractor = RichnessFeatures()
        original = deepcopy(documents)
        combined = extractor.fit_transform(documents)
        np.testing.assert_array_equal(combined, np.concatenate([clone(extractor).transform([d]) for d in documents]))
        self.assertEqual(documents, original)
        self.assertEqual(extractor.get_feature_names_out().tolist(),
                         ["mattr_window_50", "token_entropy_over_log_count", "repeated_bigram_fraction"])

    def test_richness_is_not_a_duplicate_of_length_or_one_formula(self):
        texts = ["a b a b", "a a b b", "a b c d"]
        lengths = LengthFeatures().transform(texts)
        np.testing.assert_array_equal(lengths, np.tile(lengths[0], (3, 1)))
        features = RichnessFeatures(window_size=3).transform(texts)
        self.assertEqual(features[0, 1], features[1, 1])
        self.assertNotEqual(features[0, 2], features[1, 2])
        self.assertNotEqual(features[0, 0], features[2, 0])
        self.assertNotEqual(features[0, 1], features[2, 1])

    def test_scaling_fits_training_rows_and_transform_does_not_update_it(self):
        train = ["a a a a", "a b", "a b c d e f"]
        held = ["a " * 1000, "b c d"]
        for extractor in (LengthFeatures(), RichnessFeatures()):
            pipeline = Pipeline([("extract", extractor), ("scale", StandardScaler())])
            pipeline.fit(train)
            expected_mean = extractor.transform(train).mean(axis=0)
            np.testing.assert_array_equal(pipeline.named_steps["scale"].mean_, expected_mean)
            before = deepcopy(pipeline.named_steps["scale"].__dict__)
            pipeline.transform(held)
            for key, value in before.items():
                np.testing.assert_array_equal(pipeline.named_steps["scale"].__dict__[key], value)
            np.testing.assert_allclose(pipeline.transform(train).mean(axis=0), 0, atol=1e-15)

    def test_invalid_window_and_nontext_inputs_fail(self):
        for width in (0, -1, True, 1.5, "50"):
            with self.assertRaises(ValueError):
                RichnessFeatures(width).fit(["a b"])
            with self.assertRaises(ValueError):
                RichnessFeatures(width).transform(["a b"])
        for extractor in (LengthFeatures(), RichnessFeatures()):
            with self.assertRaises(TypeError):
                extractor.transform([None])

    def test_complete_feature_and_scaling_pipeline_stays_inside_calibration(self):
        from . import q5_richness as study
        from . import run_q5 as q5
        feature_plan = q5.load_plan()
        before = deepcopy(feature_plan)
        for key in ("c60", "c60_length", "c60_richness", "c60_length_richness"):
            model = study.make_model(key, feature_plan)
            self.assertIsInstance(model, CalibratedClassifierCV)
            self.assertEqual(list(model.estimator.named_steps), ["features", "classifier"])
            self.assertEqual(model.estimator.named_steps["classifier"].C, 1)
            self.assertEqual(model.cv.n_splits, 3)
            self.assertEqual(model.cv.random_state, 242)
            self.assertFalse(model.ensemble)
            features = model.estimator.named_steps["features"]
            if key == "c60":
                self.assertIsInstance(features, TfidfVectorizer)
                self.assertEqual(features.norm, "l2")
                self.assertEqual(features.max_features, 60000)
                self.assertEqual(features.ngram_range, (3, 5))
            else:
                blocks = dict(features.transformer_list)
                self.assertIsInstance(blocks["char"], TfidfVectorizer)
                self.assertEqual(features.transformer_weights["char"], 1)
                for name, size, transformer_class in (("length", 2, LengthFeatures),
                                                       ("richness", 3, RichnessFeatures)):
                    self.assertEqual(name in blocks, name in key)
                    if name in blocks:
                        self.assertEqual(list(blocks[name].named_steps), ["extract", "scale"])
                        self.assertIsInstance(blocks[name].named_steps["extract"], transformer_class)
                        self.assertIsInstance(blocks[name].named_steps["scale"], StandardScaler)
                        self.assertEqual(features.transformer_weights[name], 1 / math.sqrt(size))
        self.assertEqual(feature_plan, before)

    def test_calibration_extractor_fits_and_independent_process_reload(self):
        from . import q5_richness as study
        from . import run_q5 as q5
        labels = np.asarray([i % 2 for i in range(24)])
        documents = [f"document{i} " + ("A human story describes quiet gardens. " if i % 2
                                        else "Generated summaries enumerate ordered steps. ") * (1 + i % 4)
                     for i in range(24)]
        model = study.make_model("c60_length_richness", q5.load_plan())
        fitted_texts = []
        original_fit = LengthFeatures.fit

        def tracking_fit(extractor, X, y=None):
            fitted_texts.append(set(X))
            return original_fit(extractor, X, y)

        with patch.object(LengthFeatures, "fit", tracking_fit):
            model.fit(documents, labels)
        expected = [set(documents[i] for i in fit) for fit, _ in
                    StratifiedKFold(3, shuffle=True, random_state=242).split(documents, labels)]
        expected.append(set(documents))
        self.assertEqual(len(fitted_texts), 4)
        self.assertCountEqual([frozenset(x) for x in fitted_texts], [frozenset(x) for x in expected])
        fitted_union = model.calibrated_classifiers_[0].estimator.named_steps["features"]
        for name, transformer in fitted_union.transformer_list:
            if name == "char":
                continue
            values = transformer.named_steps["extract"].transform(documents)
            np.testing.assert_allclose(transformer.named_steps["scale"].mean_, values.mean(axis=0), atol=1e-15)
            np.testing.assert_allclose(transformer.named_steps["scale"].var_, values.var(axis=0), atol=1e-15)
        probe = ["A human garden story.", "Generated summaries enumerate steps. " * 3]
        probabilities = model.predict_proba(probe)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            input_path = Path(directory) / "probe.json"
            joblib.dump(model, model_path)
            input_path.write_text(json.dumps(probe))
            source = ("import joblib,json,sys; model=joblib.load(sys.argv[1]); "
                      "texts=json.load(open(sys.argv[2])); print(json.dumps(model.predict_proba(texts).tolist()))")
            restored = json.loads(subprocess.check_output(
                [sys.executable, "-c", source, str(model_path), str(input_path)], text=True))
        np.testing.assert_array_equal(restored, probabilities)

    def test_cv_fits_only_training_folds_and_resume_reuses_all_checkpoints(self):
        from . import q5_richness as study
        from . import run_q5 as q5
        plan, feature_plan = study.load_plan(), q5.load_plan()
        plan["reference_study"] = "toy_reference"
        rows = [{"id": str(i), "label": str(i % 2),
                 "text": f"document{i} " + ("A human garden story. " if i % 2
                                             else "A generated summary lists steps. ") * (1 + i % 5)}
                for i in range(36)]
        observed_char, observed_numeric = [], []
        original_features, original_read = q5.feature_transformer, study.read_json
        original_length, original_richness = LengthFeatures.fit, RichnessFeatures.fit

        def tracking_features(key, fp):
            vectorizer = original_features(key, fp)
            original_fit = vectorizer.fit_transform

            def fit_transform(documents, y=None, **kwargs):
                documents = list(documents)
                observed_char.append(frozenset(documents))
                return original_fit(documents, y, **kwargs)

            vectorizer.fit_transform = fit_transform
            return vectorizer

        def tracking_length(extractor, X, y=None):
            observed_numeric.append(("length", frozenset(X)))
            return original_length(extractor, X, y)

        def tracking_richness(extractor, X, y=None):
            observed_numeric.append(("richness", frozenset(X)))
            return original_richness(extractor, X, y)

        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            folder = Path(directory) / "toy_study"
            folder.mkdir()

            def read_fixture(path):
                path = Path(path)
                if path.parent.name == "toy_reference" and path.name == "cv_progress.json":
                    return [{"model": "c60", "fold": row["fold"], "auroc": row["original_auroc"]}
                            for row in original_read(folder / "cv_progress.json") if row["model_id"] == "c60"]
                return original_read(path)

            with patch.object(q5, "feature_transformer", tracking_features), \
                    patch.object(LengthFeatures, "fit", tracking_length), \
                    patch.object(RichnessFeatures, "fit", tracking_richness), \
                    patch.object(study, "read_json", read_fixture):
                first = study.run_cv(folder, plan, feature_plan, rows)
            with patch.object(study, "read_json", read_fixture), \
                    patch.object(q5, "fit_record", side_effect=AssertionError("A completed fold was refitted")):
                second = study.run_cv(folder, plan, feature_plan, rows)
            self.assertEqual(first, second)
            progress = original_read(folder / "cv_progress.json")
            self.assertEqual(len(progress), 12)
            self.assertEqual(len(list((folder / "cv_predictions").glob("*.csv"))), 12)
            self.assertFalse(original_read(folder / "cv_candidate.json")["selection_uses_validation"])
        folds = list(StratifiedKFold(3, shuffle=True, random_state=142).split(rows, [r["label"] for r in rows]))
        self.assertEqual(len(observed_char), 12)
        for index, documents in enumerate(observed_char):
            fit, held = folds[index % 3]
            expected = frozenset(rows[i]["text"] for i in fit)
            excluded = frozenset(rows[i]["text"] for i in held)
            self.assertEqual(documents, expected)
            self.assertTrue(documents.isdisjoint(excluded))
        expected_numeric = Counter((name, frozenset(rows[i]["text"] for i in fit))
                                   for name in ("length", "richness")
                                   for fit, _ in folds for _ in range(2))
        self.assertEqual(Counter(observed_numeric), expected_numeric)


if __name__ == "__main__":
    unittest.main()
