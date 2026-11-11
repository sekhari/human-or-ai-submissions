"""Guard quote canonicalization, serialization and learning boundaries."""
import json
from copy import deepcopy
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
from sklearn.model_selection import StratifiedKFold

from .q5_sensitivity import apply_condition


class QuoteChecks(unittest.TestCase):
    def test_quote_mapping_is_stateless_idempotent_and_preserves_words(self):
        from .quote_normalizer import QuoteNormalizer
        text = ["Human \u2018words\u2019 and \u201cQuotes\u201d \u2014 123.", "ASCII 'words' and \"Quotes\" stay."]
        original = list(text)
        transformer = QuoteNormalizer()
        self.assertIs(transformer.fit(text, [0, 1]), transformer)
        transformed = list(transformer.transform(text))
        self.assertEqual(transformed, [apply_condition(s, "quote_typography") for s in text])
        self.assertEqual(list(transformer.transform(transformed)), transformed)
        self.assertEqual(text, original)
        self.assertIn("\u2014", transformed[0])
        self.assertEqual(list(clone(transformer).fit(text).transform(text)), transformed)
        from sklearn.feature_extraction.text import TfidfVectorizer
        analyze = TfidfVectorizer().build_analyzer()
        self.assertEqual([analyze(s) for s in original], [analyze(s) for s in transformed])

    def test_normalizer_reloads_in_an_independent_process(self):
        from .quote_normalizer import QuoteNormalizer
        document = "A \u201cquoted\u201d phrase and an \u2018apostrophe\u2019."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normalizer.joblib"
            joblib.dump(QuoteNormalizer().fit([document]), path)
            source = ("import joblib,json,sys; "
                      "m=joblib.load(sys.argv[1]); "
                      "print(json.dumps(list(m.transform([sys.argv[2]]))))")
            output = subprocess.check_output([sys.executable, "-c", source, str(path), document], text=True)
        self.assertEqual(json.loads(output), [apply_condition(document, "quote_typography")])

    def test_complete_preprocessing_stays_inside_calibration(self):
        from . import q5_quotes as quotes
        from . import run_q5 as q5
        features = q5.load_plan()
        original = deepcopy(features)
        for representation in ("c60", "wc120"):
            for normalized in (False, True):
                model = quotes.make_model(representation, normalized, features)
                self.assertIsInstance(model, CalibratedClassifierCV)
                expected = ["quote_normalize", "features", "classifier"] if normalized else ["features", "classifier"]
                self.assertEqual(list(model.estimator.named_steps), expected)
                self.assertEqual(model.estimator.named_steps["classifier"].C, 1)
                self.assertEqual(model.cv.random_state, 242)
                self.assertEqual(model.cv.n_splits, 3)
                self.assertFalse(model.ensemble)
        self.assertEqual(features, original)

    def test_fitted_normalized_pipeline_reloads_in_an_independent_process(self):
        from . import q5_quotes as quotes
        from . import run_q5 as q5
        features = q5.load_plan()
        documents = [
            f"Human story number {i}: a \u201cquiet garden\u201d and a familiar neighbor."
            if i % 2 else f"Generated summary number {i}: a \u2018clear process\u2019 with ordered steps."
            for i in range(24)
        ]
        model = quotes.make_model("wc120", True, features)
        model.fit(documents, np.asarray([i % 2 for i in range(24)]))
        probe = ["A \u201cquiet garden\u201d has familiar neighbors.", "An ordered \u2018clear process\u2019 has steps."]
        expected = model.predict_proba(probe)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            input_path = Path(directory) / "probe.json"
            joblib.dump(model, model_path)
            input_path.write_text(json.dumps(probe))
            source = ("import joblib,json,sys; m=joblib.load(sys.argv[1]); "
                      "x=json.load(open(sys.argv[2])); print(json.dumps(m.predict_proba(x).tolist()))")
            restored = json.loads(subprocess.check_output(
                [sys.executable, "-c", source, str(model_path), str(input_path)], text=True))
        np.testing.assert_array_equal(restored, expected)
        np.testing.assert_array_equal(model.predict_proba(probe),
                                      model.predict_proba([apply_condition(s, "quote_typography") for s in probe]))

    def test_cv_fits_only_the_training_fold_and_resume_skips_completed_fits(self):
        from . import q5_quotes as quotes
        from . import run_q5 as q5
        plan, features = quotes.load_plan(), q5.load_plan()
        plan["reference_study"] = "toy_reference"
        rows = [{"id": str(i), "label": str(i % 2),
                 "text": f"record{i} " + ("A human \u201cstory\u201d describes a garden." if i % 2
                                            else "A generated \u2018summary\u2019 describes ordered steps.")}
                for i in range(36)]
        observed = []
        original_features = q5.feature_transformer
        original_json = quotes.read_json

        def tracking_features(rep, feature_plan):
            transformer = original_features(rep, feature_plan)
            method = transformer.fit_transform

            def fit_transform(documents, y=None, **kwargs):
                documents = list(documents)
                observed.append((rep, set(documents)))
                return method(documents, y, **kwargs)

            transformer.fit_transform = fit_transform
            return transformer

        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            study = Path(directory) / "toy_study"
            study.mkdir()

            def read_fixture(path):
                path = Path(path)
                if path.parent.name == "toy_reference" and path.name == "cv_progress.json":
                    return [{"model": r["representation"], "fold": r["fold"], "auroc": r["original_auroc"]}
                            for r in original_json(study / "cv_progress.json") if not r["normalize_quotes"]]
                return original_json(path)

            with patch.object(q5, "feature_transformer", tracking_features), patch.object(quotes, "read_json", read_fixture):
                first = quotes.run_cv(study, plan, features, rows)
            with patch.object(quotes, "read_json", read_fixture), patch.object(q5, "fit_record", side_effect=AssertionError("A completed fold was refitted")):
                second = quotes.run_cv(study, plan, features, rows)
            from . import verify_q5_quotes as audit
            with patch.object(audit, "read_json", read_fixture):
                reconstructed = audit.verify_cv(study, plan, rows)
            self.assertEqual(set(reconstructed), {"c60_raw", "c60_norm", "wc120_raw", "wc120_norm"})
        self.assertEqual(first, second)
        self.assertEqual(len(observed), 12)
        folds = list(StratifiedKFold(3, shuffle=True, random_state=142).split(rows, [r["label"] for r in rows]))
        for index, (rep, documents) in enumerate(observed):
            normalized = (index // 3) % 2 == 1
            fit, held = folds[index % 3]
            expected = {apply_condition(rows[i]["text"], "quote_typography") if normalized else rows[i]["text"] for i in fit}
            excluded = {apply_condition(rows[i]["text"], "quote_typography") if normalized else rows[i]["text"] for i in held}
            self.assertEqual(rep, ("c60", "wc120")[index // 6])
            self.assertEqual(documents, expected)
            self.assertTrue(documents.isdisjoint(excluded))


if __name__ == "__main__":
    unittest.main()
