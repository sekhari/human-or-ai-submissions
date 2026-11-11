"""Guard the six-condition tuning protocol and its resumable fold boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from . import q5_tune as tune
from . import run_q5 as q5


class Q5TuningChecks(unittest.TestCase):
    def setUp(self):
        self.plan = tune.load_plan()
        self.features = q5.load_plan()

    def test_parameter_selection_uses_cv_with_predeclared_ties(self):
        rows = []
        for rep in tune.REPRESENTATIONS:
            rows += [{"candidate_id": tune.candidate_id(rep, c), "representation": rep, "C": c,
                      "mean_cv_auroc": mean, "std_cv_auroc": sd, "validation_auroc": validation}
                     for c, mean, sd, validation in [(0.1, .91, .01, .99), (1, .92, .02, .80), (10, .92, .03, .98)]]
        chosen = tune.choose_parameters(rows)
        self.assertEqual([chosen[rep]["C"] for rep in tune.REPRESENTATIONS], [1, 1])
        for row in rows:
            row.update(mean_cv_auroc=.92, std_cv_auroc=.02)
        chosen = tune.choose_parameters(rows)
        self.assertEqual([chosen[rep]["C"] for rep in tune.REPRESENTATIONS], [1, 1])

    def test_changing_C_preserves_features_and_calibration_boundaries(self):
        before = deepcopy(self.features)
        for rep in tune.REPRESENTATIONS:
            for c in self.plan["C_candidates"]:
                model = tune.make_model(rep, c, self.features)
                self.assertIsInstance(model, CalibratedClassifierCV)
                self.assertIn("features", model.estimator.named_steps)
                self.assertEqual(model.estimator.named_steps["classifier"].C, c)
                self.assertEqual(model.cv.random_state, self.features["seeds"]["calibration"])
                self.assertFalse(model.ensemble)
        self.assertEqual(self.features, before)

    def test_fold_local_feature_cache_and_completed_resume(self):
        passages = ["A quiet garden and a familiar neighbor tell human stories.",
                    "A formal summary explains the machine process in clear steps.",
                    "A familiar neighbor tells quiet stories in the spring garden.",
                    "The machine process follows formal steps in its summary."]
        records = [{"id": str(i), "text": f"record{i} " + passages[i % 4], "label": str(i % 2)} for i in range(36)]
        seen = []
        original = q5.feature_transformer

        def tracking(rep, plan):
            features = original(rep, plan)
            method = features.fit_transform
            def fit_transform(documents, y=None, **kwargs):
                documents = list(documents)
                seen.append((rep, set(documents)))
                return method(documents, y, **kwargs)
            features.fit_transform = fit_transform
            return features

        folds = list(StratifiedKFold(3, shuffle=True, random_state=142).split(records, [r["label"] for r in records]))
        # Make an independent tiny C=1 reference so the real resume gate is exercised.
        reference = []
        for rep in tune.REPRESENTATIONS:
            for fold, (fit, held) in enumerate(folds):
                model = tune.make_model(rep, 1, self.features, calibrated=False)
                model.fit([records[i]["text"] for i in fit], [int(records[i]["label"]) for i in fit])
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score([int(records[i]["label"]) for i in held], model.decision_function([records[i]["text"] for i in held]))
                reference.append({"model": rep, "fold": fold, "auroc": float(auc)})
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            parent = Path(directory)
            study = parent / "tiny_study"
            study.mkdir()
            previous = parent / self.plan["reference_study"]
            previous.mkdir()
            (previous / "cv_progress.json").write_text(json.dumps(reference))
            with patch.object(q5, "feature_transformer", tracking):
                first_rows, first_chosen = tune.run_cv(study, self.plan, self.features, records)
            saved = json.loads((study / "cv_progress.json").read_text())
            with patch.object(q5, "fit_record", side_effect=AssertionError("Completed fold was refitted")), \
                    patch.object(q5, "feature_transformer", side_effect=AssertionError("Completed features were refitted")):
                second_rows, second_chosen = tune.run_cv(study, self.plan, self.features, records)
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first_chosen, second_chosen)
        self.assertEqual(len(saved), 18)
        self.assertEqual(len(seen), 6)
        for position, (rep, documents) in enumerate(seen):
            fit, held = folds[position % 3]
            self.assertEqual(rep, tune.REPRESENTATIONS[position // 3])
            self.assertEqual(documents, {records[i]["text"] for i in fit})
            self.assertTrue(documents.isdisjoint({records[i]["text"] for i in held}))


if __name__ == "__main__":
    unittest.main()
