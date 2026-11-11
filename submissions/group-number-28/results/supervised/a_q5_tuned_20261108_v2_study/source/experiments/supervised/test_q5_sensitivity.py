"""Check that typography diagnostics are single-factor, paired and read-only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from human_ai.data import read_csv
from .q5_sensitivity import apply_condition, evaluate_sensitivity


class FrozenProbabilityModel:
    classes_ = np.asarray([0, 1])

    def fit(self, *args, **kwargs):
        raise AssertionError("Sensitivity evaluation must not fit a model")

    def predict_proba(self, documents):
        result = []
        for document in documents:
            base = 0.2 if "human" in document else 0.8 if "machine" in document else 0.3 if document.startswith("C") else 0.7
            typography = 0.02 * sum(character in document for character in ('“', '‘', '–', '—'))
            p = base + typography
            result.append([1 - p, p])
        return np.asarray(result)


class SensitivityChecks(unittest.TestCase):
    def test_each_condition_changes_only_its_declared_characters(self):
        original = "MiXeD ‘a’ ‚b‛ “c” „d‟ – e — f 'g' - h"
        quoted = apply_condition(original, "quote_typography")
        dashed = apply_condition(original, "dash_typography")
        self.assertEqual(quoted, 'MiXeD \'a\' \'b\' "c" "d" – e — f \'g\' - h')
        self.assertEqual(dashed, "MiXeD ‘a’ ‚b‛ “c” „d‟ - e - f 'g' - h")
        self.assertEqual(apply_condition(original, "original"), original)
        self.assertEqual(apply_condition(quoted, "quote_typography"), quoted)
        with self.assertRaises(ValueError):
            apply_condition(original, "unknown")

    def test_predictions_keep_ids_labels_and_do_not_mutate_records_or_fit(self):
        records = [
            {"id": "a", "label": "0", "text": "A “human” – note"},
            {"id": "b", "label": "1", "text": "B ‘machine’ — note"},
            {"id": "c", "label": "0", "text": "C plain note"},
            {"id": "d", "label": "1", "text": "D plain note"},
        ]
        unchanged = deepcopy(records)
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            summary = evaluate_sensitivity(study, {"candidate": FrozenProbabilityModel()}, records, {})
            self.assertEqual(len(summary["rows"]), 3)
            for condition in summary["conditions"]:
                prediction = read_csv(study / "format_predictions" / f"candidate_{condition}.csv")
                self.assertEqual([(r["id"], r["label"]) for r in prediction], [(r["id"], r["label"]) for r in records])
            for condition in ("quote_typography", "dash_typography"):
                row = summary["models"]["candidate"][condition]
                self.assertEqual(row["n_changed"], 2)
                self.assertEqual(row["n_changed_human"], 1)
                self.assertEqual(row["n_changed_ai"], 1)
                self.assertGreater(row["mean_abs_probability_change"], 0)
                self.assertEqual(row["changed_subset_auroc"], 1)
            self.assertIsNone(summary["models"]["candidate"]["original"]["changed_subset_auroc"])
        self.assertEqual(records, unchanged)


if __name__ == "__main__":
    unittest.main()
