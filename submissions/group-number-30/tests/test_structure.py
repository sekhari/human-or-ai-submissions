import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structure import ClassConditionalPOSSurprisal, POSTransformer, StyleFeatures


class StructureTests(unittest.TestCase):
    def test_surface_contract_handles_empty_and_repeated_text(self):
        model = StyleFeatures(use_pos=False).fit(["", "word word word word"])
        values = model.transform(["", "word word word word"])
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values.shape, (2, len(model.get_feature_names_out())))
        self.assertTrue(np.array_equal(values[0], np.zeros(values.shape[1])))
        features = dict(zip(model.get_feature_names_out(), values[1]))
        self.assertEqual(features["repeated_trigram_ratio"], 0.5)
        self.assertEqual(features["type_token_ratio"], 0.25)

    def test_pos_ratios_and_sequence(self):
        with patch("structure._tagger"), patch("structure.pos_tags", return_value=("PRP", "VB", "JJS", "JJ", "NN")):
            m = StyleFeatures().fit(["I see the best happy dog"])
            f = dict(zip(m.get_feature_names_out(), m.transform(["I see the best happy dog"])[0]))
            self.assertAlmostEqual(f["adjective_pos_ratio"], 0.4)
            self.assertAlmostEqual(f["superlative_pos_ratio"], 0.2)
            self.assertEqual(POSTransformer().fit_transform(["ignored"])[0], "PRP VB JJS JJ NN")

    def test_surprisal_validation_is_read_only_even_for_unknown_contexts(self):
        with patch("structure.pos_tags", side_effect=lambda text: tuple(text.split())):
            m = ClassConditionalPOSSurprisal(order=3).fit(["NN VB NN", "JJ NN JJ"], [0, 1])
            before = copy.deepcopy(m.__dict__)
            values = m.transform(["UNKNOWN UNSEEN NN", ""])
            self.assertTrue(np.isfinite(values).all())
            self.assertEqual(values.shape, (2, 10))
            self.assertEqual(before, m.__dict__)

    def test_oof_features_are_reproducible_and_different_from_self_fit(self):
        X = ["NN VB NN", "NN VB JJ", "NN VB RB", "JJ NN JJ", "JJ NN NN", "JJ NN VB"]
        y = [0, 0, 0, 1, 1, 1]
        with patch("structure.pos_tags", side_effect=lambda text: tuple(text.split())):
            a, b = ClassConditionalPOSSurprisal(order=3), ClassConditionalPOSSurprisal(order=3)
            oof = a.fit_transform(X, y)
            self.assertTrue(np.array_equal(oof, b.fit_transform(X, y)))
            self.assertFalse(np.allclose(oof, a.transform(X)))
            self.assertEqual(a.n_training_samples_, 6)
            self.assertEqual(a.cross_fit_folds_, 3)

    def test_score_orientation_and_small_class_guard(self):
        with patch("structure.pos_tags", side_effect=lambda text: tuple(text.split())):
            m = ClassConditionalPOSSurprisal().fit(["NN VB NN"] * 3 + ["JJ JJ NN"] * 3, [0] * 3 + [1] * 3)
            result = m.transform(["NN VB NN", "JJ JJ NN"])
            self.assertLess(result[0, 2], 0)
            self.assertGreater(result[1, 2], 0)
        with self.assertRaisesRegex(ValueError, "at least two"):
            ClassConditionalPOSSurprisal().fit_transform(["one", "two"], [0, 1])


if __name__ == "__main__":
    unittest.main()
