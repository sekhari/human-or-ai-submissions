"""Independent rank-metric edge cases; no Torch/model/data fitting."""
import math
import random
import unittest

from .verify_roberta import auc


class IndependentAucTests(unittest.TestCase):
    def test_tied_rank_credit_matches_explicit_positive_negative_pairs(self):
        rng = random.Random(168)
        for n in (4, 21, 80):
            labels = [0, 1] + [rng.randrange(2) for _ in range(n - 2)]
            scores = [rng.randrange(6) / 3 for _ in labels]
            positive = [score for label, score in zip(labels, scores) if label]
            negative = [score for label, score in zip(labels, scores) if not label]
            expected = sum(float(p > n) + .5 * float(p == n)
                           for p in positive for n in negative) / (len(positive) * len(negative))
            self.assertEqual(auc(labels, scores), expected)

    def test_perfect_reversed_and_constant_scores(self):
        self.assertEqual(auc([0, 0, 1, 1], [0., 1., 2., 3.]), 1.)
        self.assertEqual(auc([0, 0, 1, 1], [3., 2., 1., 0.]), 0.)
        self.assertEqual(auc([0, 0, 1, 1], [1., 1., 1., 1.]), .5)

    def test_probability_saturation_must_not_replace_raw_logit_ranks(self):
        labels, margins = [0, 1, 0, 1], [31., 32., 33., 34.]
        saturated_float32_probabilities = [1., 1., 1., 1.]
        self.assertEqual(auc(labels, margins), .75)
        self.assertEqual(auc(labels, saturated_float32_probabilities), .5)

    def test_single_class_is_undefined_and_invalid_inputs_are_rejected(self):
        self.assertIsNone(auc([1, 1], [0., 1.]))
        for labels, scores in (([0, 1], [1.]), ([0, 2], [1., 2.]),
                               ([0, 1], [math.nan, 1.]), ([], [])):
            with self.assertRaises(ValueError):
                auc(labels, scores)


if __name__ == "__main__":
    unittest.main()
