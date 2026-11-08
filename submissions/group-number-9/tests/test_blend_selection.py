import unittest

import numpy as np

from src.blend_selection import logit_mixture, select_weights


class BlendSelectionTests(unittest.TestCase):
    def test_perfect_booster_keeps_zero_weight_despite_bad_neural_scores(self):
        y = np.array([0, 1] * 10)
        b = np.where(y == 1, 0.9, 0.1)
        result = select_weights(y, b, np.where(y == 1, -8., 8.), resamples=20)
        self.assertEqual(result['best_weight'], 0)
        self.assertEqual(result['conservative_weight'], 0)
        np.testing.assert_array_equal(logit_mixture(b, np.zeros(20), 0), b)

    def test_selection_is_reproducible_and_conservative_choice_is_eligible(self):
        rng = np.random.default_rng(15)
        y = np.array([0, 1] * 40)
        b = 1 / (1 + np.exp(-(y * 2 - 1 + rng.normal(size=80))))
        n = (y * 2 - 1) * 1.5 + rng.normal(size=80)
        a = select_weights(y, b, n, resamples=30)
        self.assertEqual(a, select_weights(y, b, n, resamples=30))
        chosen = next(r for r in a['candidates'] if r['weight'] == a['conservative_weight'])
        self.assertTrue(chosen['eligible_for_conservative_rule'])
        self.assertLessEqual(a['conservative_weight'], a['best_weight'])

    def test_nonfinite_or_misaligned_components_are_rejected(self):
        with self.assertRaises(ValueError):
            logit_mixture([0.5, 0.6], [float('nan'), 1], 0.2)
        with self.assertRaises(ValueError):
            logit_mixture([0.5], [0, 1], 0.2)


if __name__ == '__main__':
    unittest.main()
