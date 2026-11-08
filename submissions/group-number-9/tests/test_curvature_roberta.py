import unittest

import numpy as np

from src.curvature_roberta import aligned_probability, average_probabilities, blend, select_configuration


class CombinationTests(unittest.TestCase):
    def test_aligns_by_id_and_rejects_missing_duplicates_and_invalid_scores(self):
        np.testing.assert_allclose(aligned_probability([3, 1], [1, 2, 3], [.1, .2, .3]), [.3, .1])
        for ids, source, p in [([4], [1, 2], [.1, .2]), ([1], [1, 1], [.1, .2]),
                               ([1, 1], [1, 2], [.1, .2]), ([1], [1], [float('nan')]),
                               ([1], [1], [1.2])]:
            with self.assertRaises(ValueError):
                aligned_probability(ids, source, p)

    def test_probability_average_and_logit_blend_endpoints(self):
        a, b = np.array([.1, .7]), np.array([.4, .9])
        np.testing.assert_allclose(average_probabilities([a, b]), [.25, .8])
        np.testing.assert_array_equal(blend(a, b, 0), a)
        np.testing.assert_allclose(blend(a, b, 1), b)
        with self.assertRaises(ValueError):
            average_probabilities([a, b[:1]])

    def test_selection_retains_reference_when_neural_adds_no_ranking_information(self):
        y = [0, 0, 1, 1]
        base = np.array([.1, .2, .8, .9])
        bad = np.array([.9, .8, .2, .1])
        selected = select_configuration(y, base, {1: bad, 2: bad}, y, base, {1: bad, 2: bad})
        self.assertEqual(selected['chosen']['weight'], 0)
        self.assertEqual(selected['chosen']['epoch'], 1)
        self.assertFalse(selected['fixed_validation_used'])
        self.assertEqual(selected['training_only_gain'], 0)

    def test_selection_finds_complementary_signal_without_domain_regression(self):
        y = np.array([0] * 4 + [1] * 4)
        base = np.array([.1, .2, .8, .3, .4, .6, .7, .9])
        neural = np.array([.1, .2, .3, .4, .6, .7, .8, .9])
        domain = neural.copy()
        selected = select_configuration(y, base, {1: neural, 2: neural}, y, domain,
                                        {1: np.full(8, .5), 2: np.full(8, .5)})
        self.assertGreater(selected['training_only_gain'], 0)
        self.assertGreater(selected['chosen']['weight'], 0)
        self.assertGreaterEqual(selected['chosen']['domain_auroc'], selected['reference_domain_auroc'] - .0005)


if __name__ == '__main__':
    unittest.main()
