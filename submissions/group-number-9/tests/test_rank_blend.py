import unittest

import numpy as np

from src.rank_blend import blend_rank_reference


class RankBlendTests(unittest.TestCase):
    def test_independent_permutations_align_ids_and_preserve_reference_scale(self):
        result = blend_rank_reference([10, 20, 30], [30, 10, 20], [.9, .1, .6],
                                      [20, 30, 10], [.7, .2, .9], .1)
        np.testing.assert_array_equal(result.id, [10, 20, 30])
        np.testing.assert_allclose(result.label, [.9 * .1 + .1, .9 * .6 + .1 * 2/3,
                                                .9 * .9 + .1 / 3])

    def test_ties_get_average_rank_and_zero_weight_returns_exact_reference(self):
        result = blend_rank_reference([1, 2, 3], [1, 2, 3], [.12, .7, .85],
                                      [1, 2, 3], [.5, .5, .8], 1.)
        np.testing.assert_allclose(result.label, [.5, .5, 1.])
        original = blend_rank_reference([1, 2], [2, 1], [.83, .12], [2, 1], [.6, .2], 0.)
        np.testing.assert_array_equal(original.label, [.12, .83])

    def test_rejects_extra_missing_duplicate_or_nonfinite_inputs(self):
        cases = [([1, 2], [.2, .3], [1], [.5]),
                 ([1, 2, 3], [.1, .2, .3], [1, 2], [.5, .6]),
                 ([1, 1], [.2, .3], [1, 2], [.5, .6]),
                 ([1, 2], [.2, .3], [1, 2], [.5, float('nan')])]
        for rids, ref, nids, n in cases:
            with self.assertRaises(ValueError):
                blend_rank_reference([1, 2], rids, ref, nids, n, .1)


if __name__ == '__main__':
    unittest.main()
