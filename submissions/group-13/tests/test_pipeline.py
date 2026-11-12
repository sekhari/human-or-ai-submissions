"""CPU checks requiring no competition data or pretrained downloads."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_data, write_predictions, read_json, ROOT
from ensembles import combine, families


class PipelineTests(unittest.TestCase):
    def test_family_and_recipe_directions(self):
        # A unanimous model committee must preserve human/AI direction.
        p = np.array([0.01, 0.25, 0.75, 0.99])
        predictions = {k: p.copy() for k in ['r42', 'r43', 'r44', 'rl1', 'rl2', 't42', 't43', 'q42', 'q43']}
        c = families(predictions)
        for recipe in ['best', 'equal', 't5-qwen', 'roberta']:
            np.testing.assert_allclose(combine(c, recipe), p, atol=1e-14)
        self.assertTrue(np.all(np.diff(combine(c, 'stack')) > 0))
        np.testing.assert_allclose(combine(c, 'midpoint'), (combine(c, 'best') + combine(c, 'stack')) / 2)

    def test_roberta_log_odds_pooling(self):
        p = {k: np.array([.2]) for k in ['r42', 'r43', 'r44']}
        p.update({k: np.array([.6]) for k in ['rl1', 'rl2', 't42', 't43', 'q42', 'q43']})
        expected = expit(.5 * np.log(.2/.8) + .5 * np.log(.6/.4))
        self.assertAlmostEqual(families(p)[0, 0], expected)

    def test_stack_extremes_are_finite(self):
        p = combine(np.array([[0., 0., 0.], [1., 1., 1.]]), 'stack')
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(0 < p[0] < p[1] < 1)

    def test_csv_order_roundtrip_and_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'predictions.csv'
            frame = pd.DataFrame({'id': [13, 11], 'text': ['sample one', 'sample two']})
            write_predictions(path, frame, [.8, .2])
            self.assertEqual(pd.read_csv(path).id.tolist(), [13, 11])
            with self.assertRaises(FileExistsError):
                write_predictions(path, frame, [.8, .2])
            with self.assertRaises(ValueError):
                write_predictions(Path(folder) / 'bad.csv', frame, [np.nan, .2])
            frame.id = [13, 13]
            frame.to_csv(Path(folder) / 'input.csv', index=False)
            with self.assertRaises(ValueError):
                read_data(Path(folder) / 'input.csv')

    def test_frozen_partition_coverage(self):
        plan = read_json(ROOT / 'configs/stack_splits.json')
        universe = set()
        excluded = set()
        for split in plan['splits']:
            parts = [set(split['parts'][k]) for k in ['base_train', 'meta', 'evaluation']]
            joined = set.union(*parts)
            self.assertEqual(len(joined), 10000)
            self.assertEqual(sum(map(len, parts)), len(joined))
            if universe:
                self.assertEqual(universe, joined)
            universe = joined
            excluded |= parts[1] | parts[2]
        self.assertEqual(excluded, universe)


if __name__ == '__main__':
    unittest.main()
