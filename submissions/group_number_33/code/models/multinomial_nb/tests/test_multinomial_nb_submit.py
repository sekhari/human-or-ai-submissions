import unittest

import numpy as np
import pandas as pd

from models.multinomial_nb.multinomial_nb_submit import make_submission


class SubmissionContractTests(unittest.TestCase):
    def test_predictions_follow_ids_when_template_order_differs(self):
        test = pd.DataFrame({'id': [30, 10, 20]})
        template = pd.DataFrame({'id': [10, 20, 30], 'label': [0, 0, 0]})
        result = make_submission(test, template, np.array([.3, .1, .2]))
        self.assertEqual(list(result.columns), ['id', 'label'])
        self.assertEqual(result.id.tolist(), [10, 20, 30])
        np.testing.assert_array_equal(result.label, [.1, .2, .3])

    def test_duplicate_missing_or_unexpected_ids_are_rejected(self):
        test = pd.DataFrame({'id': [1, 2, 3]})
        for ids in [[1, 1, 3], [1, 2], [1, 2, 4]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                make_submission(test, pd.DataFrame({'id': ids, 'label': 0}), [.1, .2, .3])
        with self.assertRaises(ValueError):
            make_submission(pd.DataFrame({'id': [1, 1, 3]}),
                            pd.DataFrame({'id': [1, 2, 3], 'label': 0}), [.1, .2, .3])

    def test_invalid_prediction_values_or_length_are_rejected(self):
        test = pd.DataFrame({'id': [1, 2]})
        template = pd.DataFrame({'id': [1, 2], 'label': 0})
        for values in [[.1], [.1, np.nan], [.1, np.inf], [-.1, .2], [.1, 1.1], [[.1], [.2]]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                make_submission(test, template, values)


if __name__ == '__main__':
    unittest.main()
