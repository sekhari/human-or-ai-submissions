"""Integrity regression checks using temporary fixtures, never competition scores."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import load_data, get_or_create_split, nested_stratified_subsets, audit_data


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / 'train.csv'
        self.frame = pd.DataFrame({'id': [f'{i:04d}' for i in range(100)],
                                  'text': [f'Unique example {i} has several words.' for i in range(100)],
                                  'label': [i % 2 for i in range(100)]})
        self.frame.to_csv(self.path, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_ids_schema_and_invalid_cells(self):
        self.assertEqual(load_data(self.path).id.iloc[0], '0000')
        cases = [self.frame.drop(columns='text'), self.frame.assign(text=' '),
                 self.frame.assign(id='same'), self.frame.assign(label=2)]
        for frame in cases:
            frame.to_csv(self.path, index=False)
            with self.assertRaises(ValueError):
                load_data(self.path)

    def test_split_is_order_stable_and_rejects_changed_data(self):
        data = load_data(self.path)
        path = self.root / 'split.json'
        train, val = get_or_create_split(data, path, validation_size=20)
        saved = path.read_bytes()
        train2, val2 = get_or_create_split(data.iloc[::-1], path, validation_size=20)
        self.assertEqual(train.id.tolist(), train2.id.tolist())
        self.assertEqual(val.id.tolist(), val2.id.tolist())
        self.assertEqual(saved, path.read_bytes())
        self.assertEqual(val.label.value_counts().to_dict(), {0: 10, 1: 10})
        changed = data.copy()
        changed.loc[0, 'text'] = 'A changed passage that must invalidate the split.'
        with self.assertRaisesRegex(ValueError, 'data_sha256'):
            get_or_create_split(changed, path, validation_size=20)
        with self.assertRaisesRegex(ValueError, 'seed'):
            get_or_create_split(data, path, validation_size=20, seed=42)
        self.path.write_text(self.path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'source_sha256'):
            get_or_create_split(load_data(self.path), path, validation_size=20)

    def test_split_rejects_overlap_and_membership_edits(self):
        data = load_data(self.path)
        path = self.root / 'split.json'
        get_or_create_split(data, path, validation_size=20)
        manifest = json.loads(path.read_text())
        manifest['train_ids'][0] = manifest['validation_ids'][0]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            get_or_create_split(data, path, validation_size=20)

    def test_nested_subsets_are_balanced_training_only_prefixes(self):
        train, val = get_or_create_split(load_data(self.path), self.root / 'split.json', validation_size=20)
        subsets = nested_stratified_subsets(train, (10, 20, 50, 80))
        previous = []
        for size, subset in subsets.items():
            self.assertEqual(subset.id.tolist()[:len(previous)], previous)
            self.assertFalse(set(subset.id) & set(val.id))
            self.assertEqual(subset.label.value_counts().to_dict(), {0: size // 2, 1: size // 2})
            previous = subset.id.tolist()

    def test_audit_sensitivity_and_starter_identity(self):
        data = pd.DataFrame({'id': ['a', 'b', 'c', 'd'],
                             'text': ['one two three four five six', 'one two three four five seven',
                                      'completely other unique words here today', 'a different final passage to inspect'],
                             'label': [0, 1, 1, 0]})
        train, val = data.iloc[[0, 2]], data.iloc[[1, 3]]
        report = audit_data(data, train, val, starter=data.iloc[[0, 1]])
        self.assertEqual(report['source_overlap_sensitivity']['excluded_validation_ids'], ['b'])
        self.assertEqual(report['source_overlap_sensitivity']['retained_rows'], 1)
        self.assertTrue(report['starter']['exact_labeled_subset'])
        starter = data.iloc[[0, 1]].copy()
        starter.loc[0, 'label'] = 1
        with self.assertRaisesRegex(ValueError, 'Starter'):
            audit_data(data, train, val, starter=starter)
        self.assertNotIn('one two three', json.dumps(report))


if __name__ == '__main__':
    unittest.main()
