"""Resume-integrity regressions on tiny temporary fixtures, not dataset evidence."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import evaluate, tagger_resource_identity, verify_environment


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.train = pd.DataFrame({
            'id': [f't{i}' for i in range(12)],
            'text': ['bright thoughtful beautiful writing', 'rough brief direct writing'] * 6,
            'label': [1, 0] * 6})
        self.val = pd.DataFrame({'id': [f'v{i}' for i in range(4)],
                                 'text': ['bright beautiful writing', 'rough direct writing'] * 2,
                                 'label': [1, 0] * 2})
        for frame in (self.train, self.val):
            frame.attrs['source_sha256'] = 'temporary-fixture-source'
        self.config = {'kind': 'word_lr', 'C': 1.0}
        self.hash_patch = patch('experiment.implementation_hash', return_value='fixture-implementation')
        self.runtime_patch = patch('experiment.runtime_identity', return_value={
            'python': 'test', 'versions': {'sklearn': 'test'}, 'nltk_tagger_resource': None})
        self.hash_patch.start()
        self.runtime_patch.start()
        self._run()

    def tearDown(self):
        self.hash_patch.stop()
        self.runtime_patch.stop()
        self.temporary.cleanup()

    def _run(self, resume=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluate(self.config, self.train, self.val, self.root, 'fixture',
                            save_model=True, resume=resume)

    def _edit_probabilities(self, operation):
        path = self.root / 'fixture.predictions.tsv'
        predictions = pd.read_csv(path, sep='\t', dtype={'id': str})
        predictions['probability_ai'] = operation(predictions.probability_ai)
        predictions.to_csv(path, sep='\t', index=False)

    def test_unchanged_run_resumes(self):
        original = json.loads((self.root / 'fixture.json').read_text())
        self.assertEqual(self._run(resume=True), original)

    def test_ranking_preserving_out_of_range_predictions_are_rejected(self):
        self._edit_probabilities(lambda p: p + 1)
        with self.assertRaisesRegex(ValueError, 'invalid probabilities'):
            self._run(resume=True)

    def test_ranking_preserving_probability_change_is_rejected_by_log_loss(self):
        self._edit_probabilities(lambda p: p ** 2)
        with self.assertRaisesRegex(ValueError, 'log loss'):
            self._run(resume=True)

    def test_prediction_checksum_detects_same_values_with_changed_file_bytes(self):
        path = self.root / 'fixture.predictions.tsv'
        with path.open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self._run(resume=True)

    def test_legacy_without_checksums_still_verifies_model_metadata(self):
        metrics_path = self.root / 'fixture.json'
        metrics = json.loads(metrics_path.read_text())
        metrics.pop('model_sha256')
        metrics.pop('predictions_sha256')
        metrics_path.write_text(json.dumps(metrics))
        self._run(resume=True)
        path = self.root / 'fixture.joblib'
        bundle = joblib.load(path)
        bundle['config'] = {'kind': 'word_lr', 'C': 99.0}
        joblib.dump(bundle, path)
        with self.assertRaisesRegex(ValueError, 'model metadata'):
            self._run(resume=True)

    def test_model_checksum_detects_changed_artifact(self):
        path = self.root / 'fixture.joblib'
        bundle = joblib.load(path)
        bundle['fit_scope'] = 'all_labeled'
        joblib.dump(bundle, path)
        with self.assertRaisesRegex(ValueError, 'model file checksum'):
            self._run(resume=True)


class EnvironmentTests(unittest.TestCase):
    def test_dependency_resource_and_python_changes_block_resume(self):
        original = {'python': '3.12', 'versions': {'numpy': '1.0'},
                    'nltk_tagger_resource': {'sha256': 'first'}, 'NLTK_DATA': '/old/path'}
        same = dict(original, NLTK_DATA='/new/path')
        verify_environment(original, same)
        for key, value in [('python', '3.13'), ('versions', {'numpy': '2.0'}),
                           ('nltk_tagger_resource', {'sha256': 'changed'})]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                verify_environment(original, dict(same, **{key: value}))
        legacy = {key: value for key, value in original.items() if key != 'nltk_tagger_resource'}
        with self.assertRaisesRegex(ValueError, 'lacks provenance'):
            verify_environment(legacy, same)

    def test_tagger_fingerprint_is_path_independent_and_changes_with_bytes(self):
        import nltk.data
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resource = 'averaged_perceptron_tagger_eng'
            folder = root / 'first' / 'taggers' / resource
            folder.mkdir(parents=True)
            for suffix in ('classes', 'tagdict', 'weights'):
                (folder / f'{resource}.{suffix}.json').write_text('{}')
            with patch.object(nltk.data, 'path', [str(root / 'first')]):
                first = tagger_resource_identity()
            shutil.copytree(root / 'first', root / 'second')
            with patch.object(nltk.data, 'path', [str(root / 'second')]):
                self.assertEqual(first, tagger_resource_identity())
                (root / 'second' / 'taggers' / resource / f'{resource}.weights.json').write_text('{"changed": 1}')
                self.assertNotEqual(first, tagger_resource_identity())


if __name__ == '__main__':
    unittest.main()
