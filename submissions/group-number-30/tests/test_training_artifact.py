"""Artifact publication checks on tiny fixtures, never competition evidence."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train


class FullTrainingArtifactTests(unittest.TestCase):
    def invoke(self, root):
        data = pd.DataFrame({'id': ['1', '2', '3', '4'],
                             'text': ['one', 'two', 'three', 'four'], 'label': [0, 1, 0, 1]})
        data.attrs['source_sha256'] = 'temporary-fixture'
        config = root / 'config.json'
        config.write_text(json.dumps({'kind': 'word_lr', 'C': 1.}))
        argv = ['train.py', '--train', 'fixture.csv', '--config', str(config),
                '--full-data', '--model-out', str(root / 'model.joblib')]
        with patch.object(sys, 'argv', argv), patch.object(train, 'load_data', return_value=data), \
                patch.object(train, 'build_model', return_value=DummyClassifier()), \
                contextlib.redirect_stdout(io.StringIO()):
            train.main()

    def test_successful_artifact_is_readable_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.invoke(root)
            saved = joblib.load(root / 'model.joblib')
            self.assertEqual(saved['train_rows'], 4)
            self.assertEqual(saved['fit_scope'], 'all_labeled')
            self.assertFalse((root / 'model.joblib.tmp').exists())
            self.assertTrue((root / 'model.metadata.json').is_file())

    def test_unreadable_write_is_not_published_as_completed_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def short_write(bundle, path, **kwargs):
                Path(path).write_bytes(b'incomplete serialized model')
            with patch.object(train.joblib, 'dump', side_effect=short_write):
                with self.assertRaises(Exception):
                    self.invoke(root)
            self.assertFalse((root / 'model.joblib').exists())
            self.assertFalse((root / 'model.metadata.json').exists())


if __name__ == '__main__':
    unittest.main()
