"""CLI boundary checks; opt into real-data reproduction with COMPETITION_DATA_DIR."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd


TEAM = Path(__file__).resolve().parent
sys.path.insert(0, str(TEAM / 'code'))
from models.dave_extratrees.model import DaveExtraTreesClassifier, CHAMPION_PARAMS


class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_cli(self, script, *args):
        return subprocess.run(
            [sys.executable, str(TEAM / script), *map(str, args)],
            cwd=self.root, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
            capture_output=True, text=True, timeout=180,
        )

    def test_train_rejects_modified_data_before_writing_outputs(self):
        (self.root / 'train.csv').write_text('id,text,label\n1,changed,0\n')
        output = self.root / 'run'
        result = self.run_cli('train.py', '--data-dir', self.root, '--output-dir', output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('train.csv checksum', result.stderr)
        self.assertFalse(output.exists())

    def test_train_preserves_existing_run(self):
        output = self.root / 'run'
        output.mkdir()
        marker = output / 'metrics.json'
        marker.write_text('existing run')
        result = self.run_cli('train.py', '--data-dir', self.root, '--output-dir', output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already exists', result.stderr)
        self.assertEqual(marker.read_text(), 'existing run')

    def test_predict_never_overwrites_source_or_existing_output(self):
        for name in ('test.csv', 'sample_submission.csv', 'model.joblib', 'submission.csv'):
            target = self.root / name
            target.write_bytes(b'keep this file')
            result = self.run_cli('predict.py', '--data-dir', self.root,
                                  '--model', self.root / 'model.joblib', '--out', target)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('already exists', result.stderr)
            self.assertEqual(target.read_bytes(), b'keep this file')

    def test_predict_rejects_a_model_that_is_not_the_final_fit(self):
        texts = ['plain short words.'] * 16 + ['# TITLE\n\n- Item ONE!'] * 16
        model = DaveExtraTreesClassifier({**CHAMPION_PARAMS, 'n_estimators': 3}).fit(
            texts, [0] * 16 + [1] * 16)
        path = self.root / 'model.joblib'
        joblib.dump({'model': model, 'mode': 'validate', 'training_rows': 32}, path)
        result = self.run_cli('predict.py', '--data-dir', self.root, '--model', path,
                              '--out', self.root / 'submission.csv')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('final 10000-row', result.stderr)
        self.assertFalse((self.root / 'submission.csv').exists())


@unittest.skipUnless(os.environ.get('COMPETITION_DATA_DIR'), 'Set COMPETITION_DATA_DIR for real-data reproduction')
class ReproductionTests(EntrypointTests):
    def test_separate_validation_training_and_prediction(self):
        source = Path(os.environ['COMPETITION_DATA_DIR']).resolve()
        train_only = self.root / 'train-only'
        train_only.mkdir()
        shutil.copyfile(source / 'train.csv', train_only / 'train.csv')
        validation = self.root / 'validation'
        result = self.run_cli('train.py', '--data-dir', train_only, '--output-dir', validation)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metrics = json.loads((validation / 'metrics.json').read_text())
        self.assertEqual(metrics['benchmark_training_rows'], 8000)
        self.assertEqual(metrics['benchmark_validation_rows'], 2000)
        self.assertAlmostEqual(metrics['validation_auroc'], 0.9956835, places=12)
        self.assertEqual(set(metrics['data_sha256']), {'train.csv'})
        self.assertFalse((validation / 'model.joblib').exists())
        self.assertFalse((validation / 'submission.csv').exists())
        predictions = pd.read_csv(validation / 'validation_predictions.csv')
        self.assertEqual(len(predictions), 2000)
        self.assertEqual(predictions.label.value_counts().to_dict(), {0: 1000, 1: 1000})

        final = self.root / 'final'
        result = self.run_cli('train.py', '--mode', 'final', '--data-dir', train_only,
                              '--output-dir', final)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metrics = json.loads((final / 'metrics.json').read_text())
        self.assertEqual(metrics['final_training_rows'], 10000)
        self.assertNotIn('validation_auroc', metrics)
        self.assertFalse((final / 'submission.csv').exists())
        bundle = joblib.load(final / 'model.joblib')
        self.assertEqual(bundle['model'].n_training_rows_, 10000)
        self.assertEqual(bundle['model'].classifier_.n_features_in_, 145)

        test_only = self.root / 'test-only'
        test_only.mkdir()
        for name in ('test.csv', 'sample_submission.csv'):
            shutil.copyfile(source / name, test_only / name)
        output = self.root / 'submission.csv'
        result = self.run_cli('predict.py', '--data-dir', test_only,
                              '--model', final / 'model.joblib', '--out', output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        submission = pd.read_csv(output)
        self.assertEqual(list(submission), ['id', 'label'])
        self.assertEqual(len(submission), 3000)
        sample = pd.read_csv(test_only / 'sample_submission.csv')
        np.testing.assert_array_equal(submission.id, sample.id)
        self.assertTrue(submission.label.between(0, 1).all())
        self.assertTrue(np.isfinite(submission.label).all())


if __name__ == '__main__':
    unittest.main()
