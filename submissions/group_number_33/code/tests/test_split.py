import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocess
from preprocess import split_validation_set


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "train.csv"
        pd.DataFrame({
            "id": range(200),
            "text": [f"example {i}" for i in range(200)],
            "label": [0] * 100 + [1] * 100,
        }).to_csv(self.source, index=False)

    def test_split_is_stratified_fixed_and_complete(self):
        train, val = split_validation_set(self.source)
        again_train, again_val = split_validation_set(self.source)
        self.assertEqual(len(train), 160)
        self.assertEqual(len(val), 40)
        self.assertEqual(train.label.value_counts().to_dict(), {0: 80, 1: 80})
        self.assertEqual(val.label.value_counts().to_dict(), {0: 20, 1: 20})
        self.assertEqual(train.id.tolist(), again_train.id.tolist())
        self.assertEqual(val.id.tolist(), again_val.id.tolist())
        self.assertFalse(set(train.id) & set(val.id))
        self.assertEqual(set(train.id) | set(val.id), set(range(200)))

    def test_export_creates_reproducible_raw_csvs(self):
        output = self.root / "splits"
        script = Path(__file__).resolve().parents[1] / "preprocess.py"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        args = [sys.executable, str(script), "--export-split", "--input", str(self.source), "--output-dir", str(output)]
        subprocess.run(args, cwd=self.root, check=True, env=env, capture_output=True, text=True)
        train_path, val_path = output / "train_split.csv", output / "validation_split.csv"
        self.assertTrue(train_path.exists())
        self.assertTrue(val_path.exists())
        first = (train_path.read_bytes(), val_path.read_bytes())
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        self.assertEqual(list(train.columns), ["id", "text", "label"])
        self.assertEqual(list(val.columns), ["id", "text", "label"])
        self.assertEqual(train.label.value_counts().to_dict(), {0: 80, 1: 80})
        self.assertEqual(val.label.value_counts().to_dict(), {0: 20, 1: 20})
        subprocess.run(args, cwd=self.root, check=True, env=env, capture_output=True, text=True)
        self.assertEqual(first, (train_path.read_bytes(), val_path.read_bytes()))

    def test_default_data_file_does_not_depend_on_current_directory(self):
        old_cwd = os.getcwd()
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(self.root)
        self.assertEqual(
            preprocess._data_file("train.csv"),
            Path(preprocess.__file__).resolve().parent / "data" / "train.csv",
        )

    def test_export_matches_rows_used_by_preprocess(self):
        test_path = self.root / "test.csv"
        pd.DataFrame({"id": [900], "text": ["test text"]}).to_csv(test_path, index=False)
        output = self.root / "splits"
        train_file, validation_file = preprocess.export_split(self.source, output)
        files = {"train.csv": self.source, "test.csv": test_path}
        with patch.object(preprocess, "_data_file", side_effect=lambda name: files[name]):
            train_text, train_label, val_text, val_label, _ = preprocess.preprocess_data(
                featurization=lambda a, b, c: (a, b, c), n=160
            )
        exported_train = pd.read_csv(train_file)
        exported_val = pd.read_csv(validation_file)
        self.assertEqual(train_text.tolist(), exported_train.text.tolist())
        self.assertEqual(train_label.tolist(), exported_train.label.tolist())
        self.assertEqual(val_text.tolist(), exported_val.text.tolist())
        self.assertEqual(val_label.tolist(), exported_val.label.tolist())

    def test_export_rejects_overwriting_its_source(self):
        source = self.root / "train_split.csv"
        source.write_bytes(self.source.read_bytes())
        original = source.read_bytes()
        with self.assertRaises(ValueError):
            preprocess.export_split(source, self.root)
        self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
