"""Meaningful checks for shared data/ID/probability contracts, not model scores."""
import json
import tempfile
from pathlib import Path
import unittest

from human_ai.check_predictions import check
from human_ai.data import load_partitions, sha256, write_csv
from human_ai.prepare_data import create_split


class SplitContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.split = self.root / "splits"
        rows = [
            {"id": str(i), "text": f"unique example {i}", "label": str(i % 2)}
            for i in range(10000)
        ]
        write_csv(self.raw / "train.csv", ["id", "text", "label"], rows)

    def test_balanced_disjoint_complete_and_repeatable(self):
        train, validation = create_split(self.raw, self.split)
        self.assertEqual((len(train), len(validation)), (8000, 2000))
        self.assertEqual(sum(row["label"] == "1" for row in validation), 1000)
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in validation})
        self.assertEqual(len({r["id"] for r in train + validation}), 10000)
        hashes = [sha256(self.split / name) for name in ("train_ids.csv", "validation_ids.csv")]
        self.assertEqual((train, validation), create_split(self.raw, self.split))
        other = self.root / "another_split"
        create_split(self.raw, other)
        self.assertEqual(hashes, [sha256(other / name) for name in ("train_ids.csv", "validation_ids.csv")])

    def test_modified_source_is_rejected(self):
        create_split(self.raw, self.split)
        with (self.raw / "train.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "canonical source"):
            load_partitions(self.raw, self.split)

    def test_modified_split_is_rejected(self):
        create_split(self.raw, self.split)
        with (self.split / "validation_ids.csv").open("a") as stream:
            stream.write("extra-id\n")
        with self.assertRaisesRegex(ValueError, "Modified canonical"):
            load_partitions(self.raw, self.split)

    def test_whitespace_normalized_duplicates_are_rejected(self):
        path = self.raw / "train.csv"
        text = path.read_text().replace("unique example 1,1\n", "unique   example 0,1\n")
        path.write_text(text)
        with self.assertRaisesRegex(ValueError, "Duplicate normalized text"):
            create_split(self.raw, self.split)

    def test_manifest_reports_class_balance(self):
        create_split(self.raw, self.split)
        manifest = json.loads((self.split / "manifest.json").read_text())
        self.assertEqual(manifest["label_counts"]["train"], {"0": 4000, "1": 4000})


class PredictionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prediction = self.root / "predictions.csv"
        write_csv(self.root / "test.csv", ["id", "text"],
                  ({"id": str(i), "text": "example"} for i in range(3000)))
        self.rows = [{"id": str(i), "label": "0.25"} for i in range(3000)]

    def save_and_check(self):
        write_csv(self.prediction, ["id", "label"], self.rows)
        return check(self.prediction, "submission", self.root)

    def test_reordered_predictions_are_valid_by_id(self):
        self.rows.reverse()
        self.assertEqual(self.save_and_check(), 3000)

    def test_duplicate_id_is_rejected(self):
        self.rows[-1]["id"] = "0"
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            self.save_and_check()

    def test_missing_id_is_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "Missing or unexpected"):
            self.save_and_check()

    def test_nonfinite_and_out_of_range_are_rejected(self):
        for value in ("nan", "inf", "-0.1", "1.01"):
            with self.subTest(value=value):
                self.rows[0]["label"] = value
                with self.assertRaisesRegex(ValueError, "Invalid probability"):
                    self.save_and_check()

    def test_extra_column_is_rejected(self):
        write_csv(self.prediction, ["id", "label", "text"],
                  ({**row, "text": "example"} for row in self.rows))
        with self.assertRaisesRegex(ValueError, "Expected exactly"):
            check(self.prediction, "submission", self.root)


if __name__ == "__main__":
    unittest.main()
