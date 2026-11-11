"""Check prediction contracts without uploading anything to Kaggle."""
import argparse
import csv
import math

from .data import load_partitions, raw_path, read_csv


def check(path, kind, raw_dir=None):
    rows = read_csv(path)
    with open(path, newline="", encoding="utf-8-sig") as stream:
        columns = next(csv.reader(stream))
    expected_columns = ["id", "label"] if kind == "submission" else ["id", "label", "p_ai"]
    if columns != expected_columns:
        raise ValueError(f"Expected exactly {expected_columns}, got {columns}")
    reference = read_csv(raw_path("test.csv", raw_dir)) if kind == "submission" else load_partitions(raw_dir)[1]
    expected_count = 3000 if kind == "submission" else 2000
    if len(reference) != expected_count:
        raise ValueError(f"Expected {expected_count} reference rows, got {len(reference)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prediction IDs")
    if set(ids) != {row["id"] for row in reference}:
        raise ValueError("Missing or unexpected IDs")
    probability_column = "label" if kind == "submission" else "p_ai"
    for row in rows:
        value = float(row[probability_column])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid probability for ID {row['id']}")
    if kind == "validation":
        labels = {row["id"]: row["label"] for row in reference}
        if any(row["label"] != labels[row["id"]] for row in rows):
            raise ValueError("Validation labels disagree with train.csv")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--kind", choices=["submission", "validation"], required=True)
    parser.add_argument("--raw-dir")
    args = parser.parse_args()
    count = check(args.path, args.kind, args.raw_dir)
    print(f"Valid {args.kind}: {count} unique IDs, correct schema, finite probabilities in [0, 1]")


if __name__ == "__main__":
    main()

