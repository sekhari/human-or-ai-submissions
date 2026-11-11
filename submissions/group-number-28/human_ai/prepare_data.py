"""Create the team's fixed split and local materialized CSVs; never overwrite IDs."""
import argparse
from collections import Counter
import json
import random

from .data import ROOT, SPLIT_DIR, load_partitions, raw_path, read_csv, sha256, write_csv

SEED = 42


def create_split(raw_dir=None, split_dir=SPLIT_DIR):
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = [split_dir / name for name in ("train_ids.csv", "validation_ids.csv", "manifest.json")]
    if any(path.exists() for path in paths):
        if not all(path.exists() for path in paths):
            raise ValueError("Incomplete split files; recover the canonical files from Git")
        return load_partitions(raw_dir, split_dir)
    source = raw_path("train.csv", raw_dir)
    rows = read_csv(source)
    if len(rows) != 10000 or Counter(row["label"] for row in rows) != {"0": 5000, "1": 5000}:
        raise ValueError("Expected 10,000 labeled rows: 5,000 per class")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Training IDs must be unique")
    texts = [" ".join(row["text"].split()) for row in rows]
    if any(not text for text in texts):
        raise ValueError("Empty training text")
    if len(set(texts)) != len(texts):
        raise ValueError("Duplicate normalized text: agree on a group-aware split before proceeding")
    # Sort before shuffling, then shuffle each class with the same seeded RNG.
    # The committed IDs, rather than a library-specific split implementation,
    # are the authoritative partition for all teammates.
    rng = random.Random(SEED)
    validation_ids = set()
    for label in ("0", "1"):
        ids = sorted(row["id"] for row in rows if row["label"] == label)
        rng.shuffle(ids)
        validation_ids.update(ids[:1000])
    counts = {}
    label_counts = {}
    hashes = {}
    for name in ("train", "validation"):
        selected = sorted(
            (row for row in rows if (row["id"] in validation_ids) == (name == "validation")),
            key=lambda row: row["id"],
        )
        path = split_dir / f"{name}_ids.csv"
        write_csv(path, ["id"], ({"id": row["id"]} for row in selected))
        counts[name] = len(selected)
        label_counts[name] = dict(Counter(row["label"] for row in selected))
        hashes[name] = sha256(path)
    manifest = {
        "version": 1,
        "seed": SEED,
        "algorithm": "lexicographically sorted string IDs; Python Random(42); shuffle class 0 then class 1; first 1000/class for validation",
        "source": "train.csv",
        "source_sha256": sha256(source),
        "counts": counts,
        "label_counts": label_counts,
        "split_sha256": hashes,
        "duplicate_audit": "No exact duplicates after whitespace normalization; near-duplicates and source leakage not assessed",
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return load_partitions(raw_dir, split_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", help="Directory containing the original competition CSVs")
    args = parser.parse_args()
    train, validation = create_split(args.raw_dir)
    target = ROOT / "data" / "processed"
    target.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        write_csv(target / f"{name}.csv", ["id", "text", "label"], rows)
    print(f"Verified canonical split: {len(train)} train / {len(validation)} validation")
    print(f"Local CSVs: {target}")


if __name__ == "__main__":
    main()

