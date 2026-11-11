"""Standard-library-only data loading and canonical split verification."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
ORIGINAL_DIR = ROOT / "cornell-tech-cs-5785-combined-2026-ai-vs-human-text"


def raw_path(name, raw_dir=None):
    if raw_dir is not None:
        return Path(raw_dir) / name
    canonical = ROOT / "data" / "raw" / name
    return canonical if canonical.exists() else ORIGINAL_DIR / name


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV rows: {path}")
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_partitions(raw_dir=None, split_dir=SPLIT_DIR):
    """Return train/validation records in the committed split's ID order.

    Text remains raw: fit vectorizers and other learned preprocessing on the
    training records only. All labels in these records are strings.
    """
    split_dir = Path(split_dir)
    manifest = json.loads((split_dir / "manifest.json").read_text())
    path = raw_path("train.csv", raw_dir)
    if sha256(path) != manifest["source_sha256"]:
        raise ValueError("train.csv differs from the canonical source; do not silently regenerate splits")
    records = read_csv(path)
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate training IDs")
    partitions = {}
    all_ids = []
    for name in ("train", "validation"):
        id_path = split_dir / f"{name}_ids.csv"
        if sha256(id_path) != manifest["split_sha256"][name]:
            raise ValueError(f"Modified canonical {name} IDs")
        ids = [row["id"] for row in read_csv(id_path)]
        if len(ids) != manifest["counts"][name]:
            raise ValueError(f"Incorrect {name} size")
        partitions[name] = [by_id[identifier] for identifier in ids]
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(by_id):
        raise ValueError("Split overlap, missing IDs, or duplicate IDs")
    return partitions["train"], partitions["validation"]

