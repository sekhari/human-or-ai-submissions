"""Dataset loading and the shared Group 26 validation split.

The loader and the split logic are ported from Kevin's `kevin_train.py` with
their behaviour preserved exactly, because the shared validation IDs depend on
it. Two details matter and are easy to get wrong:

  * `load_labeled()` returns rows sorted by the **string** form of the id, so
    "10" sorts before "100". The validation file is only reproducible from
    seed 20261023 in that ordering.
  * CSVs are read with `encoding="utf-8-sig"` so a BOM, if present, does not
    end up glued to the first column name.

Nothing here fits a model or touches `test.csv` labels. Run this module
directly to verify the split end to end:

    python data_loading.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

import config

Row = dict[str, str]


# --- raw CSV loading --------------------------------------------------------

def _read_csv(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_labeled(path: Path | None = None) -> list[Row]:
    """Load train.csv, validated, sorted by STRING id.

    The sort is load-bearing: it is what makes `SPLIT_SEED` reproduce
    `group26_validation_ids.txt`.
    """
    if path is None:
        path = config.find_data_dir() / "train.csv"
    rows = _read_csv(path)

    if len(rows) != config.LABELED_ROWS:
        raise ValueError(
            f"Expected the full {config.LABELED_ROWS:,}-row labeled train.csv, "
            f"got {len(rows):,} rows from {path}."
        )
    if {row["label"] for row in rows} != {"0", "1"}:
        raise ValueError(
            f"Expected labels {{'0','1'}}, got "
            f"{sorted({row['label'] for row in rows})}."
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Training IDs must be unique.")

    return sorted(rows, key=lambda row: row["id"])


def load_unlabeled(name: str, expected_rows: int | None = None,
                   path: Path | None = None) -> list[Row]:
    """Load an unlabeled CSV (`test.csv`, `unlabeled.csv`) in file order.

    Row order is preserved, because `test.csv` order defines the required
    submission order.
    """
    if path is None:
        path = config.find_data_dir() / name
    rows = _read_csv(path)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows in {name}, got {len(rows):,}."
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{name} IDs must be unique.")
    return rows


def load_test(path: Path | None = None) -> list[Row]:
    return load_unlabeled("test.csv", config.TEST_ROWS, path)


# --- the shared validation split -------------------------------------------

def load_validation_ids(path: Path | None = None) -> list[str]:
    """Parse the shared ID file: one integer per line, no header.

    IDs are kept as strings so they compare directly against the CSV's `id`
    column without any int/str coercion ambiguity.
    """
    if path is None:
        path = config.VALIDATION_IDS
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Shared validation IDs not found at {path}. This file defines the "
            "team's holdout and is required."
        )
    ids = [line.strip() for line in Path(path).read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]

    if len(ids) != config.VALIDATION_ROWS:
        raise ValueError(
            f"Expected exactly {config.VALIDATION_ROWS:,} validation IDs, "
            f"got {len(ids):,}."
        )
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Duplicate validation IDs: {duplicates[:20]}")
    for line_no, row_id in enumerate(ids, start=1):
        if not row_id.lstrip("-").isdigit():
            raise ValueError(f"{path}:{line_no}: not an integer ID: {row_id!r}")
    return ids


def split_labeled(rows: list[Row] | None = None,
                  ids_path: Path | None = None) -> tuple[list[Row], list[Row]]:
    """Partition the labeled rows into (train, validation) by shared ID.

    Validation is exactly the IDs in the shared file; training is everything
    else. Every structural expectation is asserted rather than assumed.
    """
    if rows is None:
        rows = load_labeled()
    validation_ids = set(load_validation_ids(ids_path))

    known = {row["id"] for row in rows}
    missing = sorted(validation_ids - known)
    if missing:
        raise ValueError(
            f"{len(missing)} validation IDs are absent from train.csv, "
            f"e.g. {missing[:20]}"
        )

    train = [row for row in rows if row["id"] not in validation_ids]
    validation = [row for row in rows if row["id"] in validation_ids]

    expected_train = config.LABELED_ROWS - config.VALIDATION_ROWS
    if len(validation) != config.VALIDATION_ROWS:
        raise ValueError(
            f"Expected {config.VALIDATION_ROWS:,} validation rows, "
            f"got {len(validation):,}."
        )
    if len(train) != expected_train:
        raise ValueError(
            f"Expected {expected_train:,} training rows, got {len(train):,}."
        )
    if set(r["id"] for r in train) & validation_ids:
        raise ValueError("Train/validation ID overlap.")

    half = config.VALIDATION_ROWS // 2
    val_counts = Counter(row["label"] for row in validation)
    if val_counts != {"0": half, "1": half}:
        raise ValueError(
            f"Validation must be {half}/{half} per class, got {dict(val_counts)}."
        )
    train_counts = Counter(row["label"] for row in train)
    per_class = expected_train // 2
    if train_counts != {"0": per_class, "1": per_class}:
        raise ValueError(
            f"Training must be {per_class}/{per_class} per class, "
            f"got {dict(train_counts)}."
        )
    return train, validation


def regenerate_validation_ids(rows: list[Row] | None = None,
                              seed: int = config.SPLIT_SEED) -> list[str]:
    """Re-derive the validation IDs from the seed, as originally generated.

    Mirrors the original generator: rows already sorted by string id, then a
    stratified split holding out `VALIDATION_ROWS`. Returned sorted the same
    way the ID file is written (lexicographically, as strings).
    """
    if rows is None:
        rows = load_labeled()
    _, validation = train_test_split(
        rows,
        test_size=config.VALIDATION_ROWS,
        stratify=[row["label"] for row in rows],
        random_state=seed,
    )
    return sorted(row["id"] for row in validation)


def verify_split_provenance(rows: list[Row] | None = None,
                            ids_path: Path | None = None) -> dict:
    """Check that the seed reproduces the committed ID file.

    Also records what the *wrong* row orderings produce, so a future reader
    can see why the ordering matters instead of rediscovering it.
    """
    if rows is None:
        rows = load_labeled()
    supplied = set(load_validation_ids(ids_path))

    regenerated = set(regenerate_validation_ids(rows))

    # Contrast: the same seed on rows ordered numerically by id.
    numeric_order = sorted(rows, key=lambda row: int(row["id"]))
    _, numeric_val = train_test_split(
        numeric_order,
        test_size=config.VALIDATION_ROWS,
        stratify=[row["label"] for row in numeric_order],
        random_state=config.SPLIT_SEED,
    )

    return {
        "seed": config.SPLIT_SEED,
        "recipe": (
            "sort train.csv rows by str(id), then train_test_split("
            f"test_size={config.VALIDATION_ROWS}, stratify=labels, "
            f"random_state={config.SPLIT_SEED})"
        ),
        "matches_supplied_file": regenerated == supplied,
        "overlap_with_supplied": len(regenerated & supplied),
        "numeric_id_order_overlap_for_contrast": len(
            {row["id"] for row in numeric_val} & supplied
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the Group 26 dataset loading and validation split."
    )
    parser.add_argument("--train", type=Path, default=None,
                        help="Path to train.csv (default: autodetected).")
    parser.add_argument("--validation-ids", type=Path, default=None,
                        help="Path to group26_validation_ids.txt.")
    args = parser.parse_args()

    rows = load_labeled(args.train)
    print(f"loaded {len(rows):,} labeled rows, sorted by string id "
          f"(first ids: {[r['id'] for r in rows[:5]]})")

    train, validation = split_labeled(rows, args.validation_ids)
    print(f"split OK: {len(train):,} train / {len(validation):,} validation")
    print(f"  train labels      : {dict(Counter(r['label'] for r in train))}")
    print(f"  validation labels : {dict(Counter(r['label'] for r in validation))}")

    prov = verify_split_provenance(rows, args.validation_ids)
    status = "MATCHES" if prov["matches_supplied_file"] else "DOES NOT MATCH"
    print(f"\nprovenance: seed {prov['seed']} {status} the committed ID file "
          f"({prov['overlap_with_supplied']}/{config.VALIDATION_ROWS})")
    print(f"  for contrast, numeric id ordering would give "
          f"{prov['numeric_id_order_overlap_for_contrast']}/"
          f"{config.VALIDATION_ROWS} (chance)")
    if not prov["matches_supplied_file"]:
        raise SystemExit("Split provenance check FAILED.")

    import features
    sample = features.featurize([r["text"] for r in rows[:50]])
    print(f"\nfeatures: {sample.shape[1]} per passage "
          f"({features.N_FEATURES} expected), sample block {sample.shape}")
    print("all checks passed")


if __name__ == "__main__":
    main()
