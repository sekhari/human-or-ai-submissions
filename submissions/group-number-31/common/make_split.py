"""Create (or verify) the fixed stratified 8,000 / 2,000 split of train.csv.

  python common/make_split.py [--data-dir PATH]

Writes common/val_ids.csv (id,label of the 2,000 validation rows, sorted by id; no text).
If the file already exists, the split is recomputed and compared instead: the script
fails if anything differs, so a teammate can check their data and environment reproduce
the committed split exactly. --overwrite replaces the file (don't: every result depends on it).
"""
from __future__ import annotations

import argparse
import hashlib
import sys

from sklearn.model_selection import train_test_split

from data import N_VAL, SEED, VAL_IDS_FILE, load_csv


def make_split(data_dir=None):
    train = load_csv("train", data_dir)
    _, val = train_test_split(train[["id", "label"]], test_size=N_VAL, stratify=train["label"],
                              random_state=SEED, shuffle=True)
    return val.sort_values("id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    csv = make_split(args.data_dir).to_csv(index=False, lineterminator="\n")
    digest = hashlib.sha256(csv.encode()).hexdigest()[:16]
    if VAL_IDS_FILE.exists() and not args.overwrite:
        old = VAL_IDS_FILE.read_text(encoding="utf-8")
        if old != csv:
            sys.exit(f"MISMATCH: recomputed split (sha256 {digest}) differs from {VAL_IDS_FILE}. "
                     "Check train.csv and your scikit-learn version; do not overwrite the committed file.")
        print(f"OK: {VAL_IDS_FILE.name} reproduced exactly (sha256 {digest})")
        return
    VAL_IDS_FILE.write_text(csv, encoding="utf-8", newline="\n")
    n = csv.count("\n") - 1
    print(f"Wrote {VAL_IDS_FILE} ({n} rows, seed {SEED}, sha256 {digest})")


if __name__ == "__main__":
    main()
