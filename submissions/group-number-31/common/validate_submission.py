"""Check a Kaggle submission file before it is committed or uploaded.

  python common/validate_submission.py <file.csv> [<file.csv> ...] [--data-dir PATH]

A valid file has columns id,label, exactly the ids of sample_submission.csv (3,000, no
duplicates), and labels that are numbers in [0, 1] with no NaN. Exit code 1 if any file fails.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from data import load_csv


def problems(sub: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    out = []
    if list(sub.columns) != ["id", "label"]:
        return [f"columns are {list(sub.columns)}, expected ['id', 'label']"]
    if len(sub) != len(sample):
        out.append(f"{len(sub)} rows, expected {len(sample)}")
    if sub["id"].duplicated().any():
        out.append(f"{sub['id'].duplicated().sum()} duplicate ids")
    missing, extra = set(sample["id"]) - set(sub["id"]), set(sub["id"]) - set(sample["id"])
    if missing or extra:
        out.append(f"{len(missing)} ids missing, {len(extra)} unexpected ids")
    label = pd.to_numeric(sub["label"], errors="coerce")
    if label.isna().any():
        out.append(f"{label.isna().sum()} NaN / non-numeric labels")
    elif not np.all((label >= 0) & (label <= 1)):
        out.append(f"labels outside [0, 1] (min {label.min()}, max {label.max()})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    sample = load_csv("sample_submission", args.data_dir)
    ok = True
    for f in args.files:
        p = problems(pd.read_csv(f), sample)
        print(f"{f}: " + ("OK" if not p else "FAIL - " + "; ".join(p)))
        ok &= not p
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
