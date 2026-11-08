"""Fixed stratified validation split used by every controlled experiment.

Usage:
    python -m src.split     # writes splits/val_ids.csv (2,000 rows: 1,000 human, 1,000 AI)

Documents with a near-duplicate anywhere in the data are never placed in validation, so no
validation passage has a near-copy in the training partition or in unlabeled.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_all, near_duplicate_pairs

VAL_IDS_PATH = Path(__file__).resolve().parents[1] / "splits" / "val_ids.csv"
VAL_PER_CLASS = 1000
SEED = 42


def make_val_ids(seed=SEED, threshold=0.8):
    df = load_all()
    pairs = near_duplicate_pairs(df.text, threshold)
    excluded = set(df.id.values[np.r_[pairs.i.values, pairs.j.values].astype(int)])
    train = df[(df.split == "train") & ~df.id.isin(excluded)]
    val = train.groupby("label").sample(VAL_PER_CLASS, random_state=seed)
    return val.id.sort_values().to_frame(), len(excluded)


def load_train_val():
    """Return (train_df, val_df): the 8,000-row training partition and the fixed validation set."""
    full = load_all()
    train = full[full.split == "train"]
    is_val = train.id.isin(pd.read_csv(VAL_IDS_PATH).id)
    return train[~is_val].reset_index(drop=True), train[is_val].reset_index(drop=True)


def main():
    ids, n_excluded = make_val_ids()
    VAL_IDS_PATH.parent.mkdir(exist_ok=True)
    ids.to_csv(VAL_IDS_PATH, index=False)
    tr, val = load_train_val()
    print(f"excluded near-duplicate docs: {n_excluded}")
    print(f"train: {len(tr)} rows, AI rate {tr.label.mean():.3f}; val: {len(val)} rows, AI rate {val.label.mean():.3f}")
    print(f"wrote {VAL_IDS_PATH}")


if __name__ == "__main__":
    main()
