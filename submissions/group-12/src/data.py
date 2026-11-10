"""Shared data loading and the fixed validation split used by every report experiment."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"
VAL_IDS_PATH = SPLIT_DIR / "val_ids.csv"

SPLIT_SEED = 42
VAL_SIZE = 2000


def load_train():
    return pd.read_csv(DATA_DIR / "train.csv")


def load_unlabeled():
    return pd.read_csv(DATA_DIR / "unlabeled.csv")


def load_test():
    return pd.read_csv(DATA_DIR / "test.csv")


def load_split():
    """Return (train_df, val_df) using the committed val_ids.csv (run `python -m src.split` once)."""
    train = load_train()
    val_ids = set(pd.read_csv(VAL_IDS_PATH)["id"])
    is_val = train["id"].isin(val_ids)
    assert is_val.sum() == VAL_SIZE, f"expected {VAL_SIZE} validation rows, got {is_val.sum()}"
    return train[~is_val].reset_index(drop=True), train[is_val].reset_index(drop=True)
