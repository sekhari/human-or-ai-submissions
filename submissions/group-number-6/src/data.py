"""Data loading and the fixed stratified validation split.

The split is created once (seed 20261102) and saved to ``splits/val_ids.txt`` so that
every experiment uses exactly the same 2,000 held-out rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_FILE = ROOT / "splits" / "val_ids.txt"
SEED = 20261102
VAL_SIZE = 2000


def load_csv(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    d = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(d / name)
    df["text"] = df["text"].astype(str)
    return df


def make_split(train: pd.DataFrame, seed: int = SEED, val_size: int = VAL_SIZE) -> np.ndarray:
    """Return the ids of a stratified ``val_size``-row validation set."""
    _, val_idx = train_test_split(
        np.arange(len(train)), test_size=val_size, stratify=train["label"], random_state=seed
    )
    return np.sort(train["id"].to_numpy()[val_idx])


def get_split(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and return (train_part, val_part) using the saved split."""
    train = load_csv("train.csv", data_dir)
    if SPLIT_FILE.exists():
        val_ids = np.loadtxt(SPLIT_FILE, dtype=int)
    else:
        val_ids = make_split(train)
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(SPLIT_FILE, val_ids, fmt="%d")
    is_val = train["id"].isin(val_ids)
    return train[~is_val].reset_index(drop=True), train[is_val].reset_index(drop=True)
