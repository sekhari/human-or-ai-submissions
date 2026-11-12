"""Step 1 - shared train/val split. Everyone imports get_split; nothing is saved to disk."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_SIZE = 2000
TRAIN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train.csv")


def get_split(seed=SEED, train_csv=TRAIN_CSV):
    """Return (train_df, val_df): stratified 8,000 / 2,000 rows of train.csv.

    Deterministic for a given seed. Both frames keep original columns (id,text,label)
    and are sorted by original row order, with a fresh 0..n-1 index.
    """
    df = pd.read_csv(train_csv)
    train_idx, val_idx = train_test_split(
        df.index.to_numpy(), test_size=VAL_SIZE, stratify=df["label"], random_state=seed
    )
    train_df = df.loc[sorted(train_idx)].reset_index(drop=True)
    val_df = df.loc[sorted(val_idx)].reset_index(drop=True)
    assert len(train_df) == len(df) - VAL_SIZE and len(val_df) == VAL_SIZE
    return train_df, val_df


if __name__ == "__main__":
    tr, va = get_split()
    print(len(tr), len(va), tr["label"].mean(), va["label"].mean())
