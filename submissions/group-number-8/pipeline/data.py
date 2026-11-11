"""Loading and the fixed stratified validation split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import SPLIT_SEED, VAL_SIZE


def load_split(path: Path, labeled: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"id", "text"} | ({"label"} if labeled else set())
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "id": df["id"].astype(int),
            "text": df["text"].fillna("").astype(str),
        }
    )
    if labeled:
        out["label"] = df["label"].astype(int)
        if not set(out["label"].unique()).issubset({0, 1}):
            raise ValueError(f"Expected labels 0 and 1 in {path}")
    return out


def validation_split(
    df: pd.DataFrame, seed: int = SPLIT_SEED, val_size: int = VAL_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_partition, val_partition). Validation is fixed by seed."""
    if len(df) <= val_size:
        raise ValueError(
            f"Need more than {val_size} labeled rows to hold out validation; got {len(df)}"
        )
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["label"].to_numpy(),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def exact_text_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[int, int]:
    """Count training rows that copy a validation passage.

    Returns (exact stripped match, case-folded stripped match).
    """
    val_text = set(val_df["text"].str.strip())
    exact = int(train_df["text"].str.strip().isin(val_text).sum())
    val_folded = set(val_df["text"].str.strip().str.casefold())
    folded = int(train_df["text"].str.strip().str.casefold().isin(val_folded).sum())
    return exact, folded


def stratified_prefix(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """First n rows of one seeded shuffle within each class.

    The same seed and a larger n returns a superset, so the learning curve
    adds examples instead of drawing a new sample.
    """
    if n > len(df):
        raise ValueError(f"Cannot take {n} rows from a partition of {len(df)}")
    if n == len(df):
        return df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    labels = sorted(int(x) for x in df["label"].unique())
    counts = df["label"].value_counts().to_dict()
    raw = {label: n * (counts[label] / len(df)) for label in labels}
    alloc = {label: int(np.floor(raw[label])) for label in labels}
    remainder = n - sum(alloc.values())
    order = sorted(labels, key=lambda lab: (raw[lab] - alloc[lab], -lab), reverse=True)
    for label in order[:remainder]:
        alloc[label] += 1
    pieces: list[np.ndarray] = []
    for label in labels:
        idx = df.index[df["label"] == label].to_numpy().copy()
        rng.shuffle(idx)
        pieces.append(idx[: alloc[label]])
    return df.loc[np.concatenate(pieces)].reset_index(drop=True)
