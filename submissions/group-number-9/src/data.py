"""Data loading helpers shared by all experiments."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
SPLITS = ("train", "unlabeled", "test")


def load_split(name, data_dir=DATA_DIR):
    return pd.read_csv(Path(data_dir) / f"{name}.csv")


def load_all(data_dir=DATA_DIR):
    """Return one DataFrame with columns id, text, label (NaN if hidden), split."""
    frames = []
    for name in SPLITS:
        df = load_split(name, data_dir)
        df["split"] = name
        if "label" not in df:
            df["label"] = float("nan")
        frames.append(df[["id", "text", "label", "split"]])
    return pd.concat(frames, ignore_index=True)


def near_duplicate_pairs(texts, threshold=0.8, chunk=1000):
    """Positional pairs (i < j) whose word 1-2gram TF-IDF cosine is >= threshold."""
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32).fit_transform(texts)
    pairs = []
    for start in range(0, X.shape[0], chunk):
        S = (X[start:start + chunk] @ X.T).toarray()
        for i_local, row in enumerate(S):
            i = start + i_local
            row[: i + 1] = 0  # upper triangle only, skip self
            for j in np.flatnonzero(row >= threshold):
                pairs.append((i, j, row[j]))
    return pd.DataFrame(pairs, columns=["i", "j", "cosine"])
