"""Data loading and shared validation splits.

Run once to create splits/splits.csv (commit that file so the whole team
uses identical splits):

    python src/data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_PATH = ROOT / "splits" / "splits.csv"

SEED = 42
VAL_FRAC = 0.2
N_CLUSTERS = 10


def load_train():
    return pd.read_csv(DATA_DIR / "train.csv")


def load_unlabeled():
    return pd.read_csv(DATA_DIR / "unlabeled.csv")


def load_test():
    return pd.read_csv(DATA_DIR / "test.csv")


def make_splits(train):
    """Create two splits over train.csv.

    random_fold: stratified 80/20 random split.
    ood_fold:    texts clustered by topic/style; whole clusters are held out
                 (~20% of data) to mimic the test set's distribution shift.
    """
    splits = pd.DataFrame({"id": train["id"], "label": train["label"]})

    # Random stratified split
    _, val_idx = train_test_split(
        np.arange(len(train)), test_size=VAL_FRAC,
        stratify=train["label"], random_state=SEED,
    )
    splits["random_fold"] = "train"
    splits.loc[val_idx, "random_fold"] = "val"

    # Topic/style clusters (labels are NOT used for clustering)
    tfidf = TfidfVectorizer(max_features=20000, stop_words="english",
                            sublinear_tf=True, min_df=3)
    X = tfidf.fit_transform(train["text"])
    n_comp = min(100, X.shape[1] - 1)
    X = TruncatedSVD(n_components=n_comp, random_state=SEED).fit_transform(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    clusters = KMeans(n_clusters=N_CLUSTERS, n_init=10,
                      random_state=SEED).fit_predict(X)
    splits["cluster"] = clusters

    # Hold out whole clusters until we reach ~VAL_FRAC of the data.
    rng = np.random.default_rng(SEED)
    held_out, n_held = [], 0
    for c in rng.permutation(N_CLUSTERS):
        if n_held >= VAL_FRAC * len(train):
            break
        held_out.append(c)
        n_held += (clusters == c).sum()
    splits["ood_fold"] = np.where(np.isin(clusters, held_out), "val", "train")
    assert splits.loc[splits["ood_fold"] == "val", "label"].nunique() == 2, \
        "OOD val set is single-class; change SEED or N_CLUSTERS"

    return splits


def load_splits():
    return pd.read_csv(SPLITS_PATH)


def get_split(train, splits, fold_col):
    """Return (train_df, val_df) for fold_col in {'random_fold', 'ood_fold'}."""
    df = train.merge(splits[["id", fold_col]], on="id")
    return df[df[fold_col] == "train"], df[df[fold_col] == "val"]


if __name__ == "__main__":
    train = load_train()
    splits = make_splits(train)
    SPLITS_PATH.parent.mkdir(exist_ok=True)
    splits.to_csv(SPLITS_PATH, index=False)

    print(f"Saved {SPLITS_PATH.relative_to(ROOT)}")
    for col in ["random_fold", "ood_fold"]:
        val = splits[splits[col] == "val"]
        print(f"{col}: {len(val)} val rows, AI fraction {val['label'].mean():.2f}")
    print("\nCluster sizes and AI fraction:")
    print(splits.groupby("cluster")["label"].agg(["size", "mean"]).round(2))
