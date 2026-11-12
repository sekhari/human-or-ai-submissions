"""
EDA for the CS5785 AI vs Human text competition.

Usage:  python eda.py --data data/
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def load(data_dir):
    train = pd.read_csv(f"{data_dir}/train.csv")
    unlab = pd.read_csv(f"{data_dir}/unlabeled.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, unlab, test


def basic_stats(train, unlab, test):
    print("=== 1. size, labels, length ===")
    for name, df in [("train", train), ("unlabeled", unlab), ("test", test)]:
        n_words = df["text"].str.split().str.len()
        print(f"{name:10s} rows={len(df):5d}  words: mean={n_words.mean():.0f} "
              f"median={n_words.median():.0f} min={n_words.min()} max={n_words.max()}")
    print("\nlabel counts in train:")
    print(train["label"].value_counts().to_string())
    print("\nwords per passage by label:")
    print(train.groupby("label")["text"].apply(lambda s: s.str.split().str.len().describe())
          .unstack()[["mean", "50%", "std"]].round(1))


def format_check(train, unlab, test):
    # share of passages that contain each pattern at least once
    print("\n=== 2. formatting patterns (share of passages containing it) ===")
    patterns = {
        "blank line (\\n\\n)": r"\n\n",
        "markdown #": "#",
        "em dash —": "—",
        "slash /": "/",
        "any digit": r"\d",
    }
    rows = []
    for name, pat in patterns.items():
        rows.append({
            "pattern": name,
            "train human": (train.loc[train.label == 0, "text"].str.contains(pat)).mean(),
            "train AI": (train.loc[train.label == 1, "text"].str.contains(pat)).mean(),
            "unlabeled": unlab["text"].str.contains(pat).mean(),
            "test": test["text"].str.contains(pat).mean(),
        })
    print(pd.DataFrame(rows).set_index("pattern").round(3))


def duplicate_check(train, unlab, test, threshold=0.9):
    print("\n=== 3. duplicates / near-duplicates ===")
    norm = lambda s: " ".join(s.lower().split())
    train_norm = train["text"].map(norm)
    print("exact duplicates inside train:", train_norm.duplicated().sum())
    print("test passages also in train:", test["text"].map(norm).isin(set(train_norm)).sum())

    # near-duplicates: cosine similarity between TF-IDF vectors
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True).fit(train["text"])
    X = vec.transform(train["text"])
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    sim = 1 - dist[:, 1]  # column 0 is the passage itself
    pairs = {tuple(sorted((i, idx[i, 1]))) for i in np.where(sim > threshold)[0]}
    print(f"near-duplicate pairs inside train (cos > {threshold}): {len(pairs)}")
    for i, j in list(pairs)[:3]:
        print(f"  id {train.id[i]} (label {train.label[i]}) ~ id {train.id[j]} (label {train.label[j]})")
        print("   ", repr(train.text[i][:120]))
        print("   ", repr(train.text[j][:120]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    train, unlab, test = load(args.data)
    basic_stats(train, unlab, test)
    format_check(train, unlab, test)
    duplicate_check(train, unlab, test)


if __name__ == "__main__":
    main()
