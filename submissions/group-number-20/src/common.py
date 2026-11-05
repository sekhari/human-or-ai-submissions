"""Shared configuration, data loading, and evaluation helpers."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

SEED = 20261023
VAL_IDS = Path(__file__).resolve().parents[1] / "splits" / "val_ids.txt"

def load_train(path):
    return pd.read_csv(path)

def split(df):
    val_ids = set(int(x) for x in VAL_IDS.read_text().split())
    m = df["id"].isin(val_ids).values
    return df[~m].reset_index(drop=True), df[m].reset_index(drop=True)

def baseline(C=1.0):
    return make_pipeline(
        TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                        min_df=2, max_df=0.95, sublinear_tf=True),
        LogisticRegression(C=C, max_iter=2000, random_state=SEED),
    )

def auroc_ci(y, p, n=2000, seed=SEED):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            s.append(roc_auc_score(y[i], p[i]))
    return np.percentile(s, [2.5, 97.5])
