"""Compare text-representation choices with 5-fold CV (TF-IDF + logistic regression).

Example:
    python ablate_features.py --train data/train.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from features import build_vectorizer


def variants() -> dict:
    full = build_vectorizer()
    return {
        "sklearn default (word unigrams, lowercase, no punctuation)": TfidfVectorizer(),
        "word 1-2 only (normalized, cased, punctuation tokens)": full.transformer_list[0][1],
        "char 2-5 only (normalized, cased)": full.transformer_list[1][1],
        "word + char_wb 3-5 (lowercased)": build_vectorizer(lowercase=True, char_analyzer="char_wb", char_ngrams=(3, 5)),
        "word + char 2-5 (lowercased)": build_vectorizer(lowercase=True),
        "word + char 2-5 (cased)  <- final": full,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()

    df = pd.read_csv(args.train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for name, vec in variants().items():
        start = time.time()
        model = make_pipeline(vec, LogisticRegression(C=4.0, max_iter=2000))
        scores = cross_val_score(model, df["text"], df["label"], cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"{scores.mean():.4f} ± {scores.std():.4f}  ({time.time() - start:4.0f}s)  {name}", flush=True)


if __name__ == "__main__":
    main()
