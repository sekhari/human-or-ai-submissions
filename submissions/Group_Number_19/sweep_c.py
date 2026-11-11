"""Sweep classifier C for the TF-IDF baseline over models, SVD dims and tf/idf schemes.

Features are fit once per fold (on that fold's training rows only) and reused
for every C, since C only affects the classifier.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from baseline_tfidf import (
    IDF_SCHEMES, MODELS, TF_SCHEMES, add_ngram_args, build_features, make_classifier,
    ngram_kwargs,
)
from folds import N_FOLDS, make_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("../../../data/train.csv"))
    parser.add_argument("--Cs", type=float, nargs="+", default=[0.3, 1, 3, 10, 30])
    parser.add_argument("--svd-dims", type=int, nargs="+", default=[0, 300],
                        help="0 = plain TF-IDF")
    parser.add_argument("--tf", nargs="+", choices=TF_SCHEMES, default=["binary"])
    parser.add_argument("--idf", nargs="+", choices=IDF_SCHEMES, default=["smooth"])
    add_ngram_args(parser)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["logreg"])
    parser.add_argument("--out", type=Path, default=Path("sweep_c.csv"))
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    fold = make_folds(train)
    y = train["label"].to_numpy()
    feature_kwargs = ngram_kwargs(args)
    config = {
        "word_ngrams": "-".join(map(str, args.word_ngrams)),
        "char_ngrams": "-".join(map(str, args.char_ngrams)),
    }

    rows = []
    for tf, idf, svd_dim in itertools.product(args.tf, args.idf, args.svd_dims):
        for k in range(N_FOLDS):
            t0 = time.time()
            tr, va = fold != k, fold == k
            features = build_features(svd_dim, tf, idf, **feature_kwargs)
            X_tr = features.fit_transform(train.loc[tr, "text"])
            X_va = features.transform(train.loc[va, "text"])
            for model, C in itertools.product(args.models, args.Cs):
                clf = make_classifier(model, C).fit(X_tr, y[tr])
                p = clf.predict_proba(X_va)[:, 1]
                rows.append({**config, "n_features": X_tr.shape[1], "model": model,
                             "tf": tf, "idf": idf, "svd": svd_dim, "C": C, "fold": k,
                             "auc": roc_auc_score(y[va], p),
                             "logloss": log_loss(y[va], p),
                             "acc": accuracy_score(y[va], p > 0.5)})
            print(f"tf={tf} idf={idf} svd={svd_dim} fold {k} done in {time.time() - t0:.0f}s "
                  f"({X_tr.shape[1]} features)", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(args.out, index=False)
    summary = results.groupby(["word_ngrams", "char_ngrams", "model", "tf", "idf", "svd", "C"])[["auc", "logloss", "acc"]].agg(["mean", "std"])
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
