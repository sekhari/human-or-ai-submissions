"""Classical baselines on the fixed validation split (report Q1).

Models (all scored on the same 2,000-row validation set):
  A. word TF-IDF + logistic regression
  B. character n-gram TF-IDF + logistic regression
  C. "weird character" / stylometric features + logistic regression

Hyperparameters are chosen by 3-fold stratified CV *inside the training
partition only*; the validation set is used once, for reporting.

Usage:
  python baselines.py                    # all three models
  python baselines.py --only char_tfidf_lr
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import SEED, Data, auc, save_preds, update_results, weird_feats


def word_pipe():
    return Pipeline([
        ("vec", TfidfVectorizer(sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=3000, solver="liblinear")),
    ])


def char_pipe():
    return Pipeline([
        ("vec", TfidfVectorizer(analyzer="char_wb", sublinear_tf=True, min_df=3, max_features=300_000)),
        ("clf", LogisticRegression(max_iter=3000, solver="liblinear")),
    ])


def weird_pipe():
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])


MODELS = {
    "word_tfidf_lr": (word_pipe, {"vec__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.3, 1, 3, 10]}, "text"),
    "char_tfidf_lr": (char_pipe, {"vec__ngram_range": [(2, 4), (3, 5)], "clf__C": [0.3, 1, 3, 10]}, "text"),
    "weird_char_lr": (weird_pipe, {"clf__C": [0.1, 1, 10]}, "weird"),
}


def run(name: str, data: Data, n_jobs: int):
    make_pipe, grid, kind = MODELS[name]
    if kind == "text":
        Xtr, Xva = data.X_tr, data.X_va
    else:
        Xtr, Xva = weird_feats(data.X_tr), weird_feats(data.X_va)
    gs = GridSearchCV(make_pipe(), grid, scoring="roc_auc", n_jobs=n_jobs, refit=True,
                      cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
    t0 = time.time()
    gs.fit(Xtr, data.y_tr)
    search_time = time.time() - t0
    best = gs.best_estimator_
    p_tr, p_va = best.predict_proba(Xtr)[:, 1], best.predict_proba(Xva)[:, 1]
    best_params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()}
    cv_table = [
        {"params": {k: str(v) for k, v in p.items()}, "cv_auc": float(s)}
        for p, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"])
    ]
    save_preds(name, p_tr=p_tr, p_va=p_va)
    out = dict(train_auc=auc(data.y_tr, p_tr), val_auc=auc(data.y_va, p_va),
               train_time_s=float(gs.refit_time_), search_time_s=search_time,
               best_params=best_params, cv_table=cv_table)
    update_results(f"model:{name}", out)
    print(f"{name}: train AUROC {out['train_auc']:.4f} | val AUROC {out['val_auc']:.4f} | "
          f"refit {out['train_time_s']:.1f}s | best {best_params}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(MODELS), default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    data = Data()
    update_results("split", dict(seed=SEED, n_train=len(data.tr), n_val=len(data.va),
                                 dups_removed=data.n_dups_removed,
                                 val_ai_frac=float(data.y_va.mean())))
    print(f"train {len(data.tr)} | val {len(data.va)} | exact dups removed {data.n_dups_removed}")
    for name in ([args.only] if args.only else MODELS):
        run(name, data, args.n_jobs)


if __name__ == "__main__":
    main()
