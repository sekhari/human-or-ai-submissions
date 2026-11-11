"""LightGBM on style features, alone and stacked on the TF-IDF logistic regression.

For each outer fold (the fixed folds from folds.py):
  1. Fit TF-IDF on the outer-train text. Get out-of-fold logistic-regression
     scores for the outer-train rows with an inner 5-fold split, and score the
     validation rows with a model fit on all outer-train rows.
  2. Train LightGBM on [style features] and on [style features + LR score],
     early-stopping on a 10% holdout of the outer-train rows.
Validation rows are never used for fitting anything, including early stopping.

--strip-layout normalizes whitespace and markdown before TF-IDF and the style
features, and drops the layout-only style features (see layout.py).

--cv cluster replaces the fixed random folds with leave-cluster-out CV: each
topic cluster from clusters.py is held out in turn, so every score measures
generalization to an unseen genre.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from baseline_tfidf import build_features, make_classifier
from clusters import make_clusters, split_shares
from folds import SEED, make_folds
from style_features import style_features

LGBM_PARAMS = dict(
    objective="binary", learning_rate=0.03, n_estimators=5000, num_leaves=31,
    min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=SEED, verbose=-1,
)


def lr_scores(X_tr, y_tr, X_va, C: float) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold LR scores for the training rows, and scores for X_va.

    The TF-IDF vocabulary/idf is fit on all training rows (label-free); only the
    classifier is refit per inner fold.
    """
    oof = np.zeros(X_tr.shape[0])
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for itr, iva in inner.split(X_tr, y_tr):
        clf = make_classifier("logreg", C).fit(X_tr[itr], y_tr[itr])
        oof[iva] = clf.predict_proba(X_tr[iva])[:, 1]
    clf = make_classifier("logreg", C).fit(X_tr, y_tr)
    return oof, clf.predict_proba(X_va)[:, 1]


def fit_lgbm(F_tr: pd.DataFrame, y_tr, F_va: pd.DataFrame) -> tuple[np.ndarray, lgb.LGBMClassifier]:
    fit_idx, stop_idx = train_test_split(
        np.arange(len(F_tr)), test_size=0.1, stratify=y_tr, random_state=SEED)
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(F_tr.iloc[fit_idx], y_tr[fit_idx],
              eval_X=F_tr.iloc[stop_idx], eval_y=y_tr[stop_idx], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
    return model.predict_proba(F_va)[:, 1], model


def scores(y, p) -> dict[str, float]:
    return {"auc": roc_auc_score(y, p), "logloss": log_loss(y, p),
            "acc": accuracy_score(y, p > 0.5)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("../../../data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("../../../data/test.csv"),
                        help="test text, used (without labels) only to build clusters")
    parser.add_argument("--cv", choices=("random", "cluster"), default="random",
                        help="fixed random 5-fold, or leave-one-topic-cluster-out")
    parser.add_argument("--C", type=float, default=30.0, help="LR inverse L2 strength")
    parser.add_argument("--strip-layout", action="store_true",
                        help="remove newlines/markdown from the inputs")
    parser.add_argument("--out", type=Path, default=Path("lgbm_stack.csv"),
                        help="per-fold metrics")
    parser.add_argument("--oof-out", type=Path, default=Path("oof_lgbm_stack.csv"),
                        help="out-of-fold predictions for every train row, one column per model")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    if args.cv == "random":
        fold, names = make_folds(train), {}
    else:
        fold, names = make_clusters(train, pd.read_csv(args.test))
    groups = np.unique(fold)
    y = train["label"].to_numpy()
    layout = not args.strip_layout
    style = style_features(train["text"], layout=layout)
    print(f"{style.shape[1]} style features")

    oof = {name: np.zeros(len(train)) for name in ("logreg", "lgbm_style", "lgbm_style_lr")}
    rows, importances = [], []
    for k in groups:
        t0 = time.time()
        tr, va = fold != k, fold == k
        features = build_features(0, layout=layout)
        X_tr = features.fit_transform(train.loc[tr, "text"])
        X_va = features.transform(train.loc[va, "text"])
        lr_tr, lr_va = lr_scores(X_tr, y[tr], X_va, args.C)
        oof["logreg"][va] = lr_va

        S_tr, S_va = style[tr].reset_index(drop=True), style[va].reset_index(drop=True)
        oof["lgbm_style"][va], _ = fit_lgbm(S_tr, y[tr], S_va)
        oof["lgbm_style_lr"][va], model = fit_lgbm(
            S_tr.assign(lr_score=lr_tr), y[tr], S_va.assign(lr_score=lr_va))
        importances.append(pd.Series(
            model.booster_.feature_importance("gain"), index=model.booster_.feature_name()))

        for name, p in oof.items():
            rows.append({"model": name, "fold": k, **scores(y[va], p[va])})
        print(f"fold {k} done in {time.time() - t0:.0f}s (best iters: {model.best_iteration_})"
              + (f" [{names[k]}]" if names else ""), flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(args.out, index=False)
    pd.DataFrame({"id": train["id"], "fold": fold, "label": y,
                  **{f"pred_{name}": p for name, p in oof.items()}}).to_csv(
        args.oof_out, index=False)
    print(results.groupby("model")[["auc", "logloss", "acc"]].agg(["mean", "std"]).round(4).to_string())

    wide = results.pivot(index="fold", columns="model", values="auc")
    for name in ("lgbm_style", "lgbm_style_lr"):
        d = wide[name] - wide["logreg"]
        print(f"AUC {name} - logreg: mean {d.mean():+.4f}, better on {(d > 0).sum()}/{len(groups)} folds")

    if args.cv == "cluster":
        shares = split_shares()
        # Reweight train rows so the cluster mix matches test's.
        weight = pd.Series(fold).map(shares["test"] / shares["train"]).to_numpy()
        table = wide.assign(train_n=pd.Series(fold).value_counts(),
                            test_share=shares["test"], terms=pd.Series(names))
        print("\nAUC per held-out cluster:")
        print(table.round(4).to_string())
        for name, p in oof.items():
            print(f"{name:14s} pooled AUC {roc_auc_score(y, p):.4f} | "
                  f"test-mix weighted {roc_auc_score(y, p, sample_weight=weight):.4f}")

    imp = pd.concat(importances, axis=1).mean(axis=1)
    print("\nTop 15 features by gain (style + LR stack):")
    print((imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_string())


if __name__ == "__main__":
    main()
