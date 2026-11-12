"""Logistic regression on word + char TF-IDF, evaluated on the fixed validation partition.

Example:
    python train_logreg.py --train data/train.csv --unlabeled data/unlabeled.csv \
        --test data/test.csv --submission submission_logreg.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from features import build_vectorizer
from split import SEED, load_split

# Inverse regularization strength: smaller C = stronger regularization.
C_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def fit_features(texts: list[str]):
    vectorizer = build_vectorizer()
    vectorizer.fit(texts)
    return vectorizer


def make_model(c: float) -> LogisticRegression:
    # L2-regularized LR. liblinear is fast and reliable on sparse, high-dimensional TF-IDF.
    return LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=SEED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate logistic regression")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--unlabeled", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--submission", type=Path, default=Path("submission_logreg.csv"))
    parser.add_argument("--results", type=Path, default=Path("results_logreg.json"))
    args = parser.parse_args()

    train_all = pd.read_csv(args.train)
    unlabeled_texts = pd.read_csv(args.unlabeled)["text"].tolist() if args.unlabeled else []
    tr, va = load_split(train_all)
    print(f"train {len(tr):,} rows | val {len(va):,} rows")

    # Vectorizer is fit on train + unlabeled TEXT only (no labels used, no validation text).
    start = time.time()
    vectorizer = fit_features(tr["text"].tolist() + unlabeled_texts)
    X_tr = vectorizer.transform(tr["text"])
    X_va = vectorizer.transform(va["text"])
    feature_time = time.time() - start
    print(f"features {X_tr.shape[1]:,} cols in {feature_time:.0f}s")

    # Choose C by 5-fold CV on the training partition only (validation set is untouched).
    search = GridSearchCV(
        make_model(1.0),
        {"C": C_GRID},
        scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )
    search.fit(X_tr, tr["label"])
    for c, s in zip(C_GRID, search.cv_results_["mean_test_score"]):
        print(f"  C={c:<5} CV AUROC {s:.4f}")
    best_c = search.best_params_["C"]

    start = time.time()
    model = make_model(best_c).fit(X_tr, tr["label"])
    fit_time = time.time() - start

    results = {
        "model": "Logistic regression (L2, liblinear)",
        "representation": "word 1-2 + char 2-5 TF-IDF (300k features)",
        "C": best_c,
        "cv_auroc_by_C": {str(c): round(float(s), 4) for c, s in zip(C_GRID, search.cv_results_["mean_test_score"])},
        "train_auroc": roc_auc_score(tr["label"], model.predict_proba(X_tr)[:, 1]),
        "val_auroc": roc_auc_score(va["label"], model.predict_proba(X_va)[:, 1]),
        "train_time_s_model_fit": round(fit_time, 1),
        "train_time_s_features_plus_fit": round(feature_time + fit_time, 1),
    }
    print(json.dumps(results, indent=2))
    args.results.write_text(json.dumps(results, indent=2))

    if args.test:
        test = pd.read_csv(args.test)
        vectorizer = fit_features(train_all["text"].tolist() + unlabeled_texts)
        final = make_model(best_c)
        final.fit(vectorizer.transform(train_all["text"]), train_all["label"])
        probs = final.predict_proba(vectorizer.transform(test["text"]))[:, 1]
        pd.DataFrame({"id": test["id"], "label": probs}).to_csv(args.submission, index=False)
        print(f"wrote {args.submission} (refit on all {len(train_all):,} labeled rows)")


if __name__ == "__main__":
    main()