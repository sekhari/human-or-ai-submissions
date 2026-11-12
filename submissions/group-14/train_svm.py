"""Linear SVM on word + char TF-IDF, evaluated on the fixed validation partition.

Example:
    python train_svm.py --train data/train.csv --unlabeled data/unlabeled.csv \
        --test data/test.csv --submission submission_svm.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC

from features import build_vectorizer
from split import SEED, load_split

C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]


def fit_features(texts: list[str]):
    vectorizer = build_vectorizer()
    vectorizer.fit(texts)
    return vectorizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate linear SVM")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--unlabeled", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--submission", type=Path, default=Path("submission_svm.csv"))
    parser.add_argument("--results", type=Path, default=Path("results_svm.json"))
    args = parser.parse_args()

    train_all = pd.read_csv(args.train)
    unlabeled_texts = pd.read_csv(args.unlabeled)["text"].tolist() if args.unlabeled else []
    tr, va = load_split(train_all)
    print(f"train {len(tr):,} rows | val {len(va):,} rows")

    start = time.time()
    vectorizer = fit_features(tr["text"].tolist() + unlabeled_texts)
    X_tr = vectorizer.transform(tr["text"])
    X_va = vectorizer.transform(va["text"])
    feature_time = time.time() - start
    print(f"features {X_tr.shape[1]:,} cols in {feature_time:.0f}s")

    search = GridSearchCV(
        LinearSVC(dual="auto", max_iter=5000, random_state=SEED),
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
    model = LinearSVC(C=best_c, dual="auto", max_iter=5000, random_state=SEED).fit(X_tr, tr["label"])
    fit_time = time.time() - start

    results = {
        "model": "Linear SVM (LinearSVC, squared hinge loss, L2)",
        "representation": "word 1-2 + char 2-5 TF-IDF (300k features)",
        "C": best_c,
        "train_auroc": roc_auc_score(tr["label"], model.decision_function(X_tr)),
        "val_auroc": roc_auc_score(va["label"], model.decision_function(X_va)),
        "train_time_s_model_fit": round(fit_time, 1),
        "train_time_s_features_plus_fit": round(feature_time + fit_time, 1),
    }
    print(json.dumps(results, indent=2))
    args.results.write_text(json.dumps(results, indent=2))

    if args.test:
        test = pd.read_csv(args.test)
        vectorizer = fit_features(train_all["text"].tolist() + unlabeled_texts)
        final = LinearSVC(C=best_c, dual="auto", max_iter=5000, random_state=SEED)
        final.fit(vectorizer.transform(train_all["text"]), train_all["label"])
        scores = final.decision_function(vectorizer.transform(test["text"]))
        pd.DataFrame({"id": test["id"], "label": 1 / (1 + np.exp(-scores))}).to_csv(args.submission, index=False)
        print(f"wrote {args.submission} (refit on all {len(train_all):,} labeled rows)")


if __name__ == "__main__":
    main()
