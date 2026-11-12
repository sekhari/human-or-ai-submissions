"""Linear SVM and logistic regression on the blank-line-aware (v2) word + char TF-IDF.

Same models, C grids and validation protocol as train_svm.py and train_logreg.py. The only change
is the normalizer: features_v2.normalize_text_v2 keeps a blank line distinct from a single line
break. The TF-IDF is built once and shared by both models.

Example:
    python train_linear_v2.py --train data/train.csv --unlabeled data/unlabeled.csv \
        --test data/test.csv

Writes results_{svm,logreg}_v2.json and, with --test, submission_{svm,logreg}_v2.csv to --out-dir.
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

import train_logreg
import train_svm
from features_v2 import build_vectorizer_v2
from split import SEED, load_split

REPRESENTATION = "blank-line-aware (v2) word 1-2 + char 2-5 TF-IDF (300k features)"


def make_svm(c: float) -> LinearSVC:
    # Same settings as train_svm.py.
    return LinearSVC(C=c, dual="auto", max_iter=5000, random_state=SEED)


# key -> (description, model factory, C grid, function giving one ranking score per row)
MODELS = {
    "svm": ("Linear SVM (LinearSVC, squared hinge loss, L2)", make_svm, train_svm.C_GRID,
            lambda model, X: model.decision_function(X)),
    "logreg": ("Logistic regression (L2, liblinear)", train_logreg.make_model, train_logreg.C_GRID,
               lambda model, X: model.predict_proba(X)[:, 1]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate linear SVM and logistic regression on v2 TF-IDF")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--unlabeled", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_all = pd.read_csv(args.train)
    unlabeled_texts = pd.read_csv(args.unlabeled)["text"].tolist() if args.unlabeled else []
    tr, va = load_split(train_all)
    print(f"train {len(tr):,} rows | val {len(va):,} rows")

    # Vocabulary/IDF are fit on train + unlabeled text only (no labels, no validation text).
    start = time.time()
    vectorizer = build_vectorizer_v2().fit(tr["text"].tolist() + unlabeled_texts)
    X_tr = vectorizer.transform(tr["text"])
    X_va = vectorizer.transform(va["text"])
    feature_time = time.time() - start
    print(f"features {X_tr.shape[1]:,} cols in {feature_time:.0f}s")

    best_c = {}
    for key, (description, make_model, c_grid, score) in MODELS.items():
        search = GridSearchCV(
            make_model(1.0),
            {"C": c_grid},
            scoring="roc_auc",
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            n_jobs=-1,
        )
        search.fit(X_tr, tr["label"])
        cv_by_c = dict(zip(c_grid, search.cv_results_["mean_test_score"]))
        for c, s in cv_by_c.items():
            print(f"  {key} C={c:<5} CV AUROC {s:.4f}")
        best_c[key] = search.best_params_["C"]

        start = time.time()
        model = make_model(best_c[key]).fit(X_tr, tr["label"])
        fit_time = time.time() - start

        results = {
            "model": description,
            "representation": REPRESENTATION,
            "C": best_c[key],
            "cv_auroc_by_C": {str(c): round(float(s), 4) for c, s in cv_by_c.items()},
            "train_auroc": round(roc_auc_score(tr["label"], score(model, X_tr)), 6),
            "val_auroc": round(roc_auc_score(va["label"], score(model, X_va)), 6),
            "train_time_s_model_fit": round(fit_time, 1),
            "train_time_s_features_plus_fit": round(feature_time + fit_time, 1),
        }
        print(json.dumps(results, indent=2))
        (args.out_dir / f"results_{key}_v2.json").write_text(json.dumps(results, indent=2))

    if args.test:
        test = pd.read_csv(args.test)
        vectorizer = build_vectorizer_v2().fit(train_all["text"].tolist() + unlabeled_texts)
        X_all = vectorizer.transform(train_all["text"])
        X_test = vectorizer.transform(test["text"])
        for key, (_, make_model, _, score) in MODELS.items():
            scores = score(make_model(best_c[key]).fit(X_all, train_all["label"]), X_test)
            if key == "svm":
                scores = 1 / (1 + np.exp(-scores))  # monotonic, so AUROC is unchanged
            path = args.out_dir / f"submission_{key}_v2.csv"
            pd.DataFrame({"id": test["id"], "label": scores}).to_csv(path, index=False)
            print(f"wrote {path} (refit on all {len(train_all):,} labeled rows)")


if __name__ == "__main__":
    main()
