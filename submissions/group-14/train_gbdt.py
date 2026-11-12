"""Gradient-boosted trees on stylometric + LSA features, evaluated on the fixed validation partition.

Features (see features_v2.py):
  * 37 stylometric statistics (layout, length, vocabulary, punctuation)
  * 200-d LSA: TruncatedSVD of the blank-line-aware (v2) word 1-2 + char 2-5 TF-IDF.
    TF-IDF and SVD are unsupervised and fit on training text + unlabeled.csv only.

Example:
    python train_gbdt.py --train data/train.csv --unlabeled data/unlabeled.csv \
        --test data/test.csv --submission submission_gbdt.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline

from features_v2 import build_vectorizer_v2, stylometric_features
from split import SEED, load_split

LSA_COMPONENTS = 200
PARAM_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31, 63],
    "l2_regularization": [0.0, 1.0],
}


def make_gbdt(**params) -> HistGradientBoostingClassifier:
    # max_iter is an upper bound; early stopping on an internal 10% holdout picks the number of trees.
    return HistGradientBoostingClassifier(
        max_iter=1000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=SEED, **params,
    )


def fit_lsa(texts: list[str]) -> Pipeline:
    return make_pipeline(build_vectorizer_v2(), TruncatedSVD(LSA_COMPONENTS, random_state=SEED)).fit(texts)


def featurize(texts, lsa: Pipeline) -> pd.DataFrame:
    texts = list(texts)
    components = pd.DataFrame(lsa.transform(texts), columns=[f"lsa_{i:03d}" for i in range(LSA_COMPONENTS)])
    return pd.concat([stylometric_features(texts), components], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate gradient-boosted trees")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--unlabeled", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--submission", type=Path, default=Path("submission_gbdt.csv"))
    parser.add_argument("--results", type=Path, default=Path("results_gbdt.json"))
    args = parser.parse_args()

    train_all = pd.read_csv(args.train)
    unlabeled_texts = pd.read_csv(args.unlabeled)["text"].tolist() if args.unlabeled else []
    tr, va = load_split(train_all)
    print(f"train {len(tr):,} rows | val {len(va):,} rows")

    start = time.time()
    lsa = fit_lsa(tr["text"].tolist() + unlabeled_texts)
    F_tr, F_va = featurize(tr["text"], lsa), featurize(va["text"], lsa)
    feature_time = time.time() - start
    stylo_cols = [c for c in F_tr.columns if not c.startswith("lsa_")]
    lsa_cols = [c for c in F_tr.columns if c.startswith("lsa_")]
    print(f"features {len(stylo_cols)} stylometric + {len(lsa_cols)} LSA in {feature_time:.0f}s")

    search = GridSearchCV(
        make_gbdt(), PARAM_GRID, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    )
    search.fit(F_tr, tr["label"])
    for params, s, sd in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"],
                             search.cv_results_["std_test_score"]):
        print(f"  {params} CV AUROC {s:.4f} ± {sd:.4f}")
    best = search.best_params_
    print(f"best {best} (CV AUROC {search.best_score_:.4f})")

    start = time.time()
    model = make_gbdt(**best).fit(F_tr, tr["label"])
    fit_time = time.time() - start
    val_scores = model.predict_proba(F_va)[:, 1]

    # Same hyperparameters, one feature group at a time.
    ablation = {}
    for name, cols in {"stylometric only": stylo_cols, "LSA only": lsa_cols}.items():
        m = make_gbdt(**best).fit(F_tr[cols], tr["label"])
        ablation[name] = round(roc_auc_score(va["label"], m.predict_proba(F_va[cols])[:, 1]), 6)
        print(f"  ablation {name}: val AUROC {ablation[name]:.4f}")

    imp = permutation_importance(model, F_va, va["label"], scoring="roc_auc", n_repeats=5, random_state=SEED)
    top = np.argsort(-imp.importances_mean)[:10]

    results = {
        "model": "Gradient-boosted trees (HistGradientBoostingClassifier)",
        "representation": f"{len(stylo_cols)} stylometric + {len(lsa_cols)}-d LSA of v2 word 1-2 + char 2-5 TF-IDF",
        "params": best,
        "n_trees": int(model.n_iter_),
        "cv_auroc": round(search.best_score_, 6),
        "train_auroc": round(roc_auc_score(tr["label"], model.predict_proba(F_tr)[:, 1]), 6),
        "val_auroc": round(roc_auc_score(va["label"], val_scores), 6),
        "val_auroc_ablation": ablation,
        "top_permutation_importance": {F_va.columns[i]: round(imp.importances_mean[i], 4) for i in top},
        "train_time_s_model_fit": round(fit_time, 1),
        "train_time_s_features_plus_fit": round(feature_time + fit_time, 1),
    }
    print(json.dumps(results, indent=2))
    args.results.write_text(json.dumps(results, indent=2))

    if args.test:
        test = pd.read_csv(args.test)
        lsa = fit_lsa(train_all["text"].tolist() + unlabeled_texts)
        final = make_gbdt(**best).fit(featurize(train_all["text"], lsa), train_all["label"])
        scores = final.predict_proba(featurize(test["text"], lsa))[:, 1]
        pd.DataFrame({"id": test["id"], "label": scores}).to_csv(args.submission, index=False)
        print(f"wrote {args.submission} (refit on all {len(train_all):,} labeled rows)")


if __name__ == "__main__":
    main()
