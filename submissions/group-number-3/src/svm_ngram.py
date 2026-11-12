"""Linear SVM on word + character n-grams, tuned for a smoother boundary.

Motivation: the stronger models (TF-IDF LogReg, DistilRoBERTa) score formal,
encyclopedic human writing (Wikipedia-style entries, abstracts) as AI. A lower
SVM penalty C shrinks the weights on individual n-grams, so no single formal
phrase ("is an American", "such as", "overall") can dominate the decision.

Self-contained: does not import models.py.

    python src/svm_ngram.py --mode sweep              # C sweep on both folds
    python src/svm_ngram.py --mode predict --C 0.1    # writes sub_svm_ngram.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data import ROOT, get_split, load_splits, load_test, load_train

SEED = 42
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]


def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5,
                                 max_features=200_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=300_000, sublinear_tf=True)),
    ])


def make_pipeline(C=0.1):
    """Pipeline whose classifier step is named `svm`, so `svm__C` can be tuned."""
    return Pipeline([("features", make_features()),
                     ("svm", LinearSVC(C=C, loss="squared_hinge", random_state=SEED))])


def formal_subset(texts):
    """Rough flag for encyclopedic/formal openings (the main false-positive source)."""
    head = texts.str[:200]
    return head.str.contains(r"\(born|\bis an? |\bwas an? ", regex=True)


def sweep():
    train, splits = load_train(), load_splits()
    rows = []
    for fold in ["random_fold", "ood_fold"]:
        tr, va = get_split(train, splits, fold)
        feats = make_features()
        Xtr, Xva = feats.fit_transform(tr["text"]), feats.transform(va["text"])
        formal = formal_subset(va["text"]).values
        for C in C_GRID:
            m = LinearSVC(C=C, random_state=SEED).fit(Xtr, tr["label"])
            tr_auc = roc_auc_score(tr["label"], m.decision_function(Xtr))
            s = m.decision_function(Xva); y = va["label"].values
            # human false-positive rate at the threshold that gets 90% of AI right
            thr = np.percentile(s[y == 1], 10)
            fpr_all = (s[y == 0] >= thr).mean()
            fpr_formal = (s[(y == 0) & formal] >= thr).mean()
            rows.append(dict(fold=fold, C=C, train_auroc=round(tr_auc, 4),
                             val_auroc=round(roc_auc_score(y, s), 4),
                             val_auroc_formal=round(roc_auc_score(y[formal], s[formal]), 4),
                             human_fpr_at_90tpr=round(fpr_all, 3),
                             human_fpr_formal=round(fpr_formal, 3),
                             n_formal=int(formal.sum())))
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "svm_ngram_sweep.csv", index=False)


def predict(C):
    train, test = load_train(), load_test()
    model = make_pipeline(C).fit(train["text"], train["label"])
    s = model.decision_function(test["text"])
    p = 1 / (1 + np.exp(-2 * s))  # monotone squash to (0,1); AUROC unaffected
    sub = pd.DataFrame({"id": test["id"], "label": p})
    assert len(sub) == 3000 and sub["id"].is_unique and sub["label"].between(0, 1).all()
    sub.to_csv(ROOT / "sub_svm_ngram.csv", index=False)
    print("Wrote sub_svm_ngram.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "predict"], default="sweep")
    ap.add_argument("--C", type=float, default=0.1)
    a = ap.parse_args()
    sweep() if a.mode == "sweep" else predict(a.C)
