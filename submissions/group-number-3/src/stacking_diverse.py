"""Stacking with base models that see DIFFERENT views of the text.

  tfidf_lr   : L2 logistic regression on word 1-2gram + char 3-5gram TF-IDF
  tfidf_svm  : linear SVM on the same TF-IDF matrix
  ppl2_gbm   : gradient boosting on the cached GPT-2 / GPT-2-medium perplexity
               features (results/ppl2_features.csv; newlines already collapsed,
               so no formatting shortcut). Run `python src/perplexity2.py` once
               to create that cache.

Meta-learner: L2 logistic regression on `cv`-fold out-of-fold base predictions
(StackingClassifier). Reports train / out-of-fold-train / validation AUROC on
both shared splits for each base model and for stacks with cv=5 and cv=10.

    python src/stacking_diverse.py
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data import ROOT, get_split, load_splits, load_train

SEED = 42
CACHE = ROOT / "results" / "ppl2_features.csv"


class Cols(BaseEstimator, TransformerMixin):
    """Select columns [start, stop) of a combined matrix; optionally densify."""
    def __init__(self, start=0, stop=None, dense=False):
        self.start, self.stop, self.dense = start, stop, dense

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Z = X[:, self.start:self.stop]
        return Z.toarray() if self.dense and sparse.issparse(Z) else Z


def tfidf():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=300_000, sublinear_tf=True)),
    ])


def bases(n_text):
    return {
        "tfidf_lr": Pipeline([("c", Cols(0, n_text)), ("m", LogisticRegression(C=4.0, max_iter=2000))]),
        "tfidf_svm": Pipeline([("c", Cols(0, n_text)), ("m", LinearSVC(C=0.3, random_state=SEED))]),
        "ppl2_gbm": Pipeline([("c", Cols(n_text, None, dense=True)),
                              ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                   random_state=SEED))]),
    }


def stack(n_text, cv):
    return StackingClassifier(
        estimators=list(bases(n_text).items()),
        final_estimator=LogisticRegression(penalty="l2", C=1.0, max_iter=1000),
        cv=StratifiedKFold(cv, shuffle=True, random_state=SEED), n_jobs=1)


def score(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)


if __name__ == "__main__":
    ppl = pd.read_csv(CACHE)
    pcols = [c for c in ppl.columns if c not in ("id", "split")]
    ppl = ppl[ppl.split == "train"].set_index("id")[pcols]

    train, splits = load_train(), load_splits()
    rows = []
    folds = sys.argv[1:] or ["random_fold", "ood_fold"]  # e.g. `ood_fold` to run one split
    for fold in folds:
        tr, va = get_split(train, splits, fold)
        f = tfidf()
        Ttr, Tva = f.fit_transform(tr["text"]), f.transform(va["text"])
        n_text = Ttr.shape[1]
        Xtr = sparse.hstack([Ttr, sparse.csr_matrix(ppl.loc[tr["id"]].values)]).tocsr()
        Xva = sparse.hstack([Tva, sparse.csr_matrix(ppl.loc[va["id"]].values)]).tocsr()
        ytr, yva = tr["label"].values, va["label"].values

        models = {**bases(n_text), "stack_cv5": stack(n_text, 5), "stack_cv10": stack(n_text, 10)}
        for name, m in models.items():
            t0 = time.time()
            cv_auc = float("nan")
            if name != "stack_cv10":  # nested OOF; cv10 stack skipped (same idea, 2x the cost)
                method = "decision_function" if name == "tfidf_svm" else "predict_proba"
                oof = cross_val_predict(m, Xtr, ytr, method=method,
                                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
                cv_auc = roc_auc_score(ytr, oof[:, 1] if oof.ndim == 2 else oof)
            m.fit(Xtr, ytr)
            row = dict(fold=fold, model=name,
                       train_auroc=round(roc_auc_score(ytr, score(m, Xtr)), 4),
                       oof_train_auroc=round(cv_auc, 4),
                       val_auroc=round(roc_auc_score(yva, score(m, Xva)), 4),
                       seconds=round(time.time() - t0))
            row["gap"] = round(row["train_auroc"] - row["val_auroc"], 4)
            rows.append(row)
            print(row, flush=True)
            (ROOT / "results").mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(ROOT / "results" / f"stacking_diverse_{'_'.join(folds)}.csv", index=False)
