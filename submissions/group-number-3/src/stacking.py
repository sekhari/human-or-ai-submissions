"""Stack L2 logistic regression + linear SVM with sklearn's StackingClassifier.

Both base models see the same word 1-2gram + char 3-5gram TF-IDF matrix. The
meta-learner (logistic regression) is trained on out-of-fold base predictions
produced by `cv`-fold cross-validation, then the base models are refit on the
full training fold. TF-IDF is fit once on the training fold (unsupervised, so
the validation fold never influences it).

Reports for each model on both shared splits:
  train_auroc : fitted model scored on its own training rows (saturates near 1)
  cv_auroc    : 5-fold out-of-fold AUROC on the training rows (an honest
                "training-side" score; the stack is refit inside each fold)
  val_auroc   : fixed validation fold

    python src/stacking.py
"""
import time

import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from data import ROOT, get_split, load_splits, load_train

SEED = 42
LR_C, SVM_C = 4.0, 0.3  # LR: baseline C (flat 4-32); SVM: best C in svm_ngram sweep


def features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=300_000, sublinear_tf=True)),
    ])


def lr():
    return LogisticRegression(penalty="l2", C=LR_C, max_iter=2000)


def svm():
    return LinearSVC(C=SVM_C, random_state=SEED)


def stack(cv):
    return StackingClassifier(
        estimators=[("lr", lr()), ("svm", svm())],
        final_estimator=LogisticRegression(penalty="l2", C=1.0, max_iter=1000),
        cv=StratifiedKFold(cv, shuffle=True, random_state=SEED),
        stack_method="auto",  # predict_proba for LR, decision_function for LinearSVC
        n_jobs=2)


def score(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)


if __name__ == "__main__":
    train, splits = load_train(), load_splits()
    rows = []
    for fold in ["random_fold", "ood_fold"]:
        tr, va = get_split(train, splits, fold)
        feats = features()
        Xtr, Xva = feats.fit_transform(tr["text"]), feats.transform(va["text"])
        ytr, yva = tr["label"].values, va["label"].values
        models = {"lr_l2": lr(), "linear_svm": svm(), "stack_cv5": stack(5), "stack_cv10": stack(10)}
        for name, m in models.items():
            t0 = time.time()
            cv_auc = float("nan")
            if True:  # nested: 5-fold OOF over the training rows (stack refit inside each fold)
                method = "decision_function" if name == "linear_svm" else "predict_proba"
                oof = cross_val_predict(m, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                        method=method)
                cv_auc = roc_auc_score(ytr, oof[:, 1] if oof.ndim == 2 else oof)
            m.fit(Xtr, ytr)
            row = dict(fold=fold, model=name,
                       train_auroc=round(roc_auc_score(ytr, score(m, Xtr)), 4),
                       cv_auroc=round(cv_auc, 4),
                       val_auroc=round(roc_auc_score(yva, score(m, Xva)), 4),
                       seconds=round(time.time() - t0))
            row["gap"] = round(row["train_auroc"] - row["val_auroc"], 4)
            rows.append(row)
            print(row, flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "stacking.csv", index=False)
