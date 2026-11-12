"""Tune the regularization strength C of the word+char TF-IDF logistic regression.

Small C = strong regularization (simpler model), large C = weak regularization.
Uses only the fixed 2,000-row validation set (random_fold).

    python src/tune.py
"""
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_train
from models import tfidf_logreg

C_VALUES = [0.25, 0.5, 1, 2, 4, 8, 16, 32]


def main():
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    train, splits = load_train(), load_splits()
    tr, va = get_split(train, splits, "random_fold")

    rows = []
    for C in C_VALUES:
        model = tfidf_logreg(C=C)
        model.fit(tr["text"], tr["label"])
        auc = roc_auc_score(va["label"], model.predict_proba(va["text"])[:, 1])
        train_auc = roc_auc_score(tr["label"], model.predict_proba(tr["text"])[:, 1])
        rows.append({"C": C, "train_auroc": round(train_auc, 4), "val_auroc": round(auc, 4)})
        print(f"C={C:<6} train AUROC={train_auc:.4f}  val AUROC={auc:.4f}")

    out = ROOT / "results" / "tune_C.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
