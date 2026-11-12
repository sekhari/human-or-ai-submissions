"""Report table Q1.2: train AUROC, validation AUROC and training time for every model.

All models are trained on the same 8,000-row training fold and scored on the
fixed 2,000-row validation set.

    python src/model_table.py
"""
import time

import pandas as pd
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_train
from models import MODELS

REPRESENTATION = {
    "word_logreg": "Word 1-2gram TF-IDF",
    "char_logreg": "Char 3-5gram TF-IDF",
    "tfidf_logreg": "Word 1-2gram + char 3-5gram TF-IDF",
    "naive_bayes": "Word 1-2gram counts",
    "linear_svm": "Word 1-2gram + char 3-5gram TF-IDF",
    "style_gbm": "26 stylometric features",
}


def main():
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    train, splits = load_train(), load_splits()
    tr, va = get_split(train, splits, "random_fold")

    rows = []
    for name, make in MODELS.items():
        start = time.time()
        model = make().fit(tr["text"], tr["label"])
        secs = time.time() - start
        tr_auc = roc_auc_score(tr["label"], model.predict_proba(tr["text"])[:, 1])
        va_auc = roc_auc_score(va["label"], model.predict_proba(va["text"])[:, 1])
        rows.append({"model": name, "representation": REPRESENTATION.get(name, ""),
                     "train_auroc": round(tr_auc, 4), "val_auroc": round(va_auc, 4),
                     "train_time_s": round(secs, 1)})
        print(f"{name:<13} train={tr_auc:.4f}  val={va_auc:.4f}  time={secs:5.1f}s")

    df = pd.DataFrame(rows).sort_values("val_auroc", ascending=False)
    df.to_csv(ROOT / "results" / "model_table.csv", index=False)
    print("\n", df.to_string(index=False))
    print("\nSaved results/model_table.csv")


if __name__ == "__main__":
    main()
