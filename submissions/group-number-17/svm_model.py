"""
Model B: word TF-IDF + linear SVM.
Step 2: baseline with one C value.

Run (from team folder): python svm_model.py
"""
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# Load data and the shared validation split
train = pd.read_csv("data/train.csv")
val_ids = set(pd.read_csv("val_ids.csv")["id"])
is_val = train["id"].isin(val_ids)
tr, va = train[~is_val], train[is_val]
print(f"Train rows: {len(tr)}, Val rows: {len(va)}")

# Word TF-IDF (single words + word pairs) -> linear SVM
results = []
for C in [0.01, 0.1, 1, 10]:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LinearSVC(C=C),
    )
    start = time.time()
    model.fit(tr["text"], tr["label"])
    secs = time.time() - start

    train_auc = roc_auc_score(tr["label"], model.decision_function(tr["text"]))
    val_auc = roc_auc_score(va["label"], model.decision_function(va["text"]))
    results.append({"C": C, "train_auroc": train_auc, "val_auroc": val_auc, "train_sec": secs})
    print(f"C={C:<5} train AUROC={train_auc:.4f}  val AUROC={val_auc:.4f}  ({secs:.1f}s)")

pd.DataFrame(results).to_csv("svm_results.csv", index=False)