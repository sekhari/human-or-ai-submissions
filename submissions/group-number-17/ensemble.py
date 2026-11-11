"""
Q5: does averaging models beat the best single model?

Everything trains on the same 8,000 rows and gets scored on the same 2,000 val rows.
The SVMs output margins (not probabilities) and NB's probabilities are all squished
near 0 or 1, so averaging raw scores would let one model take over. Instead I turn each
model's scores into ranks (percentiles) and average those. AUROC only cares about
ordering, so this is fair.

A, B, C are the team's three models. D (char n-grams + SVM) is my follow-up: A and B
turned out almost identical, so I wanted a model that looks at the text differently.

A matches James's logistic_model.py, B matches Anisha's svm_model.py.

Run (from team folder): python nb_model.py   (first)
                        python ensemble.py
Out: ensemble_results.csv, val_predictions.csv
"""
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from common import SEED, load_split

A_C = 10  # James's chosen C (with 20,000 max features, see logistic_model.py)
B_C = 1   # Person 2's SVM C (val 0.9434)

nb = pd.read_csv("nb_results.csv").sort_values("val_auroc", ascending=False).iloc[0]
nb_mf = None if str(nb["max_features"]) == "all" else int(nb["max_features"])
nb_alpha = float(nb["alpha"])

tr, va = load_split()

models = {
    "A": make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=20_000),
                       LogisticRegression(C=A_C, solver="liblinear", max_iter=2000,
                                          random_state=SEED)),
    "B": make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                       LinearSVC(C=B_C)),
    "C": make_pipeline(CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=nb_mf),
                       MultinomialNB(alpha=nb_alpha)),
    "D": make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                       sublinear_tf=True),
                       LinearSVC(C=1)),
}
names = {"A": "word TF-IDF + LogReg", "B": "word TF-IDF + LinearSVC",
         "C": "word counts + NaiveBayes", "D": "char 3-5 TF-IDF + LinearSVC"}

scores, secs, rows = {}, {}, []
for key, model in models.items():
    start = time.time()
    model.fit(tr["text"], tr["label"])
    secs[key] = time.time() - start
    if hasattr(model, "predict_proba"):
        scores[key] = model.predict_proba(va["text"])[:, 1]
    else:
        scores[key] = model.decision_function(va["text"])
    auc = roc_auc_score(va["label"], scores[key])
    rows.append({"config": f"{key}: {names[key]}", "val_auroc": round(auc, 4),
                 "train_sec": round(secs[key], 1)})
    print(f"{key}: {names[key]:<28} val AUROC={auc:.4f}  ({secs[key]:.1f}s)")

ranks = {k: rankdata(s) / len(s) for k, s in scores.items()}

print()
for size in [2, 3, 4]:
    for combo in combinations(models, size):
        auc = roc_auc_score(va["label"], np.mean([ranks[k] for k in combo], axis=0))
        t = sum(secs[k] for k in combo)
        label = "Ensemble " + " + ".join(combo)
        rows.append({"config": label, "val_auroc": round(auc, 4), "train_sec": round(t, 1)})
        print(f"{label:<22} val AUROC={auc:.4f}  ({t:.1f}s)")

# low correlation = models disagree more = more for an average to gain
print("\nSpearman correlation of val scores:")
for a, b in combinations(models, 2):
    print(f"  {a} vs {b}: {spearmanr(scores[a], scores[b])[0]:.3f}")

pd.DataFrame(rows).to_csv("ensemble_results.csv", index=False)
out = va[["id", "label"]].copy()
for k, s in scores.items():
    out[f"{k}_score"] = s
out.to_csv("val_predictions.csv", index=False)