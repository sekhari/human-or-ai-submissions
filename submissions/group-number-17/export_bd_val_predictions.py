"""
Q4: validation predictions for our final model (B + D ensemble).

Trains Model B (word TF-IDF + LinearSVC) and Model D (char 3-5 TF-IDF + LinearSVC)
on the 8,000 training rows, scores the 2,000 val rows, and rank-averages the two,
same as ensemble.py. Also prints train vs val AUROC for Q4.1.

score is in [0, 1], higher = more likely AI. With a 50/50 val set, score > 0.5
means the row is in the top half of AI-looking passages, so 0.5 is the threshold
used for the error_type column (FP = human called AI, FN = AI called human).

Run (from team folder): python export_bd_val_predictions.py
Out: bd_val_predictions.csv (id, text, label, score, error_type, n_words)
     Not committed, it contains competition text.
     bd_length_bins.png (Q4.2 figure). Bins are word-count quartiles of the 8,000
     training rows (161 / 206 / 257 words), same as Model A in logistic_model.py.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from common import load_split

tr, va = load_split()


def rank(x):
    return rankdata(x) / len(x)


models = {
    "B": make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                       LinearSVC(C=1)),
    "D": make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                       sublinear_tf=True),
                       LinearSVC(C=1)),
}

train_ranks, val_ranks = [], []
for name, model in models.items():
    model.fit(tr["text"], tr["label"])
    train_ranks.append(rank(model.decision_function(tr["text"])))
    val_ranks.append(rank(model.decision_function(va["text"])))

train_score = np.mean(train_ranks, axis=0)
val_score = np.mean(val_ranks, axis=0)
print(f"B + D train AUROC: {roc_auc_score(tr['label'], train_score):.4f}")
print(f"B + D val AUROC:   {roc_auc_score(va['label'], val_score):.4f}")

out = va[["id", "text", "label"]].copy()
out["score"] = val_score
pred = (val_score > 0.5).astype(int)
out["error_type"] = np.select([(pred == 1) & (out["label"] == 0),
                               (pred == 0) & (out["label"] == 1)], ["FP", "FN"], "correct")
out["n_words"] = out["text"].str.split().str.len()
out.to_csv("bd_val_predictions.csv", index=False)

print(out["error_type"].value_counts().to_string())
print("Saved bd_val_predictions.csv")

# Q4.2: AUROC by length bin. Thresholds come from training rows only.
cuts = np.quantile(tr["text"].str.split().str.len(), [0.25, 0.5, 0.75])
edges = [-1, *cuts, np.inf]
names = [f"<= {cuts[0]:.0f}", f"{cuts[0]+1:.0f}-{cuts[1]:.0f}",
         f"{cuts[1]+1:.0f}-{cuts[2]:.0f}", f">= {cuts[2]+1:.0f}"]
aucs, counts = [], []
print("\nAUROC by length bin (words):")
for lo, hi, name in zip(edges[:-1], edges[1:], names):
    part = out[(out["n_words"] > lo) & (out["n_words"] <= hi)]
    auc = roc_auc_score(part["label"], part["score"])
    err = (part["error_type"] != "correct").mean()
    aucs.append(auc)
    counts.append(len(part))
    print(f"  {name:<10} n={len(part):<4} AUROC={auc:.4f}  error rate={err:.1%}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3.2))
bars = ax.bar(names, aucs, color="#236B82")
for bar, auc, n in zip(bars, aucs, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.003, f"{auc:.3f}\n(n={n})",
            ha="center", fontsize=8)
ax.set_ylim(0.85, 1.0)
ax.set_xlabel("Passage length (words)")
ax.set_ylabel("Validation AUROC")
ax.set_title("B + D ensemble: validation AUROC by length bin", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("bd_length_bins.png", dpi=200)
print("Saved bd_length_bins.png")
