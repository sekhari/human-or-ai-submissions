"""
Q4.3: pull the most confident false positives and false negatives
from the final model's (B + D ensemble) validation predictions.

Input CSV columns: id, text, label, score  (score = rank-averaged B + D, 0-1)
Threshold 0.5 = middle of the ranking (scores are ranks, not true probabilities).

Run:  python error_analysis.py bd_val_predictions.csv
"""
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

THRESHOLD = 0.5
N = 5

path = sys.argv[1] if len(sys.argv) > 1 else "bd_val_predictions.csv"
df = pd.read_csv(path)
df["words"] = df["text"].str.split().str.len()

print(f"Val AUROC: {roc_auc_score(df['label'], df['score']):.4f}")

# False positive = human (0) predicted AI; false negative = AI (1) predicted human
fp = df[(df["label"] == 0) & (df["score"] >= THRESHOLD)].sort_values("score", ascending=False)
fn = df[(df["label"] == 1) & (df["score"] < THRESHOLD)].sort_values("score")
print(f"Threshold {THRESHOLD}: {len(fp)} false positives, {len(fn)} false negatives")
print(f"Median words: all {df['words'].median()}, FP {fp['words'].median()}, FN {fn['words'].median()}\n")

for name, errs in [("FALSE POSITIVES (human -> AI)", fp), ("FALSE NEGATIVES (AI -> human)", fn)]:
    print(f"===== {name}: most confident {N} =====")
    for _, row in errs.head(N).iterrows():
        print(f"\nID {row['id']} | score={row['score']:.3f} | {row['words']} words")
        print(row["text"][:400], "...")
    print()

pd.concat([fp.head(N).assign(error="FP"), fn.head(N).assign(error="FN")])[
    ["id", "error", "score", "words"]
].to_csv("error_table.csv", index=False)
print("Saved error_table.csv")
