"""Q4.3: false positives / false negatives from saved validation predictions.

Prints the most confident errors (id, P(AI), length, newline count, snippet),
accuracy at several thresholds, and a rough per-domain false-positive table.
Domain labels are keyword heuristics, not ground truth.

    python src/error_analysis_probs.py --probs results/val_probs_distilroberta-base.csv --n 8
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import ROOT, load_train


def domain(s):
    lines = s.split("\n")
    if len(lines) >= 4 and np.mean([len(l) for l in lines]) < 95 and "\n\n" not in s and s.count(". ") >= 2:
        return "hard-wrapped abstract"
    if any(k in s[:200].lower() for k in ["cup", "tablespoon", "teaspoon", "ounce"]):
        return "recipe"
    if "(born" in s[:200] or " is a " in s[:150] or " was a " in s[:150]:
        return "wiki-style entry"
    return "other"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()

    d = pd.read_csv(a.probs).merge(load_train()[["id", "text"]], on="id")
    d["words"] = d["text"].str.split().str.len()
    d["nl"] = d["text"].str.count("\n")
    d["domain"] = d["text"].map(domain)
    pred = d["p"] >= a.thr
    print(f"errors at {a.thr}: {(pred != d.label).sum()}  "
          f"FP={(pred & (d.label == 0)).sum()}  FN={(~pred & (d.label == 1)).sum()}")
    print("human flagged AI:", round((d[d.label == 0].p >= a.thr).mean(), 3),
          "| AI flagged human:", round((d[d.label == 1].p < a.thr).mean(), 3))
    for th in (0.5, 0.9, 0.99, 0.999):
        print(f"accuracy@{th}: {((d.p >= th) == d.label).mean():.3f}")

    for name, x in [("FALSE POSITIVES (human scored as AI)", d[d.label == 0].nlargest(a.n, "p")),
                    ("FALSE NEGATIVES (AI scored as human)", d[d.label == 1].nsmallest(a.n, "p"))]:
        print("\n=====", name)
        for _, r in x.iterrows():
            print(f"id={r.id} P(AI)={r.p:.3f} words={r.words} newlines={r.nl} domain={r.domain}\n"
                  f"   {r.text[:200]!r}")

    print("\nBy rough domain:")
    for g, x in d.groupby("domain"):
        h = x[x.label == 0]
        print(f"  {g:22s} n={len(x):4d} frac_ai={x.label.mean():.2f} "
              f"human_FP_rate={(h.p >= a.thr).mean():.2f} auroc={roc_auc_score(x.label, x.p):.3f}")
