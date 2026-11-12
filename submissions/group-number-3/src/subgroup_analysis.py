"""Q4.2: validation AUROC by text-length bin and by blank-line presence.

Works on any model's saved validation predictions (columns id,label,p), e.g.
results/val_probs_distilroberta-base.csv written by finetune_transformer.py.

Length bins are in words (whitespace-split), fixed width:
  fine (5 bins):  75-149, 150-199, 200-249, 250-299, 300-350
  coarse (3):     Short 75-149, Medium 150-249, Long 250-350
Blank line = text contains "\\n\\n" (0% of human, ~66% of AI in train.csv), so
that group is single-class and its AUROC is undefined; the informative
subgroup is "no blank line".
95% CIs: 500 bootstrap resamples within each group.

    python src/subgroup_analysis.py --probs results/val_probs_distilroberta-base.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import ROOT, load_train

rng = np.random.default_rng(42)


def auroc_ci(y, p, B=500):
    if len(set(y)) < 2:
        return np.nan, np.nan, np.nan
    a, bs = roc_auc_score(y, p), []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            bs.append(roc_auc_score(y[i], p[i]))
    return (a, *np.percentile(bs, [2.5, 97.5]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", default="subgroup_auroc.csv")
    args = ap.parse_args()

    v = pd.read_csv(args.probs).merge(load_train()[["id", "text"]], on="id")
    v["words"] = v["text"].str.split().str.len()
    v["fine"] = pd.cut(v["words"], [0, 149, 199, 249, 299, 10**5],
                       labels=["75-149", "150-199", "200-249", "250-299", "300-350"])
    v["coarse"] = pd.cut(v["words"], [0, 149, 249, 10**5],
                         labels=["Short (75-149)", "Medium (150-249)", "Long (250-350)"])
    v["blank"] = np.where(v["text"].str.contains("\n\n"), "has blank line", "no blank line")

    rows = []
    for col in ["fine", "coarse", "blank"]:
        for g, d in v.groupby(col, observed=True):
            a, lo, hi = auroc_ci(d["label"].values, d["p"].values)
            rows.append(dict(grouping=col, group=g, n=len(d), frac_ai=round(d["label"].mean(), 3),
                             auroc=round(a, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    rows.append(dict(grouping="all", group="all", n=len(v), frac_ai=round(v["label"].mean(), 3),
                     auroc=round(roc_auc_score(v["label"], v["p"]), 4), ci_lo=np.nan, ci_hi=np.nan))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / args.out, index=False)
