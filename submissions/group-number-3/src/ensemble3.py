"""Rank-average up to three models: DistilRoBERTa, perplexity GBM, TF-IDF LogReg.

  --mode val   grid over weights on the fixed 2,000-row validation set
  --mode test  combine three submission files with chosen weights

    python src/ensemble3.py --mode val
    python src/ensemble3.py --mode test --w 0.6 0.2 0.2
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_train
from models import tfidf_logreg

NAMES = ["transformer", "ppl_gbm", "tfidf"]
FILES = ["sub_distilroberta-base.csv", "sub_ppl_gbm.csv", "submission.csv"]


def rank(x):
    return rankdata(x) / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["val", "test"], default="val")
    p.add_argument("--w", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                   help="weights for transformer, ppl_gbm, tfidf")
    p.add_argument("--out", default="sub_ensemble3.csv")
    args = p.parse_args()

    if args.mode == "val":
        trf = pd.read_csv(ROOT / "results" / "val_probs_distilroberta-base.csv")[["id", "label", "p"]]
        ppl = pd.read_csv(ROOT / "results" / "val_probs_ppl_gbm.csv")[["id", "p"]]
        train, splits = load_train(), load_splits()
        tr, va = get_split(train, splits, "random_fold")
        lr = tfidf_logreg().fit(tr["text"], tr["label"])
        tf = pd.DataFrame({"id": va["id"], "p": lr.predict_proba(va["text"])[:, 1]})
        df = trf.merge(ppl, on="id", suffixes=("", "_ppl")).merge(tf, on="id", suffixes=("", "_tf"))
        R = np.column_stack([rank(df["p"]), rank(df["p_ppl"]), rank(df["p_tf"])])

        rows = []
        grid = np.round(np.arange(0, 1.01, 0.1), 1)
        for w1, w2 in itertools.product(grid, grid):
            w3 = round(1 - w1 - w2, 1)
            if w3 < 0:
                continue
            auc = roc_auc_score(df["label"], R @ np.array([w1, w2, w3]))
            rows.append({"w_transformer": w1, "w_ppl": w2, "w_tfidf": w3, "val_auroc": round(auc, 4)})
        res = pd.DataFrame(rows).sort_values("val_auroc", ascending=False)
        res.to_csv(ROOT / "results" / "ensemble3_val.csv", index=False)
        print("Top 10 weight combinations:\n")
        print(res.head(10).to_string(index=False))
        for name, w in zip(NAMES, np.eye(3)):
            print(f"{name:<12} alone: {roc_auc_score(df['label'], R @ w):.4f}")
        print("Saved results/ensemble3_val.csv")

    else:
        subs = [pd.read_csv(ROOT / f) for f in FILES]
        m = subs[0][["id"]]
        for i, s in enumerate(subs):
            m = m.merge(s.rename(columns={"label": f"p{i}"}), on="id")
        assert len(m) == 3000
        m["label"] = sum(w * rank(m[f"p{i}"]) for i, w in enumerate(args.w))
        m[["id", "label"]].to_csv(ROOT / args.out, index=False)
        print(f"Wrote {args.out} with weights {dict(zip(NAMES, args.w))}")


if __name__ == "__main__":
    main()
