"""Rank-average ensemble of any set of models (grid search on the fixed val set).

Model names map to files:
  tfidf          -> val: refit word+char TF-IDF LogReg on the train fold; test: submission.csv
  anything else  -> val: results/val_probs_<name>.csv ; test: sub_<name>.csv

    python src/ensemble_n.py --mode val  --models distilroberta-base roberta-base ppl2_gbm tfidf
    python src/ensemble_n.py --mode test --models distilroberta-base roberta-base ppl2_gbm tfidf --w 0.3 0.3 0.3 0.1
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_train


def rank(x):
    return rankdata(x) / len(x)


def weight_grid(k, step=0.1):
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=k - 1):
        if sum(c) <= n:
            yield np.array(list(c) + [n - sum(c)]) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["val", "test"], default="val")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--w", type=float, nargs="+")
    p.add_argument("--out", default="sub_ensemble_final.csv")
    args = p.parse_args()

    if args.mode == "val":
        df = None
        for name in args.models:
            if name == "tfidf":
                from models import tfidf_logreg
                train = load_train()
                tr, va = get_split(train, load_splits(), "random_fold")
                lr = tfidf_logreg().fit(tr["text"], tr["label"])
                d = pd.DataFrame({"id": va["id"].values, "label": va["label"].values,
                                  name: lr.predict_proba(va["text"])[:, 1]})
            else:
                d = pd.read_csv(ROOT / "results" / f"val_probs_{name}.csv").rename(columns={"p": name})
            df = d if df is None else df.merge(d.drop(columns="label"), on="id")
        R = np.column_stack([rank(df[m]) for m in args.models])
        y = df["label"].values

        for i, m in enumerate(args.models):
            print(f"{m:<22} alone: {roc_auc_score(y, R[:, i]):.4f}")
        rows = [{**{f"w_{m}": round(w_, 2) for m, w_ in zip(args.models, w)},
                 "val_auroc": round(roc_auc_score(y, R @ w), 4)}
                for w in weight_grid(len(args.models))]
        res = pd.DataFrame(rows).sort_values("val_auroc", ascending=False)
        eq = roc_auc_score(y, R.mean(1))
        print(f"{'equal weights':<22}      : {eq:.4f}\n\nTop 10 weightings:")
        print(res.head(10).to_string(index=False))
        res.to_csv(ROOT / "results" / "ensemble_n_val.csv", index=False)
        print("Saved results/ensemble_n_val.csv")

    else:
        assert args.w and len(args.w) == len(args.models), "give one weight per model"
        m = None
        for name in args.models:
            f = "submission.csv" if name == "tfidf" else f"sub_{name}.csv"
            d = pd.read_csv(ROOT / f).rename(columns={"label": name})
            m = d if m is None else m.merge(d, on="id")
        assert len(m) == 3000, "submission files do not cover the same 3,000 ids"
        m["label"] = sum(w * rank(m[n]) for n, w in zip(args.models, args.w))
        m[["id", "label"]].to_csv(ROOT / args.out, index=False)
        print(f"Wrote {args.out}: " + ", ".join(f"{n}={w}" for n, w in zip(args.models, args.w)))


if __name__ == "__main__":
    main()
