"""Rank-average ensemble of the transformer and the word+char TF-IDF LogReg.

Rank averaging converts each model's probabilities to ranks before averaging,
so models with different probability scales are weighted fairly. AUROC only
depends on ranking, so this is a natural way to combine models for this metric.

  --mode val   ensemble validation predictions and report AUROC for several weights
  --mode test  ensemble two submission files into a new submission

    python src/ensemble.py --mode val
    python src/ensemble.py --mode test --a sub_distilroberta-base.csv --b submission.csv --w 0.5
"""
import argparse

import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_train
from models import tfidf_logreg


def rank(x):
    return rankdata(x) / len(x)


def main():
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["val", "test"], default="val")
    p.add_argument("--transformer", default="distilroberta-base")
    p.add_argument("--a", help="test mode: first submission file")
    p.add_argument("--b", help="test mode: second submission file")
    p.add_argument("--w", type=float, default=0.5, help="weight of file a")
    p.add_argument("--out", default="sub_ensemble.csv")
    args = p.parse_args()

    if args.mode == "val":
        predictions = ROOT / "results" / f"val_probs_{args.transformer}.csv"
        if not predictions.exists():
            p.error("Run finetune_transformer.py --mode val with the corresponding model settings first.")
        trf = pd.read_csv(predictions)
        train, splits = load_train(), load_splits()
        tr, va = get_split(train, splits, "random_fold")
        lr = tfidf_logreg().fit(tr["text"], tr["label"])
        va = va.assign(p_lr=lr.predict_proba(va["text"])[:, 1]).merge(trf[["id", "p"]], on="id")

        rows = []
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            score = w * rank(va["p"]) + (1 - w) * rank(va["p_lr"])
            auc = roc_auc_score(va["label"], score)
            rows.append({"weight_transformer": w, "val_auroc": round(auc, 4)})
            print(f"transformer weight {w:.2f}  ->  val AUROC = {auc:.4f}")
        pd.DataFrame(rows).to_csv(ROOT / "results" / "ensemble_val.csv", index=False)
        print("Saved results/ensemble_val.csv  (weight 1.0 = transformer alone, 0.0 = TF-IDF alone)")

    else:
        a, b = pd.read_csv(ROOT / args.a), pd.read_csv(ROOT / args.b)
        m = a.merge(b, on="id", suffixes=("_a", "_b"))
        assert len(m) == 3000, "submission files do not cover the same 3,000 ids"
        m["label"] = args.w * rank(m["label_a"]) + (1 - args.w) * rank(m["label_b"])
        m[["id", "label"]].to_csv(ROOT / args.out, index=False)
        print(f"Wrote {args.out}: {args.w:.2f} x {args.a} + {1 - args.w:.2f} x {args.b}")


if __name__ == "__main__":
    main()
