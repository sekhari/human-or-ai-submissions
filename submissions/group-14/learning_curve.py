"""Effect of labeled training-set size on the final GBDT (report Q2.1).

The representation (37 stylometric + 200-d LSA, fit without labels on the 8,000
training texts + unlabeled.csv) and the hyperparameters from results_gbdt.json
are held fixed; only the number of labeled rows used to fit the trees changes.
Each size is a stratified sample from the 8,000-row training partition, nested
across sizes, repeated for several seeds. Validation is the fixed 2,000 rows.

Example:
    python learning_curve.py --train data/train.csv --unlabeled data/unlabeled.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from split import SEED, load_split
from train_gbdt import featurize, fit_lsa, make_gbdt

SIZES = [500, 1000, 2000, 5000, 8000]
BEST_PARAMS = {"learning_rate": 0.1, "max_leaf_nodes": 15, "l2_regularization": 0.0}


def nested_stratified_order(labels: pd.Series, seed: int) -> np.ndarray:
    """Row order whose every prefix is (as nearly as possible) 50/50 by class."""
    rng = np.random.default_rng(seed)
    per_class = [rng.permutation(np.flatnonzero(labels.to_numpy() == c)) for c in (0, 1)]
    return np.column_stack(per_class).ravel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Learning curve for the GBDT")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--unlabeled", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--results", type=Path, default=Path("results_learning_curve.json"))
    args = parser.parse_args()

    tr, va = load_split(pd.read_csv(args.train))
    unlabeled_texts = pd.read_csv(args.unlabeled)["text"].tolist() if args.unlabeled else []
    assert tr["label"].value_counts().min() >= max(SIZES) // 2

    start = time.time()
    lsa = fit_lsa(tr["text"].tolist() + unlabeled_texts)
    F_tr, F_va = featurize(tr["text"], lsa), featurize(va["text"], lsa)
    print(f"features built once in {time.time() - start:.0f}s")

    runs = {n: [] for n in SIZES}
    for r in range(args.repeats):
        order = nested_stratified_order(tr["label"], SEED + r)
        for n in SIZES:
            idx = order[:n]
            start = time.time()
            model = make_gbdt(**BEST_PARAMS).fit(F_tr.iloc[idx], tr["label"].iloc[idx])
            runs[n].append({
                "train_auroc": roc_auc_score(tr["label"].iloc[idx], model.predict_proba(F_tr.iloc[idx])[:, 1]),
                "val_auroc": roc_auc_score(va["label"], model.predict_proba(F_va)[:, 1]),
                "n_trees": int(model.n_iter_),
                "fit_s": time.time() - start,
            })
            print(f"repeat {r} n={n:>5}: train {runs[n][-1]['train_auroc']:.4f} "
                  f"val {runs[n][-1]['val_auroc']:.4f} trees {runs[n][-1]['n_trees']}", flush=True)

    summary = {}
    for n, rs in runs.items():
        summary[n] = {k: {"mean": round(float(np.mean([x[k] for x in rs])), 6),
                          "std": round(float(np.std([x[k] for x in rs])), 6)}
                      for k in rs[0]}
    results = {"model": "GBDT (results_gbdt.json params)", "params": BEST_PARAMS,
               "repeats": args.repeats, "validation_rows": len(va), "by_size": summary}
    args.results.write_text(json.dumps(results, indent=2))

    print(f"\n{'rows':>6} {'train AUROC':>18} {'val AUROC':>18} {'trees':>8}")
    for n, s in summary.items():
        print(f"{n:>6} {s['train_auroc']['mean']:.4f} ± {s['train_auroc']['std']:.4f} "
              f"{s['val_auroc']['mean']:.4f} ± {s['val_auroc']['std']:.4f} {s['n_trees']['mean']:>8.0f}")


if __name__ == "__main__":
    main()
