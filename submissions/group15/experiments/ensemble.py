"""Ensemble of Models A, B, C and D by equal-weight rank averaging.

Run from submissions/group15/:
    python experiments/ensemble.py               # validation table only
    python experiments/ensemble.py --submit      # A+B+C+D -> submission_ensemble_abcd.csv
    python experiments/ensemble.py --submit AC   # A+C     -> submission_ensemble_ac.csv
    python experiments/ensemble.py --submit BC   # B+C     -> submission_ensemble_bc.csv

Uses the saved predictions in preds/ (validation rows in val_ids.txt order,
test rows in test.csv order). Each model's scores are turned into ranks first,
because the models output scores on different scales (B gives SVM margins).
"""

import os
import sys
import itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import load_data, get_split

MODELS = {
    "A": "model_a",   # word TF-IDF + logistic regression
    "B": "model_b",   # char TF-IDF + linear SVM
    "C": "model_c",   # style features + random forest
    "D": "gpt2",      # GPT-2 surprisal features + logistic regression
}


def to_rank(scores):
    return rankdata(scores) / len(scores)


def rank_average(preds, names):
    return np.mean([to_rank(preds[n]) for n in names], axis=0)


if __name__ == "__main__":
    train, unlabeled, test = load_data()
    train_part, val_part = get_split(train)
    y_val = val_part["label"].values

    val = {k: np.load(f"preds/{v}_val.npy") for k, v in MODELS.items()}
    tst = {k: np.load(f"preds/{v}_test.npy") for k, v in MODELS.items()}
    for k in MODELS:
        assert val[k].shape == (len(val_part),), k
        assert tst[k].shape == (len(test),), k

    # every single model and every combination, equal weights
    rows = []
    for n in range(1, len(MODELS) + 1):
        for combo in itertools.combinations(MODELS, n):
            auc = roc_auc_score(y_val, rank_average(val, combo))
            rows.append({"models": "+".join(combo), "n_models": n, "val_auc": round(auc, 4)})
    table = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    print(table.to_string(index=False))

    # how similar the models' rankings are (low = more complementary)
    corr = pd.DataFrame({k: to_rank(v) for k, v in val.items()}).corr(method="spearman")
    print("\nSpearman correlation of validation ranks:")
    print(corr.round(2).to_string())

    abcd_val = rank_average(val, MODELS)
    print(f"\nA+B+C+D: val AUROC {roc_auc_score(y_val, abcd_val):.4f}  "
          f"(C alone {roc_auc_score(y_val, val['C']):.4f})")

    with open("experiments/ensemble_results.txt", "w") as f:
        f.write(table.to_string(index=False))
        f.write("\n\nSpearman correlation of validation ranks:\n")
        f.write(corr.round(2).to_string())

    if "--submit" in sys.argv:
        # which models to combine, e.g. --submit AC ; default is all four
        args = [a for a in sys.argv[1:] if a != "--submit"]
        combo = args[0].upper() if args else "".join(MODELS)
        assert set(combo) <= set(MODELS), f"unknown model letter in {combo}"
        name = combo.lower()

        combo_val = rank_average(val, combo)
        combo_test = rank_average(tst, combo)
        np.save(f"preds/ensemble_{name}_val.npy", combo_val)
        np.save(f"preds/ensemble_{name}_test.npy", combo_test)
        pd.DataFrame({"id": test["id"], "label": combo_test}).to_csv(
            f"submission_ensemble_{name}.csv", index=False)
        print(f"{'+'.join(combo)}: val AUROC {roc_auc_score(y_val, combo_val):.4f}; "
              f"saved preds/ensemble_{name}_*.npy and submission_ensemble_{name}.csv")
