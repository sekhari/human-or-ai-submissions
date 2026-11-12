"""G2 + G3: a classifier on the GPT-2 surprise features, and its combination with Model A.

Needs data/gpt2_features.npz (from experiments/gpt2_features.py) and preds/model_a_*.npy
(from experiments/model_a.py --submit). Run from submissions/group15/:
    python experiments/model_gpt2.py            # validation results only
    python experiments/model_gpt2.py --submit   # also retrain on all 10k and write submissions
"""

import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import load_data, get_split, SEED

FEATURE_FILE = "data/gpt2_features.npz"
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_feats(df, split):
    """Look up the 5 GPT-2 features for each row of df, in df's order."""
    z = np.load(FEATURE_FILE)
    lookup = dict(zip(z[f"{split}_ids"], z[f"{split}_feats"]))
    return np.stack([lookup[i] for i in df["id"]])


def build_model():
    # the 5 features have very different scales (e.g. max ~15, frac_low ~0.3),
    # so standardize them before logistic regression
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=SEED))


def to_rank(scores):
    """Turn scores into ranks between 0 and 1, so two models are on the same scale."""
    return rankdata(scores) / len(scores)


if __name__ == "__main__":
    train, unlabeled, test = load_data()
    train_part, val_part = get_split(train)

    # ---------- G2: GPT-2 features alone ----------
    X_train = load_feats(train_part, "train")
    X_val = load_feats(val_part, "train")
    model = build_model().fit(X_train, train_part["label"])
    train_auc = roc_auc_score(train_part["label"], model.predict_proba(X_train)[:, 1])
    gpt2_val = model.predict_proba(X_val)[:, 1]
    gpt2_val_auc = roc_auc_score(val_part["label"], gpt2_val)
    print(f"GPT-2 features only: train AUROC {train_auc:.4f} | val AUROC {gpt2_val_auc:.4f}")
    os.makedirs("preds", exist_ok=True)
    np.save("preds/gpt2_val.npy", gpt2_val)

    # ---------- G3: combine with Model A (weighted rank average) ----------
    a_val = np.load("preds/model_a_val.npy")
    print("\nweight on GPT-2 | val AUROC")
    results = []
    for w in WEIGHTS:
        combo = (1 - w) * to_rank(a_val) + w * to_rank(gpt2_val)
        auc = roc_auc_score(val_part["label"], combo)
        results.append((w, auc))
        print(f"      {w:.1f}        | {auc:.4f}")
    best_w, best_auc = max(results, key=lambda r: r[1])
    print(f"\nbest weight on GPT-2 = {best_w:.1f} (val AUROC {best_auc:.4f}); "
          f"Model A alone = {results[0][1]:.4f}")
    with open("experiments/model_gpt2_results.txt", "w") as f:
        f.write(f"GPT-2 features only: train {train_auc:.4f}, val {gpt2_val_auc:.4f}\n")
        f.write("weight_on_gpt2  val_auc\n")
        for w, auc in results:
            f.write(f"{w:.1f}  {auc:.4f}\n")

    # ---------- test predictions + Kaggle files ----------
    if "--submit" in sys.argv:
        final = build_model().fit(load_feats(train, "train"), train["label"])
        gpt2_test = final.predict_proba(load_feats(test, "test"))[:, 1]
        np.save("preds/gpt2_test.npy", gpt2_test)
        pd.DataFrame({"id": test["id"], "label": gpt2_test}).to_csv(
            "submission_gpt2_only.csv", index=False)

        a_test = np.load("preds/model_a_test.npy")
        combo_test = (1 - best_w) * to_rank(a_test) + best_w * to_rank(gpt2_test)
        pd.DataFrame({"id": test["id"], "label": combo_test}).to_csv(
            "submission_model_a_gpt2.csv", index=False)
        print("saved preds/gpt2_test.npy, submission_gpt2_only.csv and "
              f"submission_model_a_gpt2.csv (weight on GPT-2 = {best_w:.1f})")
