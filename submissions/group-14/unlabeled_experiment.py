"""Does unlabeled.csv help the final GBDT? (report Q3)

Three arms, all scored on the fixed 2,000-row validation set with the
results_gbdt.json hyperparameters:

  A  supervised only    TF-IDF/SVD fit on the 8,000 training texts; unlabeled.csv is never read
                        by the model.
  B  feature learning   TF-IDF/SVD fit on training + unlabeled text (the train_gbdt.py pipeline).
  C  B + pseudo-labels  the arm-B model scores unlabeled.csv; rows with P(AI) >= 0.99 or <= 0.01
                        are added to the training set with that label and the model is refit.

The validation rows are never used to fit the TF-IDF/SVD, to train the teacher, or to pick the
thresholds (fixed in advance). They are used once per arm for the reported AUROC, and to estimate
how often a prediction that confident is wrong (a proxy for pseudo-label noise).

Example:
    python unlabeled_experiment.py --train data/train.csv --unlabeled data/unlabeled.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from split import SEED, load_split
from train_gbdt import featurize, fit_lsa, make_gbdt

THRESH_LO, THRESH_HI = 0.01, 0.99
PARAMS = json.loads(Path(__file__).with_name("results_gbdt.json").read_text())["params"]


def fit(F: pd.DataFrame, y, seed: int):
    return make_gbdt(**PARAMS).set_params(random_state=seed).fit(F, y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised vs. pseudo-labeling with unlabeled.csv")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--unlabeled", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--results", type=Path, default=Path("results_unlabeled.json"))
    args = parser.parse_args()

    tr, va = load_split(pd.read_csv(args.train))
    un = pd.read_csv(args.unlabeled)
    y_tr, y_va = tr["label"].to_numpy(), va["label"].to_numpy()

    lsa_a = fit_lsa(tr["text"].tolist())
    Fa_tr, Fa_va = featurize(tr["text"], lsa_a), featurize(va["text"], lsa_a)
    lsa_b = fit_lsa(tr["text"].tolist() + un["text"].tolist())
    Fb_tr, Fb_va, Fb_un = (featurize(d["text"], lsa_b) for d in (tr, va, un))
    print("features built", flush=True)

    runs = []
    for seed in range(SEED, SEED + args.repeats):
        auc_a = roc_auc_score(y_va, fit(Fa_tr, y_tr, seed).predict_proba(Fa_va)[:, 1])

        teacher = fit(Fb_tr, y_tr, seed)
        p_va = teacher.predict_proba(Fb_va)[:, 1]
        auc_b = roc_auc_score(y_va, p_va)

        p_un = teacher.predict_proba(Fb_un)[:, 1]
        keep = (p_un >= THRESH_HI) | (p_un <= THRESH_LO)
        pseudo = (p_un[keep] >= THRESH_HI).astype(int)
        student = fit(pd.concat([Fb_tr, Fb_un[keep]], ignore_index=True), np.concatenate([y_tr, pseudo]), seed)
        auc_c = roc_auc_score(y_va, student.predict_proba(Fb_va)[:, 1])

        # Same confidence rule on validation, where labels are known: how noisy would the labels be?
        conf = (p_va >= THRESH_HI) | (p_va <= THRESH_LO)
        wrong = (p_va[conf] >= THRESH_HI).astype(int) != y_va[conf]
        blank = (Fb_un["n_blank_lines"] > 0).to_numpy()
        runs.append({
            "seed": seed,
            "val_auroc_supervised_only": auc_a,
            "val_auroc_feature_learning": auc_b,
            "val_auroc_pseudo_label": auc_c,
            "n_pseudo": int(keep.sum()),
            "n_pseudo_ai": int(pseudo.sum()),
            "n_pseudo_human": int((pseudo == 0).sum()),
            "pseudo_ai_with_blank_line": float(blank[keep][pseudo == 1].mean()),
            "unconfident_with_blank_line": float(blank[~keep].mean()) if (~keep).any() else None,
            "val_confident_rows": int(conf.sum()),
            "val_confident_wrong": int(wrong.sum()),
            "val_confident_wrong_ai_called_human": int((wrong & (y_va[conf] == 1)).sum()),
        })
        print(f"seed {seed}: A {auc_a:.4f} | B {auc_b:.4f} | C {auc_c:.4f} | pseudo-labels {keep.sum()} "
              f"({pseudo.sum()} AI) | confident-but-wrong on val {wrong.sum()}/{conf.sum()}", flush=True)

    keys = [k for k in runs[0] if k != "seed" and runs[0][k] is not None]
    summary = {k: {"mean": round(float(np.mean([r[k] for r in runs])), 6),
                   "std": round(float(np.std([r[k] for r in runs])), 6)} for k in keys}
    results = {"params": PARAMS, "thresholds": [THRESH_LO, THRESH_HI], "unlabeled_rows": len(un),
               "unlabeled_with_blank_line": round(float((Fb_un["n_blank_lines"] > 0).mean()), 4),
               "summary": summary, "runs": runs}
    args.results.write_text(json.dumps(results, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
