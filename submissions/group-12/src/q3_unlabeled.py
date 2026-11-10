"""Q3: does unlabeled.csv help? Self-training with confidence-based pseudo-labels.

Protocol (no validation leakage): the teacher is fit on the 8,000-row training partition only,
pseudo-labels are assigned to unlabeled.csv, the student is refit on train + pseudo-labeled rows,
and every model is scored once on the same fixed 2,000-row validation set (splits/val_ids.csv).

Run from repo root:  python -m src.q3_unlabeled [--model masked_lr]
Writes results/q3/<model>/summary.md and CSVs.
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .clusters import assign_clusters
from .data import RESULTS_DIR, load_split, load_unlabeled
from .models import MODELS

THRESHOLDS = [0.6, 0.8, 0.9, 0.95]
N_ROUNDS = 2


def fit_score(make_model, X_tr, y_tr, X_val, y_val, X_unl, fiction_val):
    t0 = time.time()
    m = make_model().fit(X_tr, y_tr)
    p_val = m.predict_proba(X_val)[:, 1]
    return {
        "model": m, "p_val": p_val, "p_unl": m.predict_proba(X_unl)[:, 1],
        "val AUROC": roc_auc_score(y_val, p_val),
        "val AUROC (fiction rows)": roc_auc_score(y_val[fiction_val], p_val[fiction_val]),
        "val error@0.5": ((p_val > 0.5) != y_val).mean(),
        "fit seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="masked_lr", choices=list(MODELS))
    args = ap.parse_args()
    out = RESULTS_DIR / "q3" / args.model
    out.mkdir(parents=True, exist_ok=True)
    make_model = MODELS[args.model]

    train, val = load_split()
    unl = load_unlabeled()
    clusters, _, fiction = assign_clusters()
    fiction_val = (val["id"].map(clusters["train"]) == fiction).values
    X_tr, y_tr = train["text"], train["label"].values
    X_val, y_val = val["text"], val["label"].values
    X_unl = unl["text"]

    rows = []
    base = fit_score(make_model, X_tr, y_tr, X_val, y_val, X_unl, fiction_val)
    rows.append({"method": "supervised only (8,000 labeled)", "threshold": None, "round": 0,
                 "pseudo rows added": 0, **{k: v for k, v in base.items() if not k.startswith(("model", "p_"))}})

    # proxy for pseudo-label noise: accuracy of the teacher's confident predictions on validation
    noise = []
    for thr in THRESHOLDS:
        conf = (base["p_val"] >= thr) | (base["p_val"] <= 1 - thr)
        noise.append({"threshold": thr, "val rows kept": int(conf.sum()), "val share kept": conf.mean(),
                      "accuracy of kept rows": ((base["p_val"][conf] > 0.5) == y_val[conf]).mean(),
                      "unlabeled rows kept": int(((base["p_unl"] >= thr) | (base["p_unl"] <= 1 - thr)).sum()),
                      "unlabeled AI share (pseudo)": float((base["p_unl"][(base["p_unl"] >= thr) | (base["p_unl"] <= 1 - thr)] > 0.5).mean())})
    noise = pd.DataFrame(noise)
    noise.to_csv(out / "pseudo_label_noise_proxy.csv", index=False)

    for thr in THRESHOLDS:
        p_unl = base["p_unl"]
        for r in range(1, N_ROUNDS + 1):
            conf = (p_unl >= thr) | (p_unl <= 1 - thr)
            X_aug = pd.concat([X_tr, X_unl[conf]], ignore_index=True)
            y_aug = np.concatenate([y_tr, (p_unl[conf] > 0.5).astype(int)])
            res = fit_score(make_model, X_aug, y_aug, X_val, y_val, X_unl, fiction_val)
            rows.append({"method": f"self-training (pseudo-label if P>={thr} or <={1-thr:.2f})", "threshold": thr,
                         "round": r, "pseudo rows added": int(conf.sum()),
                         **{k: v for k, v in res.items() if not k.startswith(("model", "p_"))}})
            p_unl = res["p_unl"]

    # also: unlabeled text used only to fit the vectorizer vocabulary (feature learning, no pseudo-labels)
    res = pd.DataFrame(rows)
    res["delta vs supervised"] = res["val AUROC"] - res.loc[0, "val AUROC"]
    res.to_csv(out / "q32_results.csv", index=False)

    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Q3 results: {args.model} (fixed split seed 42, val = 2,000 rows)\n\n")
        f.write("## Q3.2 supervised vs self-training\n\n" + res.round(4).to_markdown(index=False) + "\n\n")
        f.write("## Pseudo-label noise proxy (teacher confidence on validation rows)\n\n" + noise.round(4).to_markdown(index=False) + "\n")
    print(open(out / "summary.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
