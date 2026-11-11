"""Q3: does unlabeled.csv help?

Methods (all compared on the same fixed validation set, same model and hyperparameters):
  supervised      fit on the 8,000-row training partition only
  tfidf_unlab     same, but the TF-IDF vocabulary/IDF is fit on train + unlabeled texts
                  (unsupervised feature learning; no labels involved)
  self_train      confidence-based pseudo-labelling: fit on train, label the unlabeled
                  rows whose P(AI) is >= hi or <= lo, refit on train + pseudo-labelled
                  rows; 2 rounds. Thresholds are fixed in advance (0.9/0.1), not tuned on val.

Leakage control: validation texts are never seen by the vectoriser or the pseudo-labeller,
and no choice is made by looking at validation labels.

Pseudo-label noise is estimated without touching validation: we hold out 3,000 rows of
the *training* partition, pretend they are unlabeled, run the same pseudo-labelling step
from the remaining 5,000 rows and measure how many pseudo-labels are wrong.

Usage: python experiments/03_unlabeled.py [--model B_char_lr]
Output: results/q3_unlabeled_<model>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, get_split, load_csv  # noqa: E402
from src.models import SEED, build, proba  # noqa: E402

LO, HI, ROUNDS = 0.1, 0.9, 2


def chosen_params(name):
    comp = json.loads((ROOT / "results" / "q1_comparison.json").read_text())
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in comp[name]["params"].items()}


def self_train(make, X, y, U, rounds=ROUNDS, lo=LO, hi=HI):
    m = make().fit(list(X), y)
    info = []
    for r in range(rounds):
        p = proba(m, U)
        sel = (p >= hi) | (p <= lo)
        pl = (p[sel] >= 0.5).astype(int)
        info.append({"round": r + 1, "n_pseudo": int(sel.sum()), "frac_ai": float(pl.mean()) if sel.any() else None})
        m = make().fit(list(X) + list(np.asarray(U)[sel]), np.concatenate([y, pl]))
    return m, info, sel, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="B_char_lr")
    args = ap.parse_args()
    params = chosen_params(args.model)
    tr, va = get_split()
    un = load_csv("unlabeled.csv")
    X, y, U = tr.text.to_numpy(), tr.label.to_numpy(), un.text.to_numpy()
    make = lambda: build(args.model, **params)  # noqa: E731
    out = {"model": args.model, "params": {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
           "thresholds": [LO, HI], "rounds": ROUNDS}

    t = time.time()
    sup = make().fit(list(X), y)
    out["supervised"] = {"val_auc": roc_auc_score(va.label, proba(sup, va.text)), "seconds": time.time() - t}
    print("supervised", out["supervised"], flush=True)

    # Unsupervised feature learning: fit vectoriser on train + unlabeled, classifier on train.
    t = time.time()
    m = make()
    vec, clf = m.named_steps["vec"], m.named_steps["clf"]
    vec.fit(list(X) + list(U))
    clf.fit(vec.transform(list(X)), y)
    s = clf.predict_proba(vec.transform(list(va.text)))[:, 1]
    out["tfidf_unlab"] = {"val_auc": roc_auc_score(va.label, s), "seconds": time.time() - t}
    print("tfidf_unlab", out["tfidf_unlab"], flush=True)

    t = time.time()
    st, info, sel, p_un = self_train(make, X, y, U)
    out["self_train"] = {"val_auc": roc_auc_score(va.label, proba(st, va.text)), "seconds": time.time() - t,
                         "rounds": info}
    print("self_train", out["self_train"], flush=True)
    out["unlabeled_score_hist"] = np.histogram(p_un, bins=10, range=(0, 1))[0].tolist()

    # Pseudo-label noise estimate using only the training partition.
    A, B = train_test_split(tr, test_size=3000, stratify=tr.label, random_state=SEED)
    mA = make().fit(list(A.text), A.label.to_numpy())
    pB = proba(mA, B.text)
    selB = (pB >= HI) | (pB <= LO)
    plB = (pB[selB] >= 0.5).astype(int)
    trueB = B.label.to_numpy()[selB]
    out["pseudo_label_noise_sim"] = {
        "n_pool": len(B), "n_selected": int(selB.sum()), "coverage": float(selB.mean()),
        "error_rate": float((plB != trueB).mean()),
        "error_rate_on_AI_pseudo": float((trueB[plB == 1] == 0).mean()),
        "error_rate_on_human_pseudo": float((trueB[plB == 0] == 1).mean()),
        "auc_on_pool": roc_auc_score(B.label, pB),
        "true_ai_frac_among_unselected": float(B.label.to_numpy()[~selB].mean()),
    }
    print("noise_sim", out["pseudo_label_noise_sim"], flush=True)
    (ROOT / "results" / f"q3_unlabeled_{args.model}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
