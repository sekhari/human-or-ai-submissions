"""Q2: effect of training-set size.

One fixed configuration (default: Model C stylometric GBM, the best Q1 model, with its CV-chosen setting) is
trained on 500/1,000/2,000/5,000/8,000 rows sampled (stratified) from the 8,000-row
training partition and scored on the fixed validation set. We repeat the subsampling with
3 seeds and report mean and std; 8,000 uses the whole partition so its std is 0.

Usage: python experiments/02_learning_curve.py [--model C_style_gbm]
Outputs: results/q2_learning_curve_<model>.json, figures/q2_learning_curve_<model>.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, get_split  # noqa: E402
from src.models import build, proba  # noqa: E402

SIZES = [500, 1000, 2000, 5000, 8000]
SEEDS = [0, 1, 2]


def chosen_params(name):
    comp = json.loads((ROOT / "results" / "q1_comparison.json").read_text())
    p = comp[name]["params"]
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in p.items()}

NICE = {"C_style_gbm": "Model C: stylometric features + gradient boosting",
        "B_char_lr": "Model B: char 2-5-gram TF-IDF + logistic regression"}


def plot(model, summary):
    from matplotlib.ticker import NullLocator
    ns = [s["n"] for s in summary]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for key, label, color, marker in [("train", "Training AUROC", "#2a6f97", "o"),
                                       ("val", "Validation AUROC (fixed 2,000 rows)", "#d1495b", "s")]:
        mu = np.array([s[f"{key}_auc_mean"] for s in summary])
        sd = np.array([s[f"{key}_auc_std"] for s in summary])
        ax.plot(ns, mu, marker=marker, color=color, label=label)
        ax.fill_between(ns, mu - sd, mu + sd, color=color, alpha=0.15)
        if key == "val":
            for x, v in zip(ns, mu):
                ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, -14),
                            ha="center", fontsize=8, color=color)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n:,}" for n in ns])
    lo = min(s["val_auc_mean"] - s["val_auc_std"] for s in summary)
    ax.set_ylim(lo - 0.01, 1.004)
    ax.set_xlabel("Training rows (log scale)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"{NICE.get(model, model)}\nmean \u00b1 sd over 3 random subsamples (8,000 = full partition)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / f"q2_learning_curve_{model}.png", dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="C_style_gbm")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from saved results")
    args = ap.parse_args()
    if args.plot_only:
        plot(args.model, json.loads((ROOT / "results" / f"q2_learning_curve_{args.model}.json").read_text())["summary"])
        return
    params = chosen_params(args.model)
    tr, va = get_split()
    rows = []
    for n in SIZES:
        for seed in (SEEDS if n < len(tr) else [0]):
            if n < len(tr):
                sub, _ = train_test_split(tr, train_size=n, stratify=tr.label, random_state=seed)
            else:
                sub = tr
            m = build(args.model, **params)
            t = time.time()
            m.fit(list(sub.text), sub.label.to_numpy())
            rows.append({"n": n, "seed": seed, "fit_seconds": time.time() - t,
                         "train_auc": roc_auc_score(sub.label, proba(m, sub.text)),
                         "val_auc": roc_auc_score(va.label, proba(m, va.text))})
            print(rows[-1], flush=True)

    summary = []
    for n in SIZES:
        r = [x for x in rows if x["n"] == n]
        summary.append({"n": n,
                        "train_auc_mean": float(np.mean([x["train_auc"] for x in r])),
                        "train_auc_std": float(np.std([x["train_auc"] for x in r])),
                        "val_auc_mean": float(np.mean([x["val_auc"] for x in r])),
                        "val_auc_std": float(np.std([x["val_auc"] for x in r])),
                        "fit_seconds_mean": float(np.mean([x["fit_seconds"] for x in r]))})
    out = {"model": args.model, "params": {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
           "runs": rows, "summary": summary}
    (ROOT / "results" / f"q2_learning_curve_{args.model}.json").write_text(json.dumps(out, indent=1))

    plot(args.model, summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
