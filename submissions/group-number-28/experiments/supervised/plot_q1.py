"""Generate Q1 figures from recorded scores, without fitting or changing results."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "q1_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from human_ai.data import ROOT, read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    study = ROOT / "results/supervised" / f"{args.prefix}_study"
    rows = read_csv(study / "comparison.csv")
    intervals = json.loads((study / "bootstrap.json").read_text())["auroc_95_intervals"]
    out = ROOT / "report/figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})
    colors = {"lr": "#167d9a", "nb": "#b66338", "svm": "#6d58a5"}
    fig, ax = plt.subplots(figsize=(9.5, 3.6), constrained_layout=True)
    for i, row in enumerate(rows):
        auc = float(row["validation_auroc"])
        low, high = intervals[row["model"]]
        ax.errorbar(auc, i, xerr=[[auc-low], [high-auc]], fmt="o", color=colors[row["model"]],
                    markersize=8, capsize=4, lw=2)
        ax.scatter(float(row["train_auroc"]), i, marker="^", color="#8c959f", s=45,
                   label="Training AUROC (in sample)" if i == 0 else None)
        ax.annotate(f"{auc:.6f}", (auc, i), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9)
    ax.set_yticks(range(3), [r["name"] for r in rows])
    ax.set_xlim(0.875, 1.01)
    ax.set_ylim(2.5, -0.6)
    ax.set_xlabel("AUROC")
    ax.set_title("Q1: matched features and fixed 8,000 / 2,000 split", loc="left", pad=15)
    ax.grid(axis="x", color="#e5e9ed")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Dots: validation AUROC; whiskers: conditional 95% bootstrap intervals (not selection-adjusted)",
                 y=-0.02, fontsize=9, color="#505c68")
    fig.savefig(out / f"q1_{args.prefix}_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    cv = read_csv(study / "cv_results.csv")
    chosen = json.loads((study / "chosen_parameters.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=True, constrained_layout=True)
    for ax, family, title in zip(axes, ("lr", "nb", "svm"),
                                 ("Logistic regression", "Multinomial Naive Bayes", "Linear SVM"), strict=True):
        group = [r for r in cv if r["model"] == family]
        x = [float(r["value"]) for r in group]
        y = [float(r["mean_cv_auroc"]) for r in group]
        sd = [float(r["std_cv_auroc"]) for r in group]
        ax.errorbar(x, y, yerr=sd, fmt="o-", capsize=4, color=colors[family], lw=1.8)
        best = x.index(chosen[family])
        ax.scatter(x[best], y[best], s=140, facecolors="none", edgecolors=colors[family], lw=2)
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{value:g}" for value in x])
        ax.set_xlabel("alpha" if family == "nb" else "C")
        ax.set_title(title, fontsize=11, pad=12)
        ax.grid(axis="y", color="#e5e9ed")
        ax.set_ylim(0.865, 0.95)
        for value, auc in zip(x, y, strict=True):
            ax.annotate(f"{auc:.4f}", (value, auc), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_ylabel("Mean inner-fold AUROC")
    fig.suptitle("Training-only tuning: 3 shared stratified folds; error bars show fold SD; rings mark selected values", fontsize=11)
    fig.savefig(out / f"q1_{args.prefix}_cv.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Q1 figures to {out}")


if __name__ == "__main__":
    main()
