"""Phase 3: effect of training-set size (report Q2).

Fixed configuration: tuned masked_lr. It has no layout features and was the strongest text model,
so the curve reflects writing patterns rather than the blank-line shortcut. word_lr is run too as a
comparison. Subsets are stratified and nested within a seed, and sampled from the 8,000-row
training partition only. The same fixed validation set and hyperparameters are used throughout.

Usage:
    python -m experiments.learning_curve

Outputs (results/learning_curve/): runs.csv, summary.csv, summary.md, learning_curve.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from src.models import MODELS, build_tuned, scores
from src.split import load_train_val

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "learning_curve"
SIZES = [500, 1000, 2000, 5000, 8000]
SEEDS = [0, 1, 2, 3, 4]
REPORT_MODEL = "masked_lr"
COMPARISON_MODELS = ["masked_lr", "word_lr"]

COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#d95926"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e7e6e2"


def nested_order(train, seed):
    """Row order such that any stratified prefix is a subset of every longer prefix."""
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(np.flatnonzero(train.label.values == c)) for c in (0, 1)]
    return np.column_stack(orders).ravel()  # alternates classes, so every prefix stays balanced


def run_one(name, params, train, val, seed, size):
    spec = MODELS[name]
    subset = train.iloc[nested_order(train, seed)[:size]]
    y = subset.label.astype(int).values
    model = build_tuned(name, params).fit(spec.inputs(subset), y)
    return {
        "model": name, "seed": seed, "size": size,
        "train_auroc": roc_auc_score(y, scores(model, spec.inputs(subset))),
        "val_auroc": roc_auc_score(val.label.astype(int), scores(model, spec.inputs(val))),
    }


def plot(summary, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    d = summary[summary.model == REPORT_MODEL]
    for col, color, label, offset in (("train", COLOR_TRAIN, "Training", -14), ("val", COLOR_VAL, "Validation", 10)):
        mean, std = d[f"{col}_mean"], d[f"{col}_std"]
        ax.plot(d["size"], mean, color=color, lw=2, marker="o", ms=7, mec="white", mew=1.5, label=label)
        ax.fill_between(d["size"], mean - std, mean + std, color=color, alpha=0.18, lw=0)
        ax.annotate(f"{label} {mean.iloc[-1]:.3f}", (d["size"].iloc[-1], mean.iloc[-1]),
                    xytext=(0, offset), textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(SIZES, [f"{s:,}" for s in SIZES])
    ax.minorticks_off()
    ax.set_xlabel("Training rows (log scale)", color=INK_MUTED)
    ax.set_ylabel("AUROC", color=INK_MUTED)
    ax.set_title("masked_lr: more labeled text keeps raising validation AUROC", color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.72), labelcolor=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val = load_train_val()
    best = json.loads((ROOT / "results" / "model_comparison" / "best_params.json").read_text())
    tasks = [delayed(run_one)(m, best[m], train, val, seed, size)
             for m in COMPARISON_MODELS for size in SIZES for seed in SEEDS]
    runs = pd.DataFrame(Parallel(n_jobs=-1)(tasks))
    runs.to_csv(OUT_DIR / "runs.csv", index=False)

    summary = runs.groupby(["model", "size"], as_index=False).agg(
        train_mean=("train_auroc", "mean"), train_std=("train_auroc", "std"),
        val_mean=("val_auroc", "mean"), val_std=("val_auroc", "std"))
    summary["gap"] = summary.train_mean - summary.val_mean
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    (OUT_DIR / "summary.md").write_text(summary.round(4).to_markdown(index=False) + "\n")
    plot(summary, OUT_DIR / "learning_curve.png")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
