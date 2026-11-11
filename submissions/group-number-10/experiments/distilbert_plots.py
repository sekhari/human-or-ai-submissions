"""Figures for the DistilBERT experiments, built only from recorded results.

    python -m experiments.distilbert_plots                     # uses the baseline run for curves
    python -m experiments.distilbert_plots --run <run_name>

Writes to results/figures/:
    distilbert_loss_<run>.png       training vs validation loss by epoch
    distilbert_val_auc_by_epoch.png validation AUROC by epoch, one line per run
    distilbert_hparams.png          train/val AUROC for each configuration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import RESULTS_DIR  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
BASELINE = "distilbert_lr2e-05_ep3_bs16_len512_seed42"
# Validated categorical palette, assigned in fixed order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2,
})


def load_runs() -> list[dict]:
    runs = [json.loads(p.read_text()) for p in sorted((RESULTS_DIR / "metrics").glob("distilbert_*.json"))]
    return sorted(runs, key=lambda r: (r["hyperparameters"]["learning_rate"], r["hyperparameters"]["epochs"]))


def label(run: dict) -> str:
    h = run["hyperparameters"]
    return f"lr {h['learning_rate']:g}, {h['epochs']} ep"


def plot_loss(run: dict) -> Path:
    hist = run["history"]
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (key, name) in enumerate((("train_loss", "Training"), ("val_loss", "Validation"))):
        ys = [h[key] for h in hist]
        ax.plot(ep, ys, color=SERIES[i], marker="o", markersize=6, label=name)
        ax.annotate(name, (ep[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK_2)
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", xticks=ep,
           title=f"DistilBERT loss by epoch ({label(run)})")
    ax.set_xlim(ep[0] - 0.2, ep[-1] + 0.6)
    ax.legend(frameon=False)
    return _save(fig, f"distilbert_loss_{run['run_name']}.png")


def plot_val_auc(runs: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, run in enumerate(runs):
        hist = run["history"]
        ax.plot([h["epoch"] for h in hist], [h["val_auc"] for h in hist], color=SERIES[i % len(SERIES)],
                marker="o", markersize=6, label=label(run))
    ax.set(xlabel="Epoch", ylabel="Validation AUROC", title="DistilBERT validation AUROC by epoch")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, "distilbert_val_auc_by_epoch.png")


def plot_hparams(runs: list[dict]) -> Path:
    names = [label(r) for r in runs]
    y = range(len(runs))
    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(runs) + 1.6))
    ax.scatter([r["train_auc"] for r in runs], y, s=60, color=SERIES[0], label="Training AUROC", zorder=3,
               edgecolors=SURFACE, linewidths=2)
    ax.scatter([r["val_auc"] for r in runs], y, s=60, color=SERIES[1], label="Validation AUROC", zorder=3,
               edgecolors=SURFACE, linewidths=2)
    for i, r in enumerate(runs):
        ax.annotate(f"{r['val_auc']:.4f}", (r["val_auc"], i), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
    ax.set_yticks(list(y), names)
    ax.set(xlabel="AUROC (best epoch checkpoint)", title="DistilBERT hyperparameter comparison")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=9)
    ax.set_title("DistilBERT hyperparameter comparison", pad=28)
    return _save(fig, "distilbert_hparams.png")


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=BASELINE, help="run whose loss curves to plot")
    a = p.parse_args(argv)
    runs = load_runs()
    if not runs:
        raise SystemExit("No DistilBERT results in results/metrics/ yet. Train first.")
    by_name = {r["run_name"]: r for r in runs}
    plot_loss(by_name.get(a.run, runs[0]))
    plot_val_auc(runs)
    if len(runs) > 1:
        plot_hparams(runs)


if __name__ == "__main__":
    main()
