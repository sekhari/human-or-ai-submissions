"""Plot training and validation AUROC vs. training-set size (report Q2.2).

Example:
    python plot_learning_curve.py --results results_learning_curve.json --out learning_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results_learning_curve.json"))
    parser.add_argument("--out", type=Path, default=Path("learning_curve.png"))
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    by_size = results["by_size"]
    sizes = [int(n) for n in by_size]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, label, color, marker in [("train_auroc", "Training", "tab:blue", "o"),
                                      ("val_auroc", "Validation (fixed 2,000 rows)", "tab:orange", "s")]:
        mean = [by_size[str(n)][key]["mean"] for n in sizes]
        std = [by_size[str(n)][key]["std"] for n in sizes]
        ax.errorbar(sizes, mean, yerr=std, label=label, color=color, marker=marker, capsize=4, linewidth=2)
        for n, m in zip(sizes, mean):
            ax.annotate(f"{m:.4f}", (n, m), textcoords="offset points", xytext=(0, 8 if key == "train_auroc" else -16),
                        ha="center", fontsize=8, color=color)

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.minorticks_off()
    ax.set_ylim(0.984, 1.0015)
    ax.set_xlabel("Training rows (log scale)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"GBDT learning curve (mean ± std over {results['repeats']} samples)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
