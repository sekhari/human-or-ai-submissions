"""Plot the recorded six-configuration Q5 tuning study without fitting models."""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "q5_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "q5_xdg_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from human_ai.data import ROOT, read_csv
from .plot_q5 import draw_interval, interval


REPRESENTATIONS = ("c60", "wc120")
CS = (0.1, 1.0, 10.0)
COLORS = {"c60": "#bd773c", "wc120": "#147d92", "baseline_wc120": "#667b8c"}
REP_LABELS = {"c60": "C60: character", "wc120": "WC120: word + character"}


def finite_auc(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid AUROC: {value}")
    return value


def config_label(row):
    return f"{REP_LABELS[row['representation']]} | C={float(row['C']):g}"


def plot(prefix, output_dir):
    study = ROOT / "results/supervised" / f"{prefix}_study"
    cv = read_csv(study / "cv_results.csv")
    comparison = read_csv(study / "comparison.csv")
    bootstrap = json.loads((study / "bootstrap.json").read_text())

    if len(cv) != 6 or len({row["candidate_id"] for row in cv}) != 6:
        raise ValueError("Expected all six unique frozen candidate configurations")
    if {(row["representation"], float(row["C"])) for row in cv} != {
        (representation, value) for representation in REPRESENTATIONS for value in CS
    }:
        raise ValueError("Recorded CV grid differs from C60/WC120 x C={0.1,1,10}")
    cv.sort(key=lambda row: (REPRESENTATIONS.index(row["representation"]), float(row["C"])))
    by_id = {row["candidate_id"]: row for row in comparison}
    if len(comparison) != 3 or "baseline_wc120" not in by_id:
        raise ValueError("Expected the current baseline and two CV-selected finalists")
    finalists = [row for row in comparison if row["candidate_id"] != "baseline_wc120"]
    if {row["representation"] for row in finalists} != set(REPRESENTATIONS):
        raise ValueError("Expected one finalist per representation")
    finalist_ids = {row["candidate_id"] for row in finalists}
    if not finalist_ids <= {row["candidate_id"] for row in cv}:
        raise ValueError("Finalist missing from CV evidence")
    finalists.sort(key=lambda row: REPRESENTATIONS.index(row["representation"]))
    validation = [by_id["baseline_wc120"], *finalists]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.1), gridspec_kw={"width_ratios": (1.1, 1.0)})
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.21, top=0.77, wspace=0.55)

    cv_points = []
    for row_number, row in enumerate(cv):
        color = COLORS[row["representation"]]
        fold_scores = [finite_auc(row[f"fold_{fold}_auroc"]) for fold in range(3)]
        mean = finite_auc(row["mean_cv_auroc"])
        if abs(sum(fold_scores) / 3 - mean) > 1e-12:
            raise ValueError(f"Recorded mean does not match fold scores: {row['candidate_id']}")
        cv_points.extend(fold_scores)
        cv_points.append(mean)
        for fold, value in enumerate(fold_scores):
            axes[0].scatter(value, row_number + (fold - 1) * 0.10,
                            s=28, color=color, alpha=0.46, zorder=2)
        selected = row["candidate_id"] in finalist_ids
        axes[0].scatter(mean, row_number, marker="D", s=64, color=color,
                        edgecolors="#172b3a" if selected else "white",
                        linewidths=1.5 if selected else 0.7, zorder=3)
        axes[0].annotate(f"{mean:.6f}", (mean, row_number), xytext=(0, -17),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_yticks(range(6), [config_label(row) for row in cv])
    axes[0].set_ylim(5.55, -0.65)
    cv_pad = max(0.0015, (max(cv_points) - min(cv_points)) * 0.16)
    axes[0].set_xlim(max(0, min(cv_points) - cv_pad), min(1, max(cv_points) + cv_pad))
    axes[0].set_title("Training-only CV: all six settings", loc="left", pad=18)
    axes[0].set_xlabel("Raw-margin AUROC; small dots are folds, diamonds are means")

    baseline_auc = finite_auc(by_id["baseline_wc120"]["validation_auroc"])
    bounds = []
    labels = []
    for row_number, row in enumerate(validation):
        key = row["candidate_id"]
        value = finite_auc(row["validation_auroc"])
        recorded_interval = interval(bootstrap["auroc_95_intervals"][key], key)
        bounds.extend(recorded_interval)
        color = COLORS["baseline_wc120"] if key == "baseline_wc120" else COLORS[row["representation"]]
        draw_interval(axes[1], value, row_number, recorded_interval, color)
        axes[1].annotate(f"{value:.6f}", (value, row_number), xytext=(0, 13),
                         textcoords="offset points", ha="center", fontsize=10)
        if key == "baseline_wc120":
            labels.append("Current WC120 | C=1")
            detail = "Reference candidate"
        else:
            labels.append(config_label(row))
            delta = value - baseline_auc
            delta_interval = interval(bootstrap["delta_vs_baseline_95_intervals"][key], key + " delta")
            detail = (f"Difference {delta:+.6f}\n"
                      f"95% interval [{delta_interval[0]:+.6f}, {delta_interval[1]:+.6f}]")
        axes[1].annotate(detail, (value, row_number), xytext=(0, -17),
                         textcoords="offset points", ha="center", va="top",
                         fontsize=9, color="#405362", linespacing=1.3)
    axes[1].set_yticks(range(3), labels)
    axes[1].set_ylim(2.65, -0.65)
    validation_pad = max(0.002, (max(bounds) - min(bounds)) * 0.16)
    axes[1].set_xlim(max(0, min(bounds) - validation_pad), min(1, max(bounds) + validation_pad))
    axes[1].axvline(baseline_auc, linestyle="--", color="#a5b0b8", linewidth=1, zorder=0)
    axes[1].set_title("Development validation: two finalists", loc="left", pad=18)
    axes[1].set_xlabel("Calibrated-probability AUROC; conditional 95% intervals")
    for axis in axes:
        axis.grid(axis="x", color="#e5e9ed")
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.suptitle("Q5 follow-up: representation and SVM regularization", x=0.23,
                 y=0.97, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.23, 0.88, "C chosen within each representation using three training-only folds. "
             "Outlined diamonds mark the finalists.", ha="left", fontsize=10, color="#364958")
    fig.text(0.23, 0.075,
             "Fixed 8,000 / 2,000 split. CV and validation panels use different fitting/evaluation samples.\n"
             "Validation intervals condition on fitted models and split; they omit training variability and selection adjustment.\n"
             "This follow-up was motivated by public feedback; no new Kaggle score is implied by this figure.",
             ha="left", va="center", fontsize=9, color="#505c68", linespacing=1.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"q5_{prefix}_tuning.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved recorded Q5 tuning figure to {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report/figures")
    args = parser.parse_args()
    plot(args.prefix, args.output_dir)


if __name__ == "__main__":
    main()
