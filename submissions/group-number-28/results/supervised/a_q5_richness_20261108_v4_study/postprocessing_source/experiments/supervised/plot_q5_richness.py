"""Plot recorded four-group length/richness evidence without fitting models."""

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


ORDER = ("c60", "c60_length", "c60_richness", "c60_length_richness")
CONDITIONS = ("original", "quote_typography", "dash_typography")
LABELS = {
    "c60": "R0  Character",
    "c60_length": "R1  Character + length",
    "c60_richness": "R2  Character + richness",
    "c60_length_richness": "R3  Character + both",
}
SHORT_LABELS = {key: LABELS[key].split("  ")[0] for key in ORDER}
COLORS = {"c60": "#788693", "c60_length": "#bd773c",
          "c60_richness": "#786298", "c60_length_richness": "#147d92"}
CONTRASTS = (
    ("richness_given_length", "c60_length_richness", "c60_length", "Richness with length\nR3 - R1 (primary)"),
    ("length_only", "c60_length", "c60", "Length alone\nR1 - R0"),
    ("richness_only", "c60_richness", "c60", "Richness alone\nR2 - R0"),
    ("length_given_richness", "c60_length_richness", "c60_richness", "Length with richness\nR3 - R2"),
)


def finite_auc(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid recorded AUROC: {value}")
    return value


def recorded_data(study):
    """Validate the fixed matrix, paired contrasts and equal-ID length groups."""
    rows = read_csv(study / "comparison.csv")
    models = set(ORDER) | {"incumbent_wc120"}
    expected = {(key, condition) for key in models for condition in CONDITIONS}
    matrix = {(row["model_id"], row["condition"]): row for row in rows}
    if len(rows) != len(expected) or set(matrix) != expected:
        raise ValueError("Expected four fixed groups plus the incumbent, each under three conditions")
    sizes = {int(row["n"]) for row in rows}
    if len(sizes) != 1 or min(sizes) <= 0:
        raise ValueError("The matrix must share one nonempty validation population")
    aucs = {key: finite_auc(row["validation_auroc"]) for key, row in matrix.items()}
    bootstrap = json.loads((study / "bootstrap.json").read_text())
    auc_intervals = {
        mid: interval(bootstrap["auroc_95_intervals"][f"{mid}__original"], mid)
        for mid in ORDER
    }
    contrasts = {}
    for name, left, right, _ in CONTRASTS:
        record = bootstrap["comparisons"][name]
        if (record["left"], record["right"]) != (f"{left}__original", f"{right}__original"):
            raise ValueError(f"Recorded contrast has different groups: {name}")
        delta = aucs[(left, "original")] - aucs[(right, "original")]
        if abs(delta - float(record["point_delta"])) > 1e-12:
            raise ValueError(f"Recorded contrast differs from matrix AUROCs: {name}")
        contrasts[name] = (delta, interval(record["interval95"], name))

    subgroup_rows = read_csv(study / "length_subgroups.csv")
    subgroups = {(row["model_id"], int(row["length_bin"])): row for row in subgroup_rows}
    expected_groups = {(mid, group) for mid in models for group in range(3)}
    if len(subgroup_rows) != len(expected_groups) or set(subgroups) != expected_groups:
        raise ValueError("Expected all five models in each of three descriptive length bins")
    counts = []
    for group in range(3):
        group_counts = {(int(subgroups[(mid, group)]["n"]),
                         int(subgroups[(mid, group)]["n_human"]),
                         int(subgroups[(mid, group)]["n_ai"])) for mid in models}
        if len(group_counts) != 1:
            raise ValueError("Length-bin populations differ across models")
        n, human, ai = next(iter(group_counts))
        if human <= 0 or ai <= 0 or n != human + ai:
            raise ValueError("Every length bin must include both labels and matching counts")
        counts.append(n)
        for mid in models:
            finite_auc(subgroups[(mid, group)]["validation_auroc"])
    if sum(counts) != next(iter(sizes)):
        raise ValueError("Length bins do not partition the recorded validation population")
    cuts = json.loads((study / "length_boundaries.json").read_text())["training_word_count_tertiles"]
    if len(cuts) != 2 or not all(math.isfinite(float(c)) for c in cuts) or cuts[0] > cuts[1]:
        raise ValueError("Invalid training-only length boundaries")
    return aucs, auc_intervals, contrasts, subgroups, counts, cuts, next(iter(sizes))


def plot(prefix, output_dir):
    study = ROOT / "results/supervised" / f"{prefix}_study"
    aucs, auc_intervals, contrasts, subgroups, counts, cuts, n = recorded_data(study)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 7.0),
                             gridspec_kw={"width_ratios": (1.0, 1.15, 1.15)})
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.25, top=0.75, wspace=0.77)
    for row, mid in enumerate(ORDER):
        value = aucs[(mid, "original")]
        draw_interval(axes[0], value, row, auc_intervals[mid], COLORS[mid])
        axes[0].annotate(f"{value:.6f}", (value, row), xytext=(0, 13),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_yticks(range(4), [LABELS[mid] for mid in ORDER])
    axes[0].set_ylim(3.6, -0.65)
    bounds = [v for pair in auc_intervals.values() for v in pair]
    incumbent = aucs[("incumbent_wc120", "original")]
    bounds.append(incumbent)
    pad = max(0.0015, (max(bounds) - min(bounds)) * 0.18)
    axes[0].set_xlim(max(0, min(bounds) - pad), min(1, max(bounds) + pad))
    axes[0].axvline(incumbent, color="#9ca8b0", linestyle="--", linewidth=1)
    axes[0].set_title("Original validation text", loc="left", pad=20)
    axes[0].set_xlabel("AUROC; conditional 95% intervals")
    axes[0].text(0, -0.19, f"Dashed line: WC120 incumbent = {incumbent:.6f}",
                 transform=axes[0].transAxes, fontsize=8.5, color="#505c68")

    contrast_bounds = [0.0]
    for row, (name, left, _, _) in enumerate(CONTRASTS):
        value, bounds = contrasts[name]
        contrast_bounds.extend(bounds)
        color = COLORS[left] if name != "richness_given_length" else "#147d92"
        draw_interval(axes[1], value, row, bounds, color)
        axes[1].annotate(f"{value:+.6f}", (value, row), xytext=(0, 13),
                         textcoords="offset points", ha="center", fontsize=9,
                         fontweight="bold" if row == 0 else "normal")
    axes[1].set_yticks(range(4), [item[3] for item in CONTRASTS])
    axes[1].get_yticklabels()[0].set_fontweight("bold")
    axes[1].set_ylim(3.6, -0.65)
    pad = max(0.0008, (max(contrast_bounds) - min(contrast_bounds)) * 0.18)
    axes[1].set_xlim(min(contrast_bounds) - pad, max(contrast_bounds) + pad)
    axes[1].axvline(0, color="#9ca8b0", linestyle="--", linewidth=1)
    axes[1].set_title("Paired factorial contrasts", loc="left", pad=20)
    axes[1].set_xlabel("AUROC difference; conditional 95% intervals")

    length_values = []
    for mid in ORDER:
        values = [float(subgroups[(mid, group)]["validation_auroc"]) for group in range(3)]
        length_values.extend(values)
        axes[2].plot(range(3), values, marker="o", color=COLORS[mid], linewidth=1.8,
                     markersize=5, label=SHORT_LABELS[mid])
    axes[2].set_xticks(range(3), [f"Short\nn={counts[0]:,}", f"Middle\nn={counts[1]:,}", f"Long\nn={counts[2]:,}"])
    axes[2].set_xlim(-0.15, 2.15)
    pad = max(0.002, (max(length_values) - min(length_values)) * 0.20)
    axes[2].set_ylim(max(0, min(length_values) - pad), min(1, max(length_values) + pad))
    axes[2].set_title("Original-text length groups", loc="left", pad=20)
    axes[2].set_xlabel(f"Training word-count cutpoints: {cuts[0]:g}, {cuts[1]:g}")
    axes[2].set_ylabel("Descriptive subgroup AUROC")
    axes[2].legend(loc="best", ncols=2, fontsize=8.5, frameon=True,
                   edgecolor="none", facecolor="white", framealpha=0.92)
    for axis in axes[:2]:
        axis.grid(axis="x", color="#e5e9ed")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.set_axisbelow(True)
    axes[2].grid(axis="y", color="#e5e9ed")
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].set_axisbelow(True)

    primary, primary_bounds = contrasts["richness_given_length"]
    fig.suptitle("Q5: does lexical richness add information beyond explicit length features?",
                 x=0.14, y=0.97, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.14, 0.87,
             f"Primary R3 - R1 = {primary:+.6f}  "
             f"(conditional 95% interval [{primary_bounds[0]:+.6f}, {primary_bounds[1]:+.6f}]).  "
             f"Same {n:,} validation IDs; fixed character features and SVM C=1.",
             ha="left", fontsize=10, color="#364958")
    fig.text(0.14, 0.09,
             "R0: character; R1: + length; R2: + richness; R3: + both. Scaling is fitted within training/calibration folds; no row normalization is added.\n"
             "The primary contrast measures incremental prediction conditional on this explicit length block, not causal or complete independence from text length.\n"
             "Intervals condition on fitted models and this development split; training variability and selection adjustment are omitted. Length curves are descriptive; no Kaggle gain is implied.",
             ha="left", va="center", fontsize=9, color="#505c68", linespacing=1.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"q5_{prefix}_richness.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved recorded Q5 richness figure to {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report/figures")
    args = parser.parse_args()
    plot(args.prefix, args.output_dir)


if __name__ == "__main__":
    main()
