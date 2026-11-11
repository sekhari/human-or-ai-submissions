"""Plot recorded Q5 evidence without fitting models or generating report text."""

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

from human_ai.data import ROOT, read_csv


ORDER = ("w60", "c60", "wc120", "w120")
LABELS = {
    "w60": "W60: word baseline",
    "c60": "C60: character only",
    "wc120": "WC120: word + character",
    "w120": "W120: larger word vocabulary",
}
COLORS = {
    "w60": "#667b8c",
    "c60": "#bd773c",
    "wc120": "#147d92",
    "w120": "#786298",
}


def interval(values, name):
    """Reject invalid recorded intervals rather than silently hiding them."""
    if len(values) != 2:
        raise ValueError(f"Expected two interval endpoints for {name}")
    low, high = map(float, values)
    if not all(math.isfinite(v) for v in (low, high)) or low > high:
        raise ValueError(f"Invalid interval for {name}: {values}")
    return low, high


def draw_interval(ax, value, row, bounds, color):
    """Draw percentile intervals even if the point falls outside the interval."""
    low, high = bounds
    ax.hlines(row, low, high, color=color, linewidth=2)
    ax.vlines((low, high), row - 0.075, row + 0.075, color=color, linewidth=1.5)
    ax.scatter(value, row, color=color, s=55, zorder=3)


def plot(prefix, output_dir):
    study = ROOT / "results/supervised" / f"{prefix}_study"
    recorded = read_csv(study / "comparison.csv")
    rows = {row["model"]: row for row in recorded}
    if len(recorded) != len(ORDER) or set(rows) != set(ORDER):
        raise ValueError("Q5 comparison must contain each frozen configuration exactly once")
    bootstrap = json.loads((study / "bootstrap.json").read_text())
    aucs = {key: float(rows[key]["validation_auroc"]) for key in ORDER}
    times = {key: float(rows[key]["fit_seconds"]) for key in ORDER}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in aucs.values()):
        raise ValueError("Invalid recorded AUROC")
    if not all(math.isfinite(v) and v >= 0 for v in times.values()):
        raise ValueError("Invalid recorded fit time")
    auc_intervals = {
        key: interval(bootstrap["auroc_95_intervals"][key], key) for key in ORDER
    }
    primary_delta = aucs["wc120"] - aucs["w60"]
    if abs(primary_delta - float(bootstrap["primary_delta"])) > 1e-12:
        raise ValueError("Primary delta does not match the recorded comparison")
    delta_intervals = {"w60": (0.0, 0.0)}
    for key in ORDER[1:]:
        delta_intervals[key] = interval(
            bootstrap["deltas_vs_w60_95_intervals"][key], f"{key} minus w60"
        )
    primary_interval = interval(bootstrap["primary_delta_95_interval"], "primary delta")
    if any(abs(a - b) > 1e-12 for a, b in zip(primary_interval, delta_intervals["wc120"])):
        raise ValueError("Primary and per-model delta intervals do not match")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(15.2, 4.9), sharey=True,
        gridspec_kw={"width_ratios": (1.2, 0.9, 1.1)},
    )
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.24, top=0.77, wspace=0.27)
    for i, key in enumerate(ORDER):
        color = COLORS[key]
        draw_interval(axes[0], aucs[key], i, auc_intervals[key], color)
        axes[0].annotate(
            f"{aucs[key]:.6f}", (aucs[key], i), xytext=(0, 12),
            textcoords="offset points", ha="center", fontsize=9,
        )
        axes[1].barh(i, times[key], height=0.42, color=color, alpha=0.88)
        axes[1].annotate(
            f"{times[key]:.2f} s", (times[key], i), xytext=(5, 0),
            textcoords="offset points", va="center", fontsize=9,
        )
        delta = aucs[key] - aucs["w60"]
        draw_interval(axes[2], delta, i, delta_intervals[key], color)
        axes[2].annotate(
            "Reference" if key == "w60" else f"{delta:+.6f}",
            (delta, i), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9,
        )

    axes[0].set_yticks(range(len(ORDER)), [LABELS[key] for key in ORDER])
    axes[0].set_ylim(len(ORDER) - 0.5, -0.65)
    for label, key in zip(axes[0].get_yticklabels(), ORDER):
        if key == "wc120":
            label.set_fontweight("bold")
            label.set_color(COLORS[key])
    auc_min = min(v[0] for v in auc_intervals.values())
    auc_max = max(v[1] for v in auc_intervals.values())
    auc_pad = max(0.002, (auc_max - auc_min) * 0.16)
    axes[0].set_xlim(max(0, auc_min - auc_pad), min(1, auc_max + auc_pad))
    axes[0].set_title("Validation AUROC", loc="left", pad=20)
    axes[0].set_xlabel("AUROC")
    axes[0].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[1].set_title("Full fit time", loc="left", pad=20)
    axes[1].set_xlabel("Seconds, one CPU thread")
    axes[1].set_xlim(0, max(1.0, max(times.values()) * 1.38))
    delta_min = min(v[0] for v in delta_intervals.values())
    delta_max = max(v[1] for v in delta_intervals.values())
    delta_pad = max(0.001, (delta_max - delta_min) * 0.16)
    axes[2].set_xlim(delta_min - delta_pad, delta_max + delta_pad)
    axes[2].axvline(0, color="#87939e", linestyle="--", linewidth=1, zorder=0)
    axes[2].set_title("Change from W60", loc="left", pad=20)
    axes[2].set_xlabel("Paired AUROC difference")
    axes[2].ticklabel_format(axis="x", style="plain", useOffset=False)
    for ax in axes:
        ax.grid(axis="x", color="#e5e9ed")
        ax.set_axisbelow(True)
    fig.suptitle(
        "Q5: do character features complement word TF-IDF?",
        x=0.20, y=0.97, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.20, 0.88,
        f"Primary comparison: WC120 - W60 = {primary_delta:+.6f} "
        f"(conditional 95% interval [{primary_interval[0]:+.6f}, {primary_interval[1]:+.6f}])",
        ha="left", fontsize=10, color="#364958",
    )
    fig.text(
        0.20, 0.10,
        "Fixed 8,000 / 2,000 development split; SVM C=1 for every configuration.\n"
        "Whiskers: paired class-stratified bootstrap intervals, conditional on fitted models and split.\n"
        "Intervals are not selection-adjusted and omit training variability. Fit time includes features and calibration.",
        ha="left", va="center", fontsize=9, color="#505c68", linespacing=1.5,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"q5_{prefix}_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved recorded Q5 comparison figure to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report/figures")
    args = parser.parse_args()
    plot(args.prefix, args.output_dir)


if __name__ == "__main__":
    main()
