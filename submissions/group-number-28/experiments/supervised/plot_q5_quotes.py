"""Plot recorded quotation-normalization ablations without fitting models."""

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
CONDITIONS = ("original", "quote_typography", "dash_typography")
CONDITION_LABELS = ("Original", "Quote mapping", "Dash mapping")
REPRESENTATION_LABELS = {"c60": "C60: character", "wc120": "WC120: word + character"}
COLORS = {"c60": "#bd773c", "wc120": "#147d92", "raw": "#738491"}
CONTRASTS = ("c60_clean", "c60_quote", "wc120_clean", "wc120_quote")


def finite_auc(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid recorded AUROC: {value}")
    return value


def recorded_data(study):
    """Reject incomplete or mismatched recorded comparisons before plotting."""
    rows = read_csv(study / "comparison.csv")
    expected_models = {f"{rep}_{mode}" for rep in REPRESENTATIONS for mode in ("raw", "norm")}
    expected = {(model, condition) for model in expected_models for condition in CONDITIONS}
    lookup = {(row["model_id"], row["condition"]): row for row in rows}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("Expected four model configurations under three paired conditions")
    sizes = {int(row["n"]) for row in rows}
    if len(sizes) != 1 or min(sizes) <= 0:
        raise ValueError("Conditions must use the same nonempty evaluation population")
    for (model, condition), row in lookup.items():
        if row["representation"] != model.rsplit("_", 1)[0]:
            raise ValueError("Model identifier and representation disagree")
        finite_auc(row["validation_auroc"])
        changed = int(row["n_changed"])
        if not 0 <= changed <= int(row["n"]) or (condition == "original" and changed != 0):
            raise ValueError("Invalid modified-document count")
    bootstrap = json.loads((study / "bootstrap.json").read_text())
    recorded = bootstrap["comparisons"]
    if not set(CONTRASTS) <= set(recorded):
        raise ValueError("Missing clean or quote paired contrasts")
    contrasts = {}
    for name in CONTRASTS:
        rep, condition_name = name.rsplit("_", 1)
        condition = "original" if condition_name == "clean" else "quote_typography"
        delta = (float(lookup[(f"{rep}_norm", condition)]["validation_auroc"]) -
                 float(lookup[(f"{rep}_raw", condition)]["validation_auroc"]))
        if abs(delta - float(recorded[name]["point_delta"])) > 1e-12:
            raise ValueError(f"Recorded matched contrast differs from comparison: {name}")
        contrasts[name] = (delta, interval(recorded[name]["interval95"], name))
    return lookup, contrasts, next(iter(sizes))


def plot(prefix, output_dir):
    study = ROOT / "results/supervised" / f"{prefix}_study"
    lookup, contrasts, n_validation = recorded_data(study)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(17.8, 6.2), gridspec_kw={"width_ratios": (1.0, 1.0, 1.25)},
    )
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.25, top=0.76, wspace=0.52)

    auc_values = [float(row["validation_auroc"]) for row in lookup.values()]
    spread = max(auc_values) - min(auc_values)
    pad = max(0.0015, spread * 0.40)
    limits = (max(0, min(auc_values) - pad), min(1, max(auc_values) + pad))
    for panel, rep in enumerate(REPRESENTATIONS):
        axis = axes[panel]
        for mode, color, marker, linestyle in (
                ("raw", COLORS["raw"], "o", "--"),
                ("norm", COLORS[rep], "D", "-")):
            values = [float(lookup[(f"{rep}_{mode}", condition)]["validation_auroc"])
                      for condition in CONDITIONS]
            axis.plot(range(3), values, color=color, marker=marker, linestyle=linestyle,
                      linewidth=1.9, markersize=6,
                      label="Raw training" if mode == "raw" else "Quote-normalized training")
            for position, value in enumerate(values):
                axis.annotate(f"{value:.6f}", (position, value),
                              xytext=(0, 12 if mode == "raw" else -18),
                              textcoords="offset points", ha="center", fontsize=8.5, color=color)
        axis.set_xticks(range(3), CONDITION_LABELS)
        axis.set_xlim(-0.20, 2.20)
        axis.set_ylim(*limits)
        axis.set_title(REPRESENTATION_LABELS[rep], loc="left", pad=18)
        axis.set_xlabel("Evaluation text condition")
        axis.set_ylabel("Development AUROC" if panel == 0 else "")
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(axis="y", color="#e5e9ed")
        axis.set_axisbelow(True)
        axis.legend(loc="lower left", fontsize=8, frameon=True,
                    facecolor="white", edgecolor="none", framealpha=0.92)

    labels, bounds = [], [0.0]
    for position, name in enumerate(CONTRASTS):
        rep, condition = name.rsplit("_", 1)
        value, recorded_interval = contrasts[name]
        bounds.extend(recorded_interval)
        draw_interval(axes[2], value, position, recorded_interval, COLORS[rep])
        labels.append(f"{rep.upper()} | {'original' if condition == 'clean' else 'quote-mapped'}")
        axes[2].annotate(f"{value:+.6f}", (value, position), xytext=(0, 12),
                         textcoords="offset points", ha="center", fontsize=9)
    delta_pad = max(0.0008, (max(bounds) - min(bounds)) * 0.20)
    axes[2].set_xlim(min(bounds) - delta_pad, max(bounds) + delta_pad)
    axes[2].set_yticks(range(4), labels)
    axes[2].set_ylim(3.5, -0.60)
    axes[2].axvline(0, color="#98a6af", linestyle="--", linewidth=1, zorder=0)
    axes[2].set_title("Matched change: normalized - raw training", loc="left", pad=18)
    axes[2].set_xlabel("Paired AUROC difference; conditional 95% interval")
    axes[2].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].grid(axis="x", color="#e5e9ed")
    axes[2].set_axisbelow(True)

    quote_changed = int(lookup[("wc120_raw", "quote_typography")]["n_changed"])
    dash_changed = int(lookup[("wc120_raw", "dash_typography")]["n_changed"])
    fig.suptitle("Q5: quotation canonicalization as a training intervention",
                 x=0.065, y=0.97, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.065, 0.88,
             f"Fixed SVM C=1; same {n_validation:,} validation IDs. "
             f"Quote mapping changes {quote_changed:,} texts; dash mapping changes {dash_changed:,}.",
             ha="left", fontsize=10, color="#364958")
    fig.text(0.065, 0.09,
             "The normalized model canonicalizes quotes during both training and prediction; raw-training + quote-mapped input is the inference-only control.\n"
             "Quote invariance of the normalized pipeline is enforced by construction, not independent evidence of out-of-domain generalization.\n"
             "Intervals condition on fitted models and this development split; training variation and selection adjustment are omitted. No Kaggle gain is implied.",
             ha="left", va="center", fontsize=9, color="#505c68", linespacing=1.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"q5_{prefix}_quotes.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved recorded Q5 quotation figure to {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report/figures")
    args = parser.parse_args()
    plot(args.prefix, args.output_dir)


if __name__ == "__main__":
    main()
