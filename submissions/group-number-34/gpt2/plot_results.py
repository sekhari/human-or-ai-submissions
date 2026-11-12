"""Render the recorded GPT-2 results as PNG and SVG figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


BLUE = "#286A9E"
DARK_BLUE = "#174366"
GOLD = "#986400"
INK = "#202B35"
MUTED = "#57636E"
GRID = "#E1E6EA"


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#A5AFB7",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "gpt2-results",
        "savefig.dpi": 180,
    })


def save_figure(fig, output_dir, name):
    for extension in ("png", "svg"):
        metadata = {"Date": None} if extension == "svg" else None
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, facecolor="white", metadata=metadata)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(path)
    plt.close(fig)


def plot_format_sensitivity(snapshot, output_dir):
    variants = [
        ("original", "Original text"),
        ("spaces_only", "Collapse consecutive\nspaces and tabs"),
        ("newlines_only", "Replace line breaks\nwith spaces"),
        ("all_whitespace", "Normalize all whitespace"),
    ]
    metrics = {row["variant"]: row for row in snapshot["metrics"]}
    values = [metrics[key]["auc"] for key, _ in variants]
    original_auc = metrics["original"]["auc"]
    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_axes([0.29, 0.22, 0.43, 0.52])
    fig.text(0.045, 0.93, "GPT-2 formatting sensitivity", fontsize=20, weight="bold")
    fig.text(
        0.045, 0.865,
        f"Same {snapshot['validation_rows']:,} validation passages | "
        f"Selected epoch {snapshot['selected_epoch']} | No retraining",
        color=MUTED,
    )

    positions = list(range(len(variants)))
    ax.barh(positions, values, height=0.48, color=[DARK_BLUE, BLUE, BLUE, BLUE])
    ax.set_yticks(positions, [label for _, label in variants])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Validation AUROC", labelpad=10)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["left"].set_visible(False)

    ax.text(1.04, 1.10, "AUROC", transform=ax.transAxes, weight="bold", fontsize=10)
    ax.text(1.30, 1.10, "Change vs. original", transform=ax.transAxes, weight="bold", fontsize=10)
    for position, value in zip(positions, values):
        ax.text(1.04, position, f"{value:.7f}", transform=ax.get_yaxis_transform(), va="center")
        delta = value - original_auc
        label = "Reference" if position == 0 else f"{delta:+.7f}"
        ax.text(1.30, position, label, transform=ax.get_yaxis_transform(), va="center", color=MUTED)

    fig.text(0.045, 0.085, "All transformations preserve non-whitespace characters.", fontsize=10, color=MUTED)
    fig.text(0.045, 0.045, "Source: results_snapshot/format_sensitivity.json", fontsize=9, color=MUTED)
    save_figure(fig, output_dir, "format_sensitivity")


def plot_training_history(snapshot, output_dir):
    history = sorted(snapshot["history"], key=lambda row: row["epoch"])
    epochs = [row["epoch"] for row in history]
    losses = [row["training_loss"] for row in history]
    aucs = [row["validation_auc"] for row in history]
    best_epoch = snapshot["best_epoch"]
    best_row = next(row for row in history if row["epoch"] == best_epoch)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.24, top=0.72, wspace=0.43)
    fig.text(0.045, 0.93, "GPT-2 training history", fontsize=20, weight="bold")
    fig.text(
        0.045, 0.865,
        f"{snapshot['training_rows']:,} training passages | "
        f"Same {snapshot['validation_rows']:,} validation passages | "
        f"Selected epoch: {best_epoch}",
        color=MUTED,
    )

    loss_ax, auc_ax = axes
    loss_ax.set_title("Training loss", loc="left", fontsize=13, weight="bold", pad=14)
    loss_ax.plot(epochs, losses, color=BLUE, marker="o", linewidth=1.8, markersize=6)
    loss_ax.set_ylabel("Binary cross-entropy")
    loss_ax.set_ylim(0, max(losses) * 1.22)
    loss_ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for epoch, loss in zip(epochs, losses):
        offset = (12, 12) if epoch == epochs[1] else (0, 12)
        alignment = "left" if epoch == epochs[1] else "center"
        loss_ax.annotate(f"{loss:.7f}", (epoch, loss), xytext=offset, textcoords="offset points", ha=alignment, fontsize=10)

    auc_ax.set_title("Validation AUROC", loc="left", fontsize=13, weight="bold", pad=14)
    auc_ax.plot(epochs, aucs, color=BLUE, marker="o", linewidth=1.8, markersize=6)
    auc_ax.scatter([best_epoch], [best_row["validation_auc"]], color=GOLD, marker="D", s=65, zorder=3)
    auc_ax.set_ylabel("AUROC (zoomed scale)")
    padding = max(max(aucs) - min(aucs), 0.0001) * 0.35
    auc_ax.set_ylim(max(0, min(aucs) - padding), min(1, max(aucs) + padding))
    auc_ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    for epoch, auc in zip(epochs, aucs):
        auc_ax.annotate(
            f"{auc:.7f}", (epoch, auc), xytext=(0, -24) if epoch == epochs[1] else (0, 12),
            textcoords="offset points", ha="center", fontsize=10,
            color=GOLD if epoch == best_epoch else INK,
        )
    for ax in axes:
        ax.set_xticks(epochs)
        ax.set_xlim(min(epochs) - 0.35, max(epochs) + 0.35)
        ax.set_xlabel("Epoch", labelpad=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.8)

    fig.text(0.045, 0.10, "Loss is averaged within each epoch. The highest validation AUROC selects the model.", fontsize=10, color=MUTED)
    fig.text(0.045, 0.055, "Source: results_snapshot/training_metrics.json", fontsize=9, color=MUTED)
    save_figure(fig, output_dir, "training_history")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--snapshot-dir", type=Path, default=root / "results_snapshot")
    parser.add_argument("--output-dir", type=Path, default=root / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    training = json.loads((args.snapshot_dir / "training_metrics.json").read_text())
    formatting = json.loads((args.snapshot_dir / "format_sensitivity.json").read_text())
    plot_training_history(training, args.output_dir)
    plot_format_sensitivity(formatting, args.output_dir)


if __name__ == "__main__":
    main()
