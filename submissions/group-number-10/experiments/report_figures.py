"""Figures for the write-up, built from saved results only (no training, no GPU).

    python -m experiments.report_figures

Reads results/metrics, results/format_stress, results/distilbert_experiments.csv and the submission
files in results/submissions/morning. Public leaderboard scores are typed in below (Kaggle doesn't
give them back as a file). Writes PNGs to results/figures/.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.genres import GENRES, genre  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
SUBS = RESULTS_DIR / "submissions" / "morning"

# Validated categorical order (first three slots pass all-pairs CVD checks), plus a neutral.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NOISE = 0.002  # public leaderboard noise (900 texts)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "font.size": 9, "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 7,
})

LARGE = "deberta-v3-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed42"
LARGE_AUG = "deberta-v3-large_nl_wu0.1_aug0.5_lr1e-05_ep3_bs16_len512_seed42"
LARGE_AUG8 = "deberta-v3-large_nl_wu0.1_aug0.5_lr1e-05_ep8_bs16_len512_seed42"
LARGE_NONL = "deberta-v3-large_wu0.1_lr1e-05_ep3_bs16_len512_seed42"

LAYOUT_NAMES = {"none": "original", "unwrap": "join wrapped\nlines", "hard_wrap": "re-wrap at\n70 chars",
                "no_newlines": "remove all\nline breaks", "short_lines": "poem-like\nshort lines"}


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG_DIR / name)


def history(run: str) -> list[dict]:
    return json.loads((RESULTS_DIR / "metrics" / f"{run}.json").read_text())["history"]


def stress(run: str) -> dict:
    return json.loads((RESULTS_DIR / "format_stress" / f"{run}.json").read_text())["results"]


def experiments() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "distilbert_experiments.csv").set_index("run_name")


def fig_kaggle_progression() -> None:
    steps = [("DistilBERT", 0.89102), ("+ [NL]\nmarker", 0.94913), ("DeBERTa-v3-\nlarge (8k rows)", 0.97869),
             ("all 10k rows\n+ 2 seeds", 0.98524), ("+ layout\naugmentation", 0.99037), ("4 seeds\n(file 15)", 0.99166)]
    x = np.arange(len(steps))
    y = [s for _, s in steps]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0.94806, color=GRAY, ls="--", lw=1.2)
    ax.text(x[-1] + 0.3, 0.94806, "teammate's stacked\nlogistic regression 0.948", color=INK2, va="center", fontsize=8)
    ax.plot(x, y, color=BLUE, marker="o", markeredgecolor="white", markeredgewidth=1.5)
    for xi, (_, s) in zip(x, steps):
        ax.annotate(f"{s:.3f}", (xi, s), textcoords="offset points", xytext=(10, -12) if xi == 0 else (0, -16), ha="left" if xi == 0 else "center",
                    color=INK, fontsize=8)
    ax.set_xticks(x, [n for n, _ in steps])
    ax.set_xlim(-0.4, x[-1] + 1.6)
    ax.set_ylim(0.88, 1.0)
    ax.set_ylabel("Kaggle public AUROC")
    ax.set_title("Each step that moved the public leaderboard")
    save(fig, "fig1_kaggle_progression.png")


def fig_kaggle_deltas() -> None:
    rows = [  # change, delta on the public board, files compared
        ("[NL] newline marker (DistilBERT)", 0.94913 - 0.89102),
        ("DeBERTa-v3-large instead of DistilBERT", 0.97869 - 0.94913),
        ("all 10k rows + 2 seeds (1 vs 5)", 0.98524 - 0.97869),
        ("layout augmentation 0.5 (5 vs 13)", 0.99037 - 0.98524),
        ("4 seeds instead of 2, augmented (13 vs 15)", 0.99166 - 0.99037),
        ("4 epochs instead of 3 (13 vs 16)", 0.99061 - 0.99037),
        ("test-time augmentation (15 vs 17)", 0.99158 - 0.99166),
        ("augmentation 0.8 instead of 0.5 (13 vs 14)", 0.99018 - 0.99037),
        ("4 seeds instead of 2, plain (5 vs 8)", 0.98460 - 0.98524),
        ("pseudo-labels from unlabeled.csv (5 vs 12)", 0.98427 - 0.98524),
        ("blend in RoBERTa-large (8 vs 9)", 0.98311 - 0.98460),
    ]
    lim = 0.009
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-NOISE, NOISE, color=GRID, alpha=0.6, lw=0)
    ax.text(0, y[0] + 0.9, f"noise (±{NOISE})", ha="center", color=INK2, fontsize=8)
    ax.axvline(0, color=INK2, lw=0.8)
    for yi, (name, d) in zip(y, rows):
        color = BLUE if d > NOISE else (ORANGE if d < -NOISE else GRAY)
        shown = min(d, lim)
        ax.plot([0, shown], [yi, yi], color=color, lw=2, solid_capstyle="round")
        ax.plot(shown, yi, "o", color=color, markeredgecolor="white", markeredgewidth=1.2)
        if d > lim:
            ax.annotate(f"+{d:.3f} →", (lim, yi), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8)
        else:
            ax.annotate(f"{d:+.4f}", (shown, yi), textcoords="offset points", xytext=(8 if d >= 0 else -8, 0),
                        ha="left" if d >= 0 else "right", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [n for n, _ in rows])
    ax.set_xlim(-0.004, lim + 0.004)
    ax.set_ylim(y[-1] - 0.7, y[0] + 1.5)
    ax.set_xlabel("change in Kaggle public AUROC (file numbers in brackets)")
    ax.set_title("What moved the leaderboard, and what tied")
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_kaggle_deltas.png")


def fig_val_vs_kaggle() -> None:
    # Validation: fixed 8k/2k split, seed 42. Kaggle: public board for the same recipe.
    stages = [("DistilBERT", 0.9891, 0.89102), ("DistilBERT\n+ [NL]", 0.998891, 0.94913),
              ("DeBERTa-v3-large\n+ [NL]", 0.999816, 0.97869), ("DeBERTa-v3-large\n+ [NL] + layout aug*", 0.999422, 0.99037)]
    x = np.arange(len(stages))
    val = np.array([1 - v for _, v, _ in stages])
    kag = np.array([1 - k for _, _, k in stages])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for xi, v, k in zip(x, val, kag):
        ax.plot([xi, xi], [v, k], color=GRID, lw=3, zorder=1)
        ax.annotate(f"×{k / v:.0f}", (xi, np.sqrt(v * k)), textcoords="offset points", xytext=(8, 0), va="center",
                    color=INK2, fontsize=8)
    ax.plot(x, kag, "o", color=ORANGE, label="Kaggle public", markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    ax.plot(x, val, "o", color=BLUE, label="our validation (2,000 texts)", markeredgecolor="white",
            markeredgewidth=1.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x, [n for n, _, _ in stages])
    ax.set_xlim(-0.5, x[-1] + 0.6)
    ax.set_ylabel("error = 1 − AUROC (log scale, lower is better)")
    ax.set_title("Validation saturated long before the leaderboard did")
    ax.legend(loc="upper right")
    ax.text(0, -0.24, "* Kaggle number for the last model is the full-data 2-seed version (file 13); the other three use "
            "the same recipe as validation.", transform=ax.transAxes, fontsize=7.5, color=INK2)
    save(fig, "fig3_val_vs_kaggle_gap.png")


def fig_stress() -> None:
    models = [("DeBERTa-v3-large, no [NL] (line breaks invisible)", LARGE_NONL, GRAY, "s"),
              ("+ [NL] (2 seeds)", LARGE, ORANGE, "o"),
              ("+ [NL] + layout augmentation", LARGE_AUG, BLUE, "D")]
    layouts = list(LAYOUT_NAMES)
    x = np.arange(len(layouts))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (name, run, color, marker) in enumerate(models):
        r = stress(run)
        vals = [r[lay]["auroc"] for lay in layouts]
        ax.plot(x + (i - 1) * 0.18, vals, marker, color=color, label=name, markeredgecolor="white",
                markeredgewidth=1.2, markersize=8)
        if run == LARGE:
            for xi, v in zip(x, vals):
                if v < 0.99:
                    ax.annotate(f"{v:.3f}", (xi - 0.18, v), textcoords="offset points", xytext=(-8, 0), ha="right",
                                va="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [LAYOUT_NAMES[lay] for lay in layouts])
    ax.set_ylim(0.85, 1.003)
    ax.set_ylabel("validation AUROC")
    ax.set_title("Line-break stress test: same words, only the line breaks changed")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_stress_test.png")


def fig_short_lines_shift() -> None:
    panels = [("Without augmentation (2 seeds)", LARGE), ("With layout augmentation", LARGE_AUG)]
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (title, run) in zip(axes, panels):
        df = pd.read_csv(RESULTS_DIR / "format_stress" / f"{run}.csv")
        ai = df[df.label == 1]
        for col, color, name in [("p_none", BLUE, "original layout"), ("p_short_lines", ORANGE, "poem-like short lines")]:
            ax.hist(ai[col], bins=bins, histtype="step", lw=2, color=color, label=name)
        called = (ai.p_short_lines < 0.5).sum()
        ax.set_title(title)
        ax.text(0.22, 0.97, f"{called} of {len(ai)} AI texts called human\nonce re-broken into short lines",
                transform=ax.transAxes, fontsize=8, color=INK2, va="top")
        ax.set_xlabel("predicted P(AI) for the 1,000 AI-written validation texts")
        ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("texts (log scale)")
    axes[1].legend(loc="center left")
    save(fig, "fig5_poem_layout_shift.png")


def fig_learning_curve() -> None:
    exp = experiments()
    series = [("DeBERTa-v3-large + [NL]", LARGE, BLUE, "o", [500, 1000, 2000, 4000, 5000, 8000]),
              ("DistilBERT + [NL]", "distilbert_nl_lr3e-05_ep4_bs16_len512_seed42", ORANGE, "s",
               [1000, 2000, 4000, 8000])]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, run, color, marker, sizes in series:
        vals = [exp.loc[run if n == 8000 else f"{run}_n{n}", "val_auc"] for n in sizes]
        ax.plot(sizes, vals, marker=marker, color=color, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(name, (sizes[-1], vals[-1]), textcoords="offset points", xytext=(8, 0), va="center",
                    color=INK, fontsize=8)
    ax.set_xscale("log", base=2)
    ticks = [500, 1000, 2000, 4000, 8000]
    ax.set_xticks(ticks, ["0.5k", "1k", "2k", "4k", "8k"])
    ax.set_xlim(400, 30000)
    ax.set_xlabel("training rows (nested, class-balanced subsets; validation fixed at 2,000)")
    ax.set_ylabel("validation AUROC")
    ax.set_title("Effect of training-set size")
    save(fig, "fig6_learning_curve.png")


def fig_train_val_curve() -> None:
    """Q2.2: training and validation AUROC vs. training-set size, DeBERTa-v3-large + [NL]."""
    exp = experiments()
    sizes = [500, 1000, 2000, 4000, 5000, 8000]
    rows = [exp.loc[LARGE if n == 8000 else f"{LARGE}_n{n}"] for n in sizes]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for col, name, color, marker, dy in [("train_auc", "training", ORANGE, "s", 6),
                                         ("val_auc", "validation", BLUE, "o", -6)]:
        vals = [r[col] for r in rows]
        ax.plot(sizes, vals, marker=marker, color=color, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(name, (sizes[-1], vals[-1]), textcoords="offset points", xytext=(8, dy), va="center",
                    color=INK, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, ["0.5k", "1k", "2k", "4k", "5k", "8k"])
    ax.set_xlim(400, 16000)
    ax.set_ylim(0.995, 1.0005)
    ax.set_xlabel("training rows (class-balanced subsets of the 8k split; validation fixed at 2,000)")
    ax.set_ylabel("AUROC")
    ax.set_title("DeBERTa-v3-large + [NL]: training vs. validation AUROC")
    save(fig, "fig10_learning_curve_train_val.png")


def fig_backbones() -> None:
    exp = experiments()
    rows = [  # name, params, with [NL], without [NL]
        ("DistilBERT (66M)", "distilbert_nl_lr3e-05_ep4_bs16_len512_seed42", "distilbert_lr3e-05_ep4_bs16_len512_seed42"),
        ("DeBERTa-v3-base (184M)", "deberta-v3-base_nl_wu0.1_lr2e-05_ep3_bs16_len512_seed42", None),
        ("ELECTRA-large (335M)", "electra-large-discriminator_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed42", None),
        ("RoBERTa-large (355M)", "roberta-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed42",
         "roberta-large_wu0.1_lr1e-05_ep3_bs16_len512_seed42"),
        ("DeBERTa-v3-large (435M)", LARGE, LARGE_NONL),
        ("DeBERTa-v2-xlarge (900M, bf16)", "deberta-v2-xlarge_nl_wu0.1_bf16_lr5e-06_ep3_bs16_len512_seed42", None),
    ]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for yi, (name, nl, plain) in zip(y, rows):
        v = exp.loc[nl, "val_auc"]
        if plain:
            p = exp.loc[plain, "val_auc"]
            ax.plot([p, v], [yi, yi], color=GRID, lw=3, zorder=1)
            ax.plot(p, yi, "o", mfc="white", mec=BLUE, mew=2, zorder=3)
        ax.plot(v, yi, "o", color=BLUE, mec="white", mew=1.2, zorder=3)
        ax.annotate(f"{v:.4f}", (v, yi), textcoords="offset points", xytext=(8, 0), va="center", fontsize=8,
                    color=INK2)
    ax.plot([], [], "o", color=BLUE, label="with [NL] marker")
    ax.plot([], [], "o", mfc="white", mec=BLUE, mew=2, label="without [NL]")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0.990, 1.0008)
    ax.set_xlabel("validation AUROC (8k/2k split, seed 42)")
    ax.set_title("Backbones, with and without the newline marker")
    ax.legend(loc="lower left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig7_backbones.png")


def fig_genres() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    tr = train.text.map(genre)
    te = test.text.map(genre)
    plain = pd.read_csv(SUBS / "8_deberta_large_full10k_4seed.csv").set_index("id").label.reindex(test.id).to_numpy()
    aug = pd.read_csv(SUBS / "15_deberta_large_full10k_layoutaug_4seed.csv").set_index("id").label.reindex(test.id).to_numpy()
    order = ["poetry", "recipe", "abstract", "other"]
    assert set(order) == set(GENRES)
    names = {"poetry": "poetry", "recipe": "recipes", "abstract": "abstracts", "other": "other prose"}
    y = np.arange(len(order))[::-1]
    h = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    share_tr = [100 * (tr == g).mean() for g in order]
    share_te = [100 * (te == g).mean() for g in order]
    unsure = lambda p, g: 100 * ((p > 0.05) & (p < 0.95))[(te == g).to_numpy()].mean()  # noqa: E731
    u_plain = [unsure(plain, g) for g in order]
    u_aug = [unsure(aug, g) for g in order]
    for ax, (a, b, la, lb, title, xl) in zip(axes, [
        (share_tr, share_te, "train.csv", "test.csv", "Genre mix", "% of texts"),
        (u_plain, u_aug, "without augmentation (file 8)", "with layout augmentation (file 15)",
         "Unsure test predictions", "% of test texts with 0.05 < P(AI) < 0.95"),
    ]):
        ax.barh(y + h / 2, a, height=h - 0.04, color=ORANGE if ax is axes[1] else GRAY, label=la)
        ax.barh(y - h / 2, b, height=h - 0.04, color=BLUE, label=lb)
        for yi, va, vb in zip(y, a, b):
            ax.text(va + 0.5, yi + h / 2, f"{va:.1f}%", va="center", fontsize=7.5, color=INK2)
            ax.text(vb + 0.5, yi - h / 2, f"{vb:.1f}%", va="center", fontsize=7.5, color=INK2)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.legend(loc="center right", fontsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [names[g] for g in order])
    axes[0].set_xlim(0, 75)
    axes[1].set_xlim(0, max(u_plain + u_aug) * 1.3)
    fig.text(0.01, -0.04, "Genres come from a keyword/line-format detector (src/genres.py); 'poetry' is reliable on test but "
             "over-counts short-line prose in train.", fontsize=7.5, color=INK2)
    save(fig, "fig8_genre_shift.png")
    print("genre shares train/test:", dict(zip(order, np.round(share_tr, 1))), dict(zip(order, np.round(share_te, 1))))
    print("unsure % plain/aug:", dict(zip(order, np.round(u_plain, 1))), dict(zip(order, np.round(u_aug, 1))))


def fig_epochs() -> None:
    runs = [("[NL] + aug, 8-epoch schedule", LARGE_AUG8, BLUE, "o"), ("[NL] + aug, 3-epoch schedule", LARGE_AUG, AQUA, "D"),
            ("[NL], no aug", LARGE, ORANGE, "s"), ("no [NL]", LARGE_NONL, GRAY, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, run, color, marker in runs:
        hist = history(run)
        ep = [r["epoch"] for r in hist]
        axes[0].plot(ep, [r["val_auc"] for r in hist], marker=marker, color=color, label=name, mec="white", mew=1.2)
        if run in (LARGE_AUG8, LARGE_AUG):
            axes[1].plot(ep, [r["val_loss"] for r in hist], marker=marker, color=color, label=f"{name}: validation",
                         mec="white", mew=1.2)
            if run == LARGE_AUG8:
                axes[1].plot(ep, [r["train_loss"] for r in hist], ls="--", color=color, lw=1.5,
                             label=f"{name}: training")
    axes[0].set(title="Validation AUROC per epoch", xlabel="epoch", ylim=(0.993, 1.0005), ylabel="validation AUROC")
    axes[1].set(title="Loss per epoch", xlabel="epoch", yscale="log", ylabel="cross-entropy (log scale)")
    axes[0].legend(loc="lower right", fontsize=7.5)
    axes[1].legend(loc="lower left", fontsize=7.5)
    fig.text(0.01, -0.04, "DeBERTa-v3-large, 8k/2k split, seed 42, lr 1e-5 with linear decay to 0 at the last epoch "
             "(so the 8-epoch run decays more slowly).", fontsize=7.5, color=INK2)
    save(fig, "fig9_epochs.png")


def main() -> None:
    fig_kaggle_progression()
    fig_kaggle_deltas()
    fig_val_vs_kaggle()
    fig_stress()
    fig_short_lines_shift()
    fig_learning_curve()
    fig_train_val_curve()
    fig_backbones()
    fig_genres()
    fig_epochs()


if __name__ == "__main__":
    main()
