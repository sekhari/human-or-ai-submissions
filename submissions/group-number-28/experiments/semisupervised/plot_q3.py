import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from human_ai.data import ROOT

BLUE, INK, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()
    study = ROOT / "results" / "semisupervised" / f"{args.prefix}_study"
    rows = [r for r in csv.DictReader(open(study / "comparison.csv"))
            if r["run"].startswith("st") and r["round"] == "1"]
    rows.sort(key=lambda r: float(r["threshold"]))
    noise = json.load(open(study / "teacher_noise_on_validation.json"))
    noise.sort(key=lambda n: n["threshold"])

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)

    t = [float(r["threshold"]) for r in rows]
    d = [float(r["diff_vs_baseline"]) for r in rows]
    lo = [float(r["diff_vs_baseline"]) - float(r["ci_low"]) for r in rows]
    hi = [float(r["ci_high"]) - float(r["diff_vs_baseline"]) for r in rows]
    a.axhline(0, color=MUTED, linewidth=1)
    a.errorbar(t, d, yerr=[lo, hi], fmt="o", color=BLUE, ms=8, capsize=4,
               elinewidth=2, markeredgecolor=SURFACE, markeredgewidth=2)
    a.set_xscale("function", functions=(lambda v: -__import__("numpy").log(1.0001 - v),
                                         lambda v: 1.0001 - __import__("numpy").exp(-v)))
    a.set_xticks(t, [f"{x:g}\n+{r['pseudo_added']}" for x, r in zip(t, rows)])
    a.set_xlabel("Confidence threshold / pseudo-labeled rows added (one round)")
    a.set_ylabel("Validation AUROC minus baseline")
    a.set_title("Self-training gain, 95% bootstrap CI", loc="left", color=INK, fontsize=11)

    b.plot([n["threshold"] for n in noise], [100 * n["error_rate"] for n in noise],
           color=BLUE, linewidth=2, marker="o", ms=8, markeredgecolor=SURFACE,
           markeredgewidth=2)
    for n in noise:
        b.annotate(f"{100 * n['selected_share']:.0f}% kept", (n["threshold"], 100 * n["error_rate"]),
                   textcoords="offset points", xytext=(6, 6), fontsize=8, color=MUTED)
    b.set_xlabel("Confidence threshold")
    b.set_ylabel("Teacher error rate on selected rows (%)")
    b.set_title("Pseudo-label noise, measured on validation", loc="left", color=INK, fontsize=11)

    fig.tight_layout()
    out = ROOT / "report" / "figures" / f"q3_{args.prefix}_selftraining.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
