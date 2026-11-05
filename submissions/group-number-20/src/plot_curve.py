"""Plot Q2 training and validation AUROC against training-set size."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
c = json.load(open("results/q2_q3.json"))["curve"]; n = [r["n"] for r in c]
plt.rcParams.update({"font.size": 9, "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
f, a = plt.subplots(figsize=(5.8, 2.8), facecolor="#fcfcfb"); a.set_facecolor("#fcfcfb")
a.plot(n, [r["train"] for r in c], color="#d97a1e", lw=2, marker="s", ms=4.5, label="Training")
a.errorbar(n, [r["val"] for r in c], yerr=[r["val_sd"] for r in c], color="#2a78d6", lw=2, marker="o", ms=4.5, capsize=2, label="Validation")
a.text(n[-1], 1.008, "Training 1.000", ha="right", color="#52514e"); a.text(n[-1], c[-1]["val"] - .024, f'Validation {c[-1]["val"]:.3f}', ha="right", color="#52514e")
a.set(xscale="log", xlabel="Training rows (log scale)", ylabel="AUROC", ylim=(.84, 1.02)); a.set_xticks(n, [f"{x:,}" for x in n]); a.minorticks_off()
a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#e6e5e0", lw=.6); a.legend(frameon=False, loc="lower right", ncols=2)
f.tight_layout(); f.savefig("results/learning_curve.png", dpi=200)
