"""Head-to-head table of the six submitted models (3 pretrained blends + 3 classical models) on the
shared fixed split, plus every other model either track validated, from the tracks' own results files.

  python common/compare_tracks.py

Reads pretrained/results/fixed_split/models.csv, classical/results/*/metrics.json and submission_log.csv
(Kaggle scores); writes common/results/compare_tracks.csv, common/tables/six_models.tex,
common/tables/all_models.tex, common/figures/six_models.pdf and common/figures/cost_vs_auroc.pdf.
Both reports \\input the tables. Training time is the fixed-split fit time; for pretrained LM-feature
models that excludes LM feature extraction (cached), which the cost plot adds back from the original
5-fold runs (cv5_run_time_s_seed42 / 5, an approximate per-fit cost).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEAM = Path(__file__).resolve().parent.parent
OUT = TEAM / "common"
SIX = [("pretrained", "final_A_topic_greedy", "Blend A"), ("pretrained", "final_B_lm4", "Blend B"),
       ("pretrained", "final_C_lm4_distil", "Blend C"), ("classical", "tfidf_lr", "C1 TF-IDF + LR"),
       ("classical", "stylo_gbm", "C2 stylometric GBM"), ("classical", "transformer_scratch", "C3 Transformer (scratch)")]
# classical results are named by model, their Kaggle uploads final_<letter>_<model> (the current version's file)
UPLOAD = {"tfidf_lr": "final_E_tfidf_lr", "stylo_gbm": "final_D_stylo_gbm",
          "transformer_scratch": "final_F_transformer_scratch"}
REPR = {
    "final_A_topic_greedy": "7-model CDF blend: 3 frozen-LM + style heads, GPT-2 stats-only head, "
                            "3 fine-tuned encoders; weights 4,2,2,1,1,1,1 by greedy selection on topic-held-out CV",
    "final_B_lm4": "4 frozen-LM token-statistics + style heads (GPT-2, GPT-2-medium, 2 Qwen2.5 Binoculars), equal weights",
    "final_C_lm4_distil": "blend B + fine-tuned DistilRoBERTa, equal weights",
    "tfidf_lr": "word 1--2 + raw char 1--5 + distorted-text word 1--3 TF-IDF $\\to$ logistic regression",
    "stylo_gbm": "101 hand-crafted stylometric features $\\to$ LightGBM",
    "transformer_scratch": "single line breaks joined, 12k byte-level BPE $\\to$ 4-layer Transformer, random init",
}
BLEND_HW = "RTX 5070 + RTX 3060"  # pretrained members ran on both GPUs (see results/fixed_split/models.csv)


def short_hw(h: str) -> str:
    h = str(h)
    return "CPU" if h.startswith("CPU") else h.replace("NVIDIA GeForce ", "")


def load() -> pd.DataFrame:
    log = pd.read_csv(TEAM / "submission_log.csv", dtype=str, keep_default_na=False)
    kag = {(r.track, r.model): r.kaggle_public for _, r in log.iterrows()}
    rows = []
    p = pd.read_csv(TEAM / "pretrained" / "results" / "fixed_split" / "models.csv")
    singles = p[p.kind == "single"].set_index("model")
    for _, r in p.iterrows():
        approx = (r.fit_time_s if pd.isna(r.cv5_run_time_s_seed42) or r.kind == "blend"
                  else max(float(r.fit_time_s), float(r.cv5_run_time_s_seed42) / 5))
        if r.kind == "blend":  # members' costs, with LM features added back the same way
            members = [m.rsplit("x", 1)[0] for m in r.run.split(" + ")]
            approx = sum(max(float(singles.loc[m, "fit_time_s"]), float(singles.loc[m, "cv5_run_time_s_seed42"]) / 5)
                         for m in members)
        rows.append({"track": "pretrained", "model": r.model, "train_auroc": r.train_auroc, "val_auroc": r.val_auroc,
                     "val_tpr_at_1fpr": r.val_tpr_at_1fpr, "fit_time_s": r.fit_time_s, "approx_full_cost_s": approx,
                     "hardware": BLEND_HW if r.kind == "blend" else short_hw(r.hardware),
                     "kaggle_public": kag.get(("pretrained", r.model), "")})
    for d in sorted((TEAM / "classical" / "results").iterdir()):
        f = d / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"track": "classical", "model": m["model"], "train_auroc": m["train_auroc"], "val_auroc": m["val_auroc"],
                     "val_tpr_at_1fpr": m["val_tpr_at_1fpr"], "fit_time_s": m["fit_time_s"],
                     "approx_full_cost_s": m["fit_time_s"], "hardware": short_hw(m["hardware"]),
                     "kaggle_public": kag.get(("classical", UPLOAD.get(m["model"], m["model"])), "")})
    return pd.DataFrame(rows)


def secs(x) -> str:
    x = float(x)
    return f"{x:.0f} s" if x < 120 else f"{x / 60:.1f} min"


def kag(x) -> str:
    try:
        return f"{float(x):.5f}"
    except ValueError:
        return x or "--"


def tex(s) -> str:
    return str(s).replace("_", "\\_").replace("&", "\\&")


def main():
    df = load()
    (OUT / "results").mkdir(exist_ok=True)
    df.to_csv(OUT / "results" / "compare_tracks.csv", index=False, lineterminator="\n", float_format="%.6g")
    idx = df.set_index(["track", "model"])
    six = []
    for track, model, label in SIX:
        r = idx.loc[(track, model)]
        six.append(f"{label} & {track} & {r.train_auroc:.5f} & {r.val_auroc:.5f} & {secs(r.fit_time_s)} & "
                   f"{r.hardware} & {kag(r.kaggle_public)} \\\\")
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "tables" / "six_models.tex").write_text("\n".join(six) + "\n", encoding="utf-8")
    # the template's Q1.2 columns (model, representation, train, val, time) + Kaggle, for the combined report
    q12 = [f"{label} & {REPR[model]} & {idx.loc[(track, model)].train_auroc:.5f} & {idx.loc[(track, model)].val_auroc:.5f} & "
           f"{secs(idx.loc[(track, model)].fit_time_s)} & {kag(idx.loc[(track, model)].kaggle_public)} \\\\"
           for track, model, label in SIX]
    q12.insert(3, "\\midrule")
    (OUT / "tables" / "six_models_q12.tex").write_text("\n".join(q12) + "\n", encoding="utf-8")
    others = df[~df.set_index(["track", "model"]).index.isin([(t, m) for t, m, _ in SIX])]
    (OUT / "tables" / "all_models.tex").write_text("\n".join(
        f"\\texttt{{{tex(r.model)}}} & {r.track} & {r.train_auroc:.5f} & {r.val_auroc:.5f} & {secs(r.fit_time_s)} & "
        f"{r.hardware} \\\\" for _, r in others.sort_values("val_auroc", ascending=False).iterrows()) + "\n",
        encoding="utf-8")

    (OUT / "figures").mkdir(exist_ok=True)
    colors = {"pretrained": "#236B82", "classical": "#C0763A"}
    s = pd.DataFrame([{**idx.loc[(t, m)].to_dict(), "label": lab, "track": t} for t, m, lab in SIX])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.barh(s.label[::-1], s.val_auroc[::-1], color=[colors[t] for t in s.track[::-1]])
    for y, v in enumerate(s.val_auroc[::-1]):
        ax.text(v + 0.0005, y, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlim(0.95, 1.004)
    ax.set_xlabel("validation AUROC (same fixed 2,000 rows)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "six_models.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for t, g in df.groupby("track"):
        ax.scatter(g.approx_full_cost_s, g.val_auroc, c=colors[t], label=t, s=22)
    for _, r in s.iterrows():
        ax.annotate(r.label, (r.approx_full_cost_s, r.val_auroc), fontsize=7, xytext=(4, -3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("approx. cost of one training run, seconds (log; LM feature extraction included)")
    ax.set_ylabel("validation AUROC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "cost_vs_auroc.pdf")
    plt.close(fig)
    print(open(OUT / "tables" / "six_models.tex", encoding="utf-8").read())


if __name__ == "__main__":
    main()
