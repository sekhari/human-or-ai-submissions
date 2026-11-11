"""Turn results/*.csv into the LaTeX table rows and figures that reports/kaggle_pretrained_report.tex inputs.

  python scripts/report_tables.py

Writes reports/tables/*.tex (table bodies only, \\input inside the template's tabularx) and
reports/figures/*.pdf. Every number comes from a results file; nothing is typed by hand.
FP/FN hypotheses are the one hand-written input: reports/fp_fn_notes.yaml (id -> note), written after
reading each passage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEAM = ROOT.parent
RES = ROOT / "results"
TAB = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"

REPR = {
    "tfidf_logreg": "word 1--2 + char\\_wb 3--5 TF-IDF $\\to$ LR",
    "stylometric_gbm_ext": "92 stylometric features $\\to$ LightGBM",
    "lm_gpt2": "GPT-2 token log-prob/rank/entropy stats + style $\\to$ LightGBM",
    "lm_gpt2_nostyle": "GPT-2 token stats only $\\to$ LightGBM",
    "lm_gpt2_medium": "GPT-2-medium token stats + style $\\to$ LightGBM",
    "binoculars_qwen05b": "Qwen2.5-0.5B Binoculars + token stats + style $\\to$ LightGBM",
    "binoculars_qwen15b": "Qwen2.5-1.5B Binoculars + token stats + style $\\to$ LightGBM",
    "ft_roberta_base": "fine-tuned RoBERTa-base (512 tokens)",
    "ft_deberta_v3_large": "fine-tuned DeBERTa-v3-large (512 tokens)",
    "ft_openai_detector_large": "fine-tuned OpenAI GPT-2-output detector (RoBERTa-large)",
    "ft_distilroberta": "fine-tuned DistilRoBERTa (512 tokens)",
    "ft_modernbert_base": "fine-tuned ModernBERT-base (512 tokens)",
    "final_A_topic_greedy": "blend A: 7 models, weights 4,2,2,1,1,1,1 by greedy selection on topic-held-out CV",
    "final_B_lm4": "blend B: 4 LM-feature models, equal weights",
    "final_C_lm4_distil": "blend C: B + DistilRoBERTa, equal weights",
}
NAME = {"final_A_topic_greedy": "Blend A", "final_B_lm4": "Blend B", "final_C_lm4_distil": "Blend C"}


def tex(s) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def f4(x) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) or x == "" else f"{float(x):.4f}"


def f5(x) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) or x == "" else f"{float(x):.5f}"


def secs(x) -> str:
    x = float(x)
    return f"{x:.0f} s" if x < 120 else f"{x / 60:.1f} min"


def write(name: str, rows: list[str]) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote reports/tables/{name}")


def q12():
    df = pd.read_csv(RES / "fixed_split" / "models.csv")
    rows = []
    for _, r in df.iterrows():
        label = NAME.get(r.model, f"\\texttt{{{tex(r.model)}}}")
        rows.append(f"{label} & {REPR[r.model]} & {f5(r.train_auroc)} & {f5(r.val_auroc)} & {secs(r.fit_time_s)} \\\\")
        if r.model == "ft_openai_detector_large":
            rows.append("\\midrule")
    write("q12_models.tex", rows)
    # the 5-fold CV (seed 42, research repo) vs fixed split comparison, singles only
    s = df[df.kind == "single"]
    write("q12_cv_vs_fixed.tex", [f"\\texttt{{{tex(r.model)}}} & {f5(r.cv5_auroc_seed42)} & {f5(r.val_auroc)} & "
                                  f"{float(r.val_auroc) - float(r.cv5_auroc_seed42):+.4f} \\\\" for _, r in s.iterrows()])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = df.sort_values("val_auroc")
    ax.barh([NAME.get(m, m) for m in order.model], order.val_auroc, color=["#236B82" if k == "blend" else "#9AB7C1"
                                                                           for k in order.kind])
    ax.set_xlim(max(0.9, order.val_auroc.min() - 0.01), 1.0)
    ax.set_xlabel("validation AUROC (fixed 2,000 rows)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "val_auroc_bar.pdf")
    plt.close(fig)


def q13():
    rows = []
    for model, cols in [("tfidf_logreg", ["C", "ngram_range"]), ("lm_gpt2", ["lgbm"])]:
        p = RES / "hpsearch" / f"{model}.csv"
        if not p.exists():
            rows.append(f"\\texttt{{{tex(model)}}} & [TODO: not run] & [TODO] \\\\")
            continue
        d = pd.read_csv(p)
        best = d[d.chosen].iloc[0]
        if model == "tfidf_logreg":
            grid = (f"C $\\in$ \\{{{', '.join(f'{c:g}' for c in sorted(d.C.unique()))}\\}} $\\times$ word n-grams "
                    f"\\{{(1,1), (1,2)\\}}; {len(d)} settings")
            chosen = f"C = {best.C:g}, word {tex(best.ngram_range)}; inner CV {best.inner_cv_auroc:.4f}"
        else:
            g = d.lgbm.map(yaml.safe_load)
            vals = lambda k: ", ".join(str(v) for v in sorted({x[k] for x in g}))  # noqa: E731
            grid = (f"num\\_leaves \\{{{vals('num_leaves')}\\}} $\\times$ learning\\_rate \\{{{vals('learning_rate')}\\}} "
                    f"$\\times$ n\\_estimators \\{{{vals('n_estimators')}\\}}; {len(d)} settings")
            b = yaml.safe_load(best.lgbm)
            # The grid was run after the blends were uploaded. The submitted LM heads use the class default
            # (src/models/lm_perplexity.py: 15 leaves, lr 0.03, 500 trees), so report both settings.
            dflt = d[g.map(lambda x: (x["num_leaves"], x["learning_rate"], x["n_estimators"]) == (15, 0.03, 500))].iloc[0]
            chosen = (f"best in grid: {b['num_leaves']} leaves, lr {b['learning_rate']}, {b['n_estimators']} trees "
                      f"(inner CV {best.inner_cv_auroc:.5f}); the submitted blends keep the default 15 leaves, "
                      f"lr 0.03, 500 trees ({dflt.inner_cv_auroc:.5f})")
        rows.append(f"\\texttt{{{tex(model)}}} & {grid} & {chosen} \\\\")
    write("q13_hpsearch.tex", rows)


def q2():
    d = pd.read_csv(RES / "learning_curve.csv")
    main = d[d.model == "lm_gpt2"]
    write("q21_curve.tex", [f"{int(r.train_rows):,} & {f4(r.train_auroc)} & {f4(r.val_auroc)} & "
                            f"TF-IDF LR val {f4(d[(d.model == 'tfidf_logreg') & (d.train_rows == r.train_rows)].val_auroc.iloc[0])} \\\\"
                            for _, r in main.iterrows()])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for model, c in [("lm_gpt2", "#236B82"), ("tfidf_logreg", "#C0763A")]:
        s = d[d.model == model]
        ax.plot(s.train_rows, s.train_auroc, "--o", color=c, ms=4, label=f"{model} train")
        ax.plot(s.train_rows, s.val_auroc, "-o", color=c, ms=4, label=f"{model} validation")
    ax.set_xscale("log")
    ax.set_xticks([500, 1000, 2000, 5000, 8000], ["500", "1k", "2k", "5k", "8k"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("training rows (nested subsets of the fixed 8,000)")
    ax.set_ylabel("AUROC")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "learning_curve.pdf")
    plt.close(fig)


def q3():
    d = pd.read_csv(RES / "semi_supervised.csv")
    label = {"supervised": "Supervised only", "pseudo_label_unlabeled_r1": "Pseudo-labels, 1 round",
             "pseudo_label_unlabeled_r3": "Pseudo-labels, 3 rounds"}
    write("q32_semisup.tex", [f"{label[r.method]} (\\texttt{{{tex(r.model)}}}) & {f5(r.val_auroc)} \\\\"
                              for _, r in d.iterrows()])


def q4():
    s = pd.read_csv(RES / "error_analysis" / "subgroups.csv")
    write("q42_subgroups.tex", [f"{tex(r.group).replace('<', '$<$').replace('>', '$>$')} ({r.grouping}; {int(r.n_ai)} AI) & {int(r['count'])} & {f5(r.auroc)} \\\\"
                                for _, r in s.iterrows()])
    e = pd.read_csv(RES / "error_analysis" / "fp_fn.csv")
    e = e.sort_values("error", key=lambda c: c.map({"FP": 0, "FP (near miss)": 1, "FN": 2, "FN (near miss)": 3}),
                      kind="stable")
    notes_file = ROOT / "reports" / "fp_fn_notes.yaml"
    notes = yaml.safe_load(notes_file.read_text(encoding="utf-8")) if notes_file.exists() else {}
    write("q43_errors.tex", [f"\\mbox{{{int(r.id)}}} & {r.error} & {r.p_ai:.3f} & "
                             f"{tex(notes.get(int(r.id), '[TODO: hypothesis]'))} \\\\" for _, r in e.iterrows()])


def q5():
    """Formatting-cue ablation on the fixed split (validation AUROC only)."""
    deb = RES / "debias_fixed_split.csv"
    if deb.exists():
        d = pd.read_csv(deb)
        write("q53_debias.tex", [f"\\texttt{{{tex(r.model)}}}, {tex(r.variant)} & {f4(r.val_auroc)} \\\\"
                                 for _, r in d.iterrows()])


BREAK = chr(92) + 'allowbreak{}'  # long model names may wrap at underscores


def appendix():
    log = pd.read_csv(TEAM / "submission_log.csv", dtype=str, keep_default_na=False).sort_values(["date", "track", "model"])
    url = "https://github.com/ZhenmanShen/human-or-ai-submissions/commit/"
    rows = []
    for _, r in log.iterrows():
        commit = f"\\href{{{url}{r.commit}}}{{\\texttt{{{r.commit[:7]}}}}}" if r.commit else ""
        rows.append(f"{r.date} & {r.track} & \\texttt{{{tex(r.model).replace('_', '_' + BREAK)}}} & {commit} & {f5(r.val_auroc) if r.val_auroc else '--'} & "
                    f"{tex(r.kaggle_public)} & {tex(r.kaggle_private) or '--'} \\\\")
    write("appendix_submissions.tex", rows)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["q12", "q13", "q2", "q3", "q4", "q5", "appendix"]
    for s in steps:
        try:
            globals()[s]()
        except FileNotFoundError as e:
            print(f"skip {s}: {e}")
