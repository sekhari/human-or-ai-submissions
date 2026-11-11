"""Experiments for the pretrained-track report, all on the team's fixed 8,000 / 2,000 split.

  python scripts/report_experiments.py collect --session outputs/sessions/<fixed session>   # Q1.2 + blends
  python scripts/report_experiments.py hpsearch [--jobs 3]     # Q1.3: 5-fold CV inside the 8,000 rows
  python scripts/report_experiments.py curve                   # Q2: 500 ... 8,000 nested training rows
  python scripts/report_experiments.py semisup                 # Q3: pseudo-labelling unlabeled.csv
  python scripts/report_experiments.py errors                  # Q4: blend B (final selection) subgroups + FP/FN
  python scripts/report_experiments.py debias                  # Q5: formatting-cue ablation on the fixed split

Every subcommand writes CSVs under results/ (committed); scripts/report_tables.py turns them into the
LaTeX tables and figures in reports/. Runs go to outputs/runs/rep__* (git-ignored) and are reused when
their metrics.json exists. Seed 20261023 throughout. Nothing here reads test labels (there are none)
or tunes anything on the 2,000 validation rows: hyperparameters are chosen by inner CV, blends keep
the weights they had on Kaggle.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "common"))
from metrics import errors, length_bin_auroc, summary, verse_prose_auroc, word_count  # noqa: E402
from src.config import resolve, set_data_dir  # noqa: E402
from src.data import load_csv  # noqa: E402
from src.ensemble import rank01  # noqa: E402

SEED = 20261023
RESULTS = ROOT / "results"
RUNS = ROOT / "outputs" / "runs"
FIXED = ["cv.scheme=fixed", f"seed={SEED}"]

MODELS = ["tfidf_logreg", "stylometric_gbm_ext", "lm_gpt2", "lm_gpt2_nostyle", "lm_gpt2_medium",
          "binoculars_qwen05b", "binoculars_qwen15b", "ft_roberta_base", "ft_distilroberta", "ft_modernbert_base",
          "ft_deberta_v3_large", "ft_openai_detector_large"]
# Blend weights of the Kaggle files (../predict.py BLENDS; A from results/blend_selection/selection.json)
BLENDS = {
    "final_A_topic_greedy": {"binoculars_qwen05b": 4, "ft_distilroberta": 2, "ft_modernbert_base": 2,
                             "lm_gpt2_medium": 1, "lm_gpt2_nostyle": 1, "ft_roberta_base": 1, "lm_gpt2": 1},
    "final_B_lm4": {"lm_gpt2": 1, "lm_gpt2_medium": 1, "binoculars_qwen05b": 1, "binoculars_qwen15b": 1},
    "final_C_lm4_distil": {"lm_gpt2": 1, "lm_gpt2_medium": 1, "binoculars_qwen05b": 1,
                           "binoculars_qwen15b": 1, "ft_distilroberta": 1},
}
# original 5-fold CV runs (seed 42) behind the uploaded files, for the "CV vs fixed split" comparison
ORIGINAL_RUNS = {m: f"20261030-135404_ood__{m}" for m in MODELS}
ORIGINAL_RUNS.update(ft_deberta_v3_large="20261031-144451_ood__ft_deberta_v3_large",
                     ft_openai_detector_large="20261031-144451_ood__ft_openai_detector_large",
                     lm_gpt2_nostyle="20261031_var__lm_gpt2_nostyle")


# ------------------------------------------------------------------ running train.py
def train(run_name: str, config: str, sets: list[str], device: str | None = None) -> dict:
    """Run train.py once (or reuse a finished run) and return its metrics.json."""
    run_dir = RUNS / run_name
    if (run_dir / "metrics.json").exists():
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    cmd = [sys.executable, "train.py", "--model", config, "--run-name", run_name, "--cv-only", "--no-log",
           "--set", *sets] + (["--device", device] if device else [])
    log = RUNS.parent / "logs" / f"{run_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"{run_name} failed, see {log}")
    print(f"  done: {run_name}", flush=True)
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def fixed_preds(run_dir: Path) -> tuple[pd.Series, pd.Series]:
    """(validation predictions, training-row predictions) of a fixed-split run, indexed by id."""
    oof = pd.read_csv(run_dir / "oof.csv")
    val = oof[oof["fold"] >= 0].set_index("id")["pred"]
    tr = pd.read_csv(run_dir / "train_pred.csv").set_index("id")["pred"]
    return val, tr


def write(df: pd.DataFrame, rel: str) -> None:
    path = RESULTS / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", float_format="%.6g")
    print(f"wrote {path.relative_to(ROOT)} ({len(df)} rows)")


def labels() -> tuple[pd.Series, pd.DataFrame]:
    train = load_csv("data/train.csv").set_index("id")
    return train["label"], train


# ------------------------------------------------------------------ Q1.2 + blends
def blend(preds: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted mean of per-model percentile ranks; the uploaded blends map test scores through each
    model's OOF-score CDF, which on the rows the CDF comes from is exactly this (src/ensemble.cdf_mean)."""
    w = pd.Series(weights, dtype=float)
    return sum(w[m] / w.sum() * rank01(preds[m]) for m in w.index)


def cmd_collect(args):
    session = Path(args.session)
    y, _ = labels()
    orig = pd.read_csv(ROOT / "experiments" / "results.csv").set_index("run_name")
    rows, val, tr = [], {}, {}
    for m in MODELS:
        run = RUNS / f"{session.name}__{m}"
        met = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        val[m], tr[m] = fixed_preds(run)
        o = ORIGINAL_RUNS[m]
        rows.append({"model": m, "kind": "single", "train_auroc": met["train_auc"], "val_auroc": met["val_auc"],
                     "val_tpr_at_1fpr": met["oof_metrics"]["tpr_at_1fpr"], "fit_time_s": met["fit_time_s"],
                     "feature_time_s": met.get("feature_time_s", ""), "hardware": met.get("gpu_name", "CPU"),
                     "cv5_auroc_seed42": orig.loc[o, "cv_auc"] if o in orig.index else "",
                     # wall time of the original 5-fold run incl. LM feature extraction (fixed-split LM runs read
                     # the feature cache, so their fit_time_s is the LightGBM head only)
                     "cv5_run_time_s_seed42": orig.loc[o, "train_time_s"] if o in orig.index else "", "run": run.name})
    V = pd.DataFrame(val)
    T = pd.DataFrame(tr)
    yv, yt = y.loc[V.index].to_numpy(), y.loc[T.index].to_numpy()
    fit = {r["model"]: float(r["fit_time_s"]) for r in rows}
    for name, w in BLENDS.items():
        rows.append({"model": name, "kind": "blend", "train_auroc": summary(yt, blend(T, w))["auroc"],
                     "val_auroc": summary(yv, blend(V, w))["auroc"],
                     "val_tpr_at_1fpr": summary(yv, blend(V, w))["tpr_at_1fpr"],
                     "fit_time_s": sum(fit[m] for m in w), "hardware": "sum of members",
                     "cv5_auroc_seed42": "", "run": " + ".join(f"{m}x{k}" for m, k in w.items())})
    write(pd.DataFrame(rows), "fixed_split/models.csv")
    write(V.reset_index().rename(columns={"index": "id"}).assign(label=yv), "fixed_split/val_preds.csv")


# ------------------------------------------------------------------ Q1.3
def cmd_hpsearch(args):
    grids = {
        "tfidf_logreg": [{"model.params.C": C, "model.params.tfidf.word.ngram_range": ng}
                         for C in (0.5, 2.0, 4.0, 16.0, 64.0) for ng in ("[1,1]", "[1,2]")],
        "lm_gpt2": [{"model.params.lgbm": "{num_leaves: %d, learning_rate: %s, n_estimators: %d}" % (nl, lr, ne)}
                    for nl in (7, 15, 31, 63) for lr in (0.03, 0.1) for ne in (200, 500, 1000)],
    }
    for model, grid in grids.items():
        jobs = []
        for i, g in enumerate(grid):
            sets = ["cv.scheme=inner", "cv.n_folds=5", f"seed={SEED}"] + [f"{k}={v}" for k, v in g.items()]
            jobs.append((f"rep__hp_{model}_{i:02d}", sets, g))
        with ThreadPoolExecutor(args.jobs) as ex:
            mets = list(ex.map(lambda j: train(j[0], model, j[1], device="cpu" if model == "lm_gpt2" else None), jobs))
        rows = [{"model": model, **{k.split(".")[-1]: v for k, v in g.items()}, "inner_cv_auroc": m["cv_auc"],
                 "inner_cv_std": m["cv_auc_std"], "n_rows_scored": m["n_oof"], "run": name}
                for (name, _, g), m in zip(jobs, mets)]
        df = pd.DataFrame(rows)
        best = df.loc[df["inner_cv_auroc"].idxmax()]
        df["chosen"] = df.index == best.name
        # the chosen setting is then scored once on the fixed split (reported, not used for selection)
        g = grid[int(best.name)]
        met = train(f"rep__hp_{model}_chosen_fixed", model, FIXED + [f"{k}={v}" for k, v in g.items()])
        df["chosen_val_auroc"] = np.where(df["chosen"], met["val_auc"], np.nan)
        write(df, f"hpsearch/{model}.csv")


# ------------------------------------------------------------------ Q2
def cmd_curve(args):
    rows = []
    for model in ("lm_gpt2", "tfidf_logreg"):
        for n in (500, 1000, 2000, 5000, 8000):
            met = train(f"rep__curve_{model}_{n}", model, FIXED + [f"data.train_n={n}"])
            rows.append({"model": model, "train_rows": met["n_train"], "train_auroc": met["train_auc"],
                         "val_auroc": met["val_auc"], "fit_time_s": met["fit_time_s"]})
    write(pd.DataFrame(rows), "learning_curve.csv")


# ------------------------------------------------------------------ Q3
def cmd_semisup(args):
    _, train_df = labels()
    unl = load_csv("data/unlabeled.csv")
    val_ids = pd.read_csv(resolve(load_cfg_val_ids()))["id"]
    val_texts = set(train_df.loc[val_ids, "text"].str.strip())
    overlap = int(unl["text"].str.strip().isin(val_texts).sum())
    rows = []
    for name, sets in [("supervised", []),
                       ("pseudo_label_unlabeled_r1", ["semi_supervised.enabled=true", "semi_supervised.pool=[unlabeled]",
                                                      "semi_supervised.confident_frac=0.4", "semi_supervised.rounds=1"]),
                       ("pseudo_label_unlabeled_r3", ["semi_supervised.enabled=true", "semi_supervised.pool=[unlabeled]",
                                                      "semi_supervised.confident_frac=0.4", "semi_supervised.rounds=3"])]:
        for model in ("lm_gpt2", "tfidf_logreg"):
            met = train(f"rep__semi_{model}_{name}", model, FIXED + sets)
            rows.append({"model": model, "method": name, "train_auroc": met["train_auc"], "val_auroc": met["val_auc"],
                         "val_tpr_at_1fpr": met["oof_metrics"]["tpr_at_1fpr"], "fit_time_s": met["fit_time_s"],
                         "unlabeled_texts_identical_to_a_val_text": overlap})
    write(pd.DataFrame(rows), "semi_supervised.csv")


def load_cfg_val_ids() -> str:
    from src.config import load_config
    return load_config("lm_gpt2")["cv"]["val_ids_file"]


# ------------------------------------------------------------------ Q4
def cmd_errors(args):
    y, train = labels()
    V = pd.read_csv(RESULTS / "fixed_split" / "val_preds.csv").set_index("id")
    score = blend(V, BLENDS[args.blend])
    texts = train.loc[V.index, "text"]
    yv = y.loc[V.index].to_numpy()
    lb = length_bin_auroc(texts, yv, score).assign(grouping="word count")
    vp = verse_prose_auroc(texts, yv, score).assign(grouping="line structure")
    write(pd.concat([lb, vp], ignore_index=True), "error_analysis/subgroups.csv")
    err = errors(V.index, yv, score, threshold=0.5, k=5)
    # A strong blend makes fewer than 5 errors of a kind at 0.5: fill up with the closest misses on the right side
    # of the threshold (highest-scored humans / lowest-scored AI), marked as near misses.
    df = pd.DataFrame({"id": V.index, "label": yv, "p_ai": np.asarray(score)})
    for kind, cls, asc in (("FP", 0, False), ("FN", 1, True)):
        need = 5 - (err["error"] == kind).sum()
        if need > 0:
            cand = df[(df.label == cls) & ~df.id.isin(err.id)].sort_values("p_ai", ascending=asc).head(need)
            err = pd.concat([err, cand.assign(error=f"{kind} (near miss)")], ignore_index=True)
    err["words"] = word_count(texts.loc[err["id"]])
    err["blank_lines"] = [t.count("\n\n") for t in texts.loc[err["id"]]]
    err["lines"] = [len([ln for ln in t.splitlines() if ln.strip()]) for t in texts.loc[err["id"]]]
    for m in BLENDS[args.blend]:  # each member's percentile rank: which models were fooled
        err[f"rank_{m}"] = pd.Series(rank01(V[m]), index=V.index).loc[err["id"]].to_numpy()
    err["hypothesis"] = ""  # written by hand in the report after reading each passage
    write(err, "error_analysis/fp_fn.csv")
    n_fp = int(((score >= 0.5) & (yv == 0)).sum())
    n_fn = int(((score < 0.5) & (yv == 1)).sum())
    write(pd.DataFrame([{"threshold": 0.5, "n_val": len(yv), "false_positives": n_fp, "false_negatives": n_fn,
                         "val_auroc": summary(yv, score)["auroc"]}]), "error_analysis/confusion.csv")
    if args.show:  # read the passages (printed only, never written to results/)
        for _, r in err.iterrows():
            print(f"\n=== {r.error} id={r.id} P(AI)={r.p_ai:.3f} words={r.words}\n{train.loc[r.id, 'text'][:1200]}")


# ------------------------------------------------------------------ Q5
def cmd_debias(args):
    clean = ["preprocessing.collapse_blank_lines=true", "preprocessing.collapse_spaces=true",
             "preprocessing.normalize_quotes=true", "preprocessing.remove_zero_width=true",
             "preprocessing.map_homoglyphs=true"]
    rows = []
    for model in ("stylometric_gbm_ext", "lm_gpt2"):
        for variant, sets in [("as is", []), ("debiased", clean), ("join_lines", ["preprocessing.join_lines=true"])]:
            tag = variant.replace(" ", "_")
            met = train(f"rep__debias_{model}_{tag}", model, FIXED + sets)
            rows.append({"model": model, "variant": variant, "train_auroc": met["train_auc"],
                         "val_auroc": met["val_auc"], "val_tpr_at_1fpr": met["oof_metrics"]["tpr_at_1fpr"],
                         "fit_time_s": met["fit_time_s"], "feature_time_s": met.get("feature_time_s", "")})
    write(pd.DataFrame(rows), "debias_fixed_split.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["collect", "hpsearch", "curve", "semisup", "errors", "debias"])
    ap.add_argument("--session", help="collect: the run_all session of configs/experiments/fixed.yaml")
    ap.add_argument("--jobs", type=int, default=3, help="hpsearch: parallel train.py processes")
    ap.add_argument("--show", action="store_true", help="errors: print the FP/FN passages")
    ap.add_argument("--blend", default="final_B_lm4", choices=list(BLENDS), help="errors: the blend analysed")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    set_data_dir(args.data_dir)
    globals()[f"cmd_{args.command}"](args)


if __name__ == "__main__":
    main()
