"""Q4: generalization and error analysis on the fixed validation split.

Run from submissions/group-12:
    python -m src.q4_analysis --model masked_lr
    python -m src.q4_analysis --preds results/03/03_qwen2.5-0.5b_fixed_split.csv --name qwen05b_lora
Writes results/q4/<name>/*.csv and summary.md (numbers for report section 4).
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .clusters import assign_clusters
from .data import RESULTS_DIR, load_split
from .models import MODELS

THRESHOLD = 0.5
LENGTH_BINS = [0, 150, 250, np.inf]
LENGTH_LABELS = ["short (<150 words)", "medium (150-249)", "long (>=250)"]
N_ERRORS = 5


def format_group(texts):
    return pd.Series(np.select([texts.str.contains(r"\n\s*\n"), texts.str.contains("\n")],
                               ["blank lines", "single linebreaks"], "no linebreaks"),
                     index=texts.index)


def add_meta(df, clusters, top_terms, fiction):
    df = df.copy()
    df["n_words"] = df["text"].str.split().str.len()
    df["length_bin"] = pd.cut(df["n_words"], LENGTH_BINS, labels=LENGTH_LABELS, right=False)
    df["newlines_per_100w"] = df["text"].str.count("\n") / df["n_words"] * 100
    df["format"] = format_group(df["text"])
    df["cluster"] = df["id"].map(clusters).astype(int)
    df["topic"] = df["cluster"].map(top_terms)
    df["is_fiction"] = df["cluster"] == fiction
    return df


def group_auroc(df, col):
    rows = []
    for name, g in df.groupby(col, observed=True):
        auc = roc_auc_score(g["label"], g["p_ai"]) if g["label"].nunique() == 2 else np.nan
        rows.append({col: name, "count": len(g), "AI share": g["label"].mean(),
                     "val AUROC": auc, "error rate@0.5": ((g["p_ai"] > THRESHOLD) != g["label"]).mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="masked_lr", choices=list(MODELS))
    ap.add_argument("--preds", help="CSV of saved P(AI) for all train.csv ids from a model fit on the "
                                    "8,000-row training partition (e.g. results/03/03_qwen2.5-0.5b_fixed_split.csv)")
    ap.add_argument("--pred-col", default="qwen", help="probability column in --preds")
    ap.add_argument("--name", help="output folder / model label when using --preds")
    args = ap.parse_args()
    name = (args.name or "saved_preds") if args.preds else args.model
    out = RESULTS_DIR / "q4" / name
    out.mkdir(parents=True, exist_ok=True)

    train, val = load_split()
    clusters, top_terms, fiction = assign_clusters()
    train = add_meta(train, clusters["train"], top_terms, fiction)
    val = add_meta(val, clusters["train"], top_terms, fiction)

    if args.preds:
        # GPU models are trained in a notebook; reuse their saved fixed-split predictions here
        saved = pd.read_csv(args.preds)
        if "is_val" in saved:
            assert set(saved.loc[saved["is_val"], "id"]) == set(val["id"]), "--preds was made with a different split"
        p_by_id = saved.set_index("id")[args.pred_col]
        train["p_ai"], val["p_ai"] = train["id"].map(p_by_id), val["id"].map(p_by_id)
        assert train["p_ai"].notna().all() and val["p_ai"].notna().all(), "--preds is missing some ids"
        fit_s = np.nan
    else:
        t0 = time.time()
        model = MODELS[args.model]().fit(train["text"], train["label"])
        fit_s = time.time() - t0
        train["p_ai"] = model.predict_proba(train["text"])[:, 1]
        val["p_ai"] = model.predict_proba(val["text"])[:, 1]

    # Q4.1 train vs validation
    q41 = pd.DataFrame([{
        "model": name, "train rows": len(train), "val rows": len(val),
        "train AUROC": roc_auc_score(train["label"], train["p_ai"]),
        "val AUROC": roc_auc_score(val["label"], val["p_ai"]),
        "train error@0.5": ((train["p_ai"] > THRESHOLD) != train["label"]).mean(),
        "val error@0.5": ((val["p_ai"] > THRESHOLD) != val["label"]).mean(),
        "fit seconds": fit_s,
    }])
    q41.to_csv(out / "q41_train_vs_val.csv", index=False)

    # Q4.2 subgroups
    by_len = group_auroc(val, "length_bin")
    by_fmt = group_auroc(val, "format")
    by_topic = group_auroc(val, "topic")
    by_topic.insert(0, "cluster", [val.loc[val["topic"] == t, "cluster"].iloc[0] for t in by_topic["topic"]])
    by_topic["fiction?"] = by_topic["cluster"] == fiction
    by_len.to_csv(out / "q42_by_length.csv", index=False)
    by_fmt.to_csv(out / "q42_by_format.csv", index=False)
    by_topic.sort_values("val AUROC").to_csv(out / "q42_by_topic.csv", index=False)

    # Q4.3 most confident errors
    val["pred"] = (val["p_ai"] > THRESHOLD).astype(int)
    val["error"] = np.select([(val["label"] == 0) & (val["pred"] == 1), (val["label"] == 1) & (val["pred"] == 0)],
                             ["FP", "FN"], "")
    cols = ["id", "error", "p_ai", "n_words", "format", "newlines_per_100w", "cluster", "topic", "snippet"]
    val["snippet"] = val["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 160)
    errors = val[val["error"] != ""].copy()
    errors["confidence"] = (errors["p_ai"] - THRESHOLD).abs()
    errors.sort_values("confidence", ascending=False)[cols].to_csv(out / "q43_all_errors.csv", index=False)
    top = pd.concat([errors[errors["error"] == e].nlargest(N_ERRORS, "confidence") for e in ("FP", "FN")])
    top[cols].to_csv(out / "q43_top_errors.csv", index=False)

    # Q4.4 helpers: where do errors concentrate vs the overall validation set
    err_profile = pd.DataFrame({
        "all val": val[["n_words", "newlines_per_100w"]].mean(),
        "FP (human->AI)": errors.loc[errors["error"] == "FP", ["n_words", "newlines_per_100w"]].mean(),
        "FN (AI->human)": errors.loc[errors["error"] == "FN", ["n_words", "newlines_per_100w"]].mean(),
    }).T
    err_profile.to_csv(out / "q44_error_profile.csv")
    err_by_fmt = pd.crosstab(val["format"], val["error"].replace("", "correct"), normalize="index").round(3)
    err_by_fmt.to_csv(out / "q44_errors_by_format.csv")
    err_by_topic = pd.crosstab(val["topic"], val["error"].replace("", "correct"))
    err_by_topic.to_csv(out / "q44_errors_by_topic.csv")

    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Q4 results: {name} (threshold {THRESHOLD}, split seed 42)\n\n")
        f.write("## Q4.1 train vs validation\n\n" + q41.round(4).to_markdown(index=False) + "\n\n")
        f.write("## Q4.2 by length bin\n\n" + by_len.round(4).to_markdown(index=False) + "\n\n")
        f.write("## Q4.2 by linebreak format\n\n" + by_fmt.round(4).to_markdown(index=False) + "\n\n")
        f.write("## Q4.2 by topic cluster\n\n" + by_topic.sort_values("val AUROC").round(4).to_markdown(index=False) + "\n\n")
        f.write(f"## Q4.3 top {N_ERRORS} FP and FN\n\n" + top[cols].round(3).to_markdown(index=False) + "\n\n")
        f.write(f"Total errors on val: {len(errors)} ({(errors['error']=='FP').sum()} FP, {(errors['error']=='FN').sum()} FN)\n\n")
        f.write("## Q4.4 error profile\n\n" + err_profile.round(2).to_markdown() + "\n\n")
        f.write("### errors by format (row-normalised)\n\n" + err_by_fmt.to_markdown() + "\n\n")
        f.write("### errors by topic (counts)\n\n" + err_by_topic.to_markdown() + "\n")
    print(open(out / "summary.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
