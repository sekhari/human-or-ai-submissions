"""Effect of training-set size: train on stratified subsets of the 8,000-row
training fold and always score on the same fixed 2,000-row validation set.

    python src/learning_curve.py

Writes results/learning_curve.csv and results/learning_curve.png.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".person_b_cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data import ROOT, get_split, load_splits, load_train
from models import MODELS

MODEL_NAMES = ["tfidf_logreg", "style_gbm"]
SIZES = [250, 500, 1000, 2000, 4000, 5000, 8000]
SEEDS = [0, 1, 2]
REQUIRED_SIZES = [500, 1000, 2000, 5000, 8000]


def subsample(df, n, seed):
    if n >= len(df):
        return df
    sub, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sub


def main(output_dir=None, rerun_style=False):
    """Fit fresh curves; optionally include the style model."""
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    train, splits = load_train(), load_splits()
    tr, va = get_split(train, splits, "random_fold")

    model_names = MODEL_NAMES if rerun_style else ["tfidf_logreg"]
    rows, details = [], []
    for name in model_names:
        for n in SIZES:
            seeds = SEEDS if n < len(tr) else [0]  # full set: subsets are identical
            for seed in seeds:
                sub = subsample(tr, n, seed)
                assert not set(sub["id"]) & set(va["id"])
                model = MODELS[name]()
                started = time.perf_counter()
                model.fit(sub["text"], sub["label"])
                fit_seconds = time.perf_counter() - started
                train_auc = roc_auc_score(sub["label"], model.predict_proba(sub["text"])[:, 1])
                auc = roc_auc_score(va["label"], model.predict_proba(va["text"])[:, 1])
                row = {"model": name, "n_train": n, "seed": seed,
                       "train_auroc": train_auc, "val_auroc": auc}
                rows.append(row)
                ids = "\n".join(sorted(map(str, sub["id"]))) + "\n"
                details.append({**row, "fit_seconds": fit_seconds,
                                "training_ids_sha256": hashlib.sha256(ids.encode()).hexdigest(),
                                "training_class_counts": sub["label"].value_counts().to_dict()})
                print(f"{name:<14} n={n:<5} seed={seed} train={train_auc:.6f} val={auc:.6f}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = output_dir / "learning_curve.csv"
    df.to_csv(out_csv, index=False)

    summary = df.groupby(["model", "n_train"])["val_auroc"].agg(["mean", "std"]).fillna(0)
    print("\n", summary.round(4), flush=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in model_names:
        s = summary.loc[name]
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=name)
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlabel("Number of training examples (log scale)")
    ax.set_ylabel("Validation AUROC (fixed 2,000 rows)")
    ax.set_title("Effect of training-set size")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = output_dir / "learning_curve.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # Q2 main table: the original TF-IDF model and original seeds, averaged
    # over three stratified samples; the full pool has only one distinct fit.
    required = df[(df["model"] == "tfidf_logreg") & df["n_train"].isin(REQUIRED_SIZES)]
    table = required.groupby("n_train").agg(
        training_auroc=("train_auroc", "mean"), validation_auroc=("val_auroc", "mean"),
        training_std=("train_auroc", "std"), validation_std=("val_auroc", "std"),
        repeats=("seed", "size")).fillna(0).reset_index()
    table.to_csv(output_dir / "learning_curve_required.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.2), layout="constrained")
    for column, error, label, marker, color in [
        ("training_auroc", "training_std", "Training AUROC", "o", "#2563eb"),
        ("validation_auroc", "validation_std", "Validation AUROC", "s", "#d97706")]:
        ax.errorbar(table["n_train"], table[column], yerr=table[error],
                    marker=marker, color=color, label=label, capsize=3, linewidth=2)
    ax.set(xlabel="Labeled training examples", ylabel="AUROC", ylim=(.83, 1.006))
    ax.set_xticks(REQUIRED_SIZES, [f"{n:,}" for n in REQUIRED_SIZES], fontsize=10)
    ax.grid(alpha=.2)
    ax.legend(loc="lower right", fontsize=10)
    fig.savefig(output_dir / "learning_curve_train_val.png", dpi=240)
    plt.close(fig)
    (output_dir / "learning_curve_details.json").write_text(json.dumps({
        "based_on": "Aarushi's learning_curve.py at 0e09450; unchanged models, seeds and subsample function",
        "models": model_names, "all_scores_fitted_in_this_run": True,
        "seeds": SEEDS, "required_sizes": REQUIRED_SIZES, "runs": details,
        "validation_rows": len(va), "summary": "Mean +/- sample SD; n=8000 is one distinct full-pool fit, not an uncertainty estimate."
    }, indent=2) + "\n")
    print(f"Saved learning curves and required train/validation table in {output_dir}", flush=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--rerun-style", action="store_true", help="Also fit fresh style-model curves.")
    args = parser.parse_args()
    main(args.output_dir, args.rerun_style)
