"""Reproduce Q2–Q4 for the character TF-IDF baseline."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
from train import make_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    team_dir = Path(__file__).resolve().parent
    repo_root = team_dir.parents[1]
    default_train = repo_root / "data" / "train.csv"
    if not default_train.exists():
        default_train = team_dir / "data" / "train.csv"
    default_unlabeled = repo_root / "data" / "unlabeled.csv"
    if not default_unlabeled.exists():
        default_unlabeled = team_dir / "data" / "unlabeled.csv"
    parser.add_argument("--train", type=Path, default=default_train, help="Path to train.csv")
    parser.add_argument("--unlabeled", type=Path, default=default_unlabeled, help="Path to unlabeled.csv")
    parser.add_argument("--output-dir", type=Path, default=team_dir / "data" / "baseline-analysis", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.train)
    train, val = train_test_split(frame, test_size=2000, stratify=frame.label, random_state=args.seed)
    curve = []
    target_sizes = [s for s in (500, 1000, 2000, 5000, 8000) if s <= len(train)]
    if not target_sizes:
        target_sizes = [len(train)]
    for size in target_sizes:
        subset = train if size == len(train) else train_test_split(
            train, train_size=size, stratify=train.label, random_state=args.seed
        )[0]
        model = make_pipeline(10, args.seed).fit(subset.text, subset.label)
        train_auc = roc_auc_score(subset.label, model.predict_proba(subset.text)[:, 1])
        scores = model.predict_proba(val.text)[:, 1]
        curve.append([size, train_auc, roc_auc_score(val.label, scores)])
        print(curve[-1], flush=True)

    if args.unlabeled.exists():
        unlabeled = pd.read_csv(args.unlabeled).dropna(subset=["text"])
        unlabeled = unlabeled.drop_duplicates("text")
        unlabeled = unlabeled.loc[~unlabeled.text.isin(frame.text)]
        if len(unlabeled) > 0:
            probabilities = model.predict_proba(unlabeled.text)[:, 1]
            confident = (probabilities <= 0.05) | (probabilities >= 0.95)
            pseudo = unlabeled.loc[confident].assign(label=(probabilities[confident] >= 0.95).astype(int))
            if len(pseudo) > 0:
                augmented = pd.concat([train, pseudo], ignore_index=True)
                model = make_pipeline(10, args.seed).fit(augmented.text, augmented.label)
                semi_auc = roc_auc_score(val.label, model.predict_proba(val.text)[:, 1])
                pseudo_counts = np.bincount(pseudo.label, minlength=2).tolist()
            else:
                pseudo_counts = [0, 0]
                semi_auc = curve[-1][2]
        else:
            pseudo_counts = [0, 0]
            semi_auc = curve[-1][2]
    else:
        pseudo_counts = [0, 0]
        semi_auc = curve[-1][2]

    errors = val.assign(probability=scores, words=val.text.str.split().str.len())
    errors["wrong"] = (scores >= 0.5) != errors.label
    bins = pd.cut(errors.words, bins=[0, 100, 200, np.inf], include_lowest=True)
    groups = []
    for group, rows in errors.groupby(bins, observed=True):
        auc = roc_auc_score(rows.label, rows.probability) if len(rows.label.unique()) > 1 else float("nan")
        groups.append([str(group), len(rows), auc, int(rows.wrong.sum())])

    wrong = errors.loc[errors.wrong]
    fp = wrong.loc[wrong.label == 0].nlargest(5, "probability")
    fn = wrong.loc[wrong.label == 1].nsmallest(5, "probability")
    examples = pd.concat([fp, fn])
    examples.to_csv(output / "examples-local.csv", index=False)
    result = dict(curve=curve, pseudo_counts=pseudo_counts,
                  semi_auc=semi_auc, length_groups=groups,
                  errors_by_label=np.bincount(wrong.label, minlength=2).tolist(),
                  examples=examples[["id", "label", "probability", "words"]].to_dict("records"))
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    sizes, training, validation = zip(*curve)
    fig, ax = plt.subplots(figsize=(6, 3.3), layout="constrained")
    ax.plot(sizes, training, marker="o", label="Training")
    ax.plot(sizes, validation, marker="s", label="Validation")
    ax.set(xlabel="Labeled training passages", ylabel="AUROC", xticks=sizes)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output / "baseline_learning_curve.pdf")

    report_figures = Path(__file__).resolve().parent.parent / "report" / "figures"
    if report_figures.exists() and args.output_dir == team_dir / "data" / "baseline-analysis":
        fig.savefig(report_figures / "baseline_learning_curve.pdf")


if __name__ == "__main__":
    main()

