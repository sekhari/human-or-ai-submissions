"""Q4: generalization and error analysis for the selected Q1 model.

Uses the saved validation and training predictions of the selected pipeline
(character TF-IDF 3-5-grams + LinearSVC, C=1, training-only sigmoid
calibration). Reads train.csv only to look up texts for the chosen error IDs
and to count simple formatting features. No passages are written to disk.

    python q4_error_analysis.py --train data/train.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from baseline import ROOT, load_data

THRESHOLD = 0.5  # P(AI) >= 0.5 is predicted AI, same rule as the baseline files
LENGTH_BINS = [(75, 175), (176, 239), (240, 350)]  # equal-count tertiles used in the first Q4 draft
NEWLINE_BINS = [(0, 0), (1, 5), (6, 10_000)]
N_CASES = 5


def auroc_table(frame, column, bins):
    rows = []
    for lo, hi in bins:
        part = frame[(frame[column] >= lo) & (frame[column] <= hi)]
        rows.append({
            "bin": f"{lo}-{hi}" if hi < 10_000 else f"{lo}+",
            "count": int(len(part)),
            "n_ai": int(part.true_label.sum()),
            "n_human": int((1 - part.true_label).sum()),
            "auroc": round(float(roc_auc_score(part.true_label, part.predicted_ai_probability)), 4),
        })
    return rows


def formatting_features(texts):
    return pd.DataFrame({
        "newlines": texts.str.count("\n"),
        "blank_lines": texts.str.count("\n\n"),
        "non_ascii": texts.map(lambda t: sum(ord(c) > 127 for c in t)),
        "digits": texts.map(lambda t: sum(c.isdigit() for c in t)),
        "double_spaces": texts.str.count("  "),
    }, index=texts.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data" / "train.csv"))
    parser.add_argument("--out", default=str(ROOT / "results" / "q4"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    texts = load_data(args.train)[["id", "text"]]
    val = pd.read_csv(ROOT / "results/q1/selected_validation_predictions.csv", dtype={"id": str})
    train = pd.read_csv(ROOT / "results/q1/selected_training_predictions.csv", dtype={"id": str})
    base_val = pd.read_csv(ROOT / "results/validation_predictions.csv", dtype={"id": str})

    val = val.merge(texts, on="id", validate="one_to_one")
    val = pd.concat([val, formatting_features(val.text)], axis=1)
    pred = (val.predicted_ai_probability >= THRESHOLD).astype(int)
    assert (pred == val.predicted_label).all()

    summary = {
        "model": "char TF-IDF (3-5) + LinearSVC C=1, sigmoid-calibrated on training folds",
        "threshold": THRESHOLD,
        "train_auroc": round(float(roc_auc_score(train.true_label, train.predicted_ai_probability)), 6),
        "validation_auroc": round(float(roc_auc_score(val.true_label, val.predicted_ai_probability)), 6),
        "confusion_matrix_labels_0_1": confusion_matrix(val.true_label, pred).tolist(),
        "length_bins": auroc_table(val, "word_count", LENGTH_BINS),
        "newline_bins": auroc_table(val, "newlines", NEWLINE_BINS),
        "baseline_length_bins": auroc_table(base_val, "word_count", LENGTH_BINS),
    }
    summary["gap"] = round(summary["train_auroc"] - summary["validation_auroc"], 6)

    # How much do the selected model and the word-LR baseline fail on the same rows?
    both = val[["id", "error_type"]].merge(base_val[["id", "error_type"]], on="id", suffixes=("_svm", "_lr"))
    svm_err = both.error_type_svm != "correct"
    lr_err = both.error_type_lr != "correct"
    summary["error_overlap_with_word_lr"] = {
        "svm_errors": int(svm_err.sum()), "lr_errors": int(lr_err.sum()),
        "shared": int((svm_err & lr_err).sum()),
        "score_correlation_pearson": round(float(np.corrcoef(
            val.sort_values("id").predicted_ai_probability,
            base_val.sort_values("id").predicted_ai_probability)[0, 1]), 4),
    }

    feats = ["newlines", "blank_lines", "non_ascii", "digits", "double_spaces", "word_count"]
    summary["feature_means_by_outcome"] = val.groupby("error_type")[feats].mean().round(2).to_dict(orient="index")
    summary["feature_means_by_label"] = val.groupby("true_label")[feats].mean().round(2).to_dict(orient="index")

    # Error cases: the most confident mistakes, a fixed rule rather than hand-picked rows.
    fp = val[val.error_type == "FP"].sort_values("predicted_ai_probability", ascending=False).head(N_CASES)
    fn = val[val.error_type == "FN"].sort_values("predicted_ai_probability").head(N_CASES)
    cases = pd.concat([fp, fn])[["id", "error_type", "predicted_ai_probability", "word_count", "newlines", "double_spaces"]]
    summary["cases"] = cases.round(4).to_dict(orient="records")

    (out / "q4_summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["| id | error | P(AI) | words | newlines | double spaces |", "|---|---|---:|---:|---:|---:|"]
    for r in summary["cases"]:
        lines.append(f"| {r['id']} | {r['error_type']} | {r['predicted_ai_probability']:.4f} | "
                     f"{r['word_count']} | {r['newlines']} | {r['double_spaces']} |")
    (out / "error_cases.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in ["train_auroc", "validation_auroc", "gap", "length_bins",
                                               "newline_bins", "error_overlap_with_word_lr"]}, indent=2))
    print(pd.DataFrame(summary["feature_means_by_outcome"]).T)


if __name__ == "__main__":
    main()
