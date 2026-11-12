"""Confidence-based pseudo-labeling for the unlabeled-data question.

Fits the shared TF-IDF + logistic regression baseline on the 8,000-row
training split, keeps unlabeled rows only when P(AI) is outside [0.10, 0.90),
and refits the same pipeline. The validation split is used only for the
final AUROC comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from baseline import ROOT, load_data, load_split, make_model

LO = 0.10
HI = 0.90
PROB_BINS = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0000001]


def normalize(text):
    return text.str.lower().str.split().str.join(" ")


def load_unlabeled(path):
    df = pd.read_csv(path, dtype={"id": str})
    required = ["id", "text"]
    if not set(required).issubset(df.columns):
        raise ValueError("Expected id and text columns in unlabeled.csv.")
    df = df[required].copy()
    if df.isna().any().any() or not df.id.is_unique or df.text.str.strip().eq("").any():
        raise ValueError("Missing values, repeated IDs, or empty text in unlabeled.csv.")
    if len(df) != 3000:
        raise ValueError("Expected the 3,000-row unlabeled.csv.")
    return df


def assert_no_split_leakage(unlabeled, train, val):
    unlabeled_norm = set(normalize(unlabeled.text))
    checks = {
        "id_overlap_train": len(set(unlabeled.id) & set(train.id)),
        "id_overlap_val": len(set(unlabeled.id) & set(val.id)),
        "exact_text_overlap_train": len(set(unlabeled.text) & set(train.text)),
        "exact_text_overlap_val": len(set(unlabeled.text) & set(val.text)),
        "normalized_text_overlap_train": len(unlabeled_norm & set(normalize(train.text))),
        "normalized_text_overlap_val": len(unlabeled_norm & set(normalize(val.text))),
    }
    if any(checks.values()):
        raise ValueError("Unlabeled data overlaps a labeled split: " + json.dumps(checks))
    return checks


def ai_probability(model, text):
    column = int(np.flatnonzero(model.classes_ == 1)[0])
    prob = model.predict_proba(text)[:, column]
    if not np.isfinite(prob).all() or not ((prob >= 0) & (prob <= 1)).all():
        raise ValueError("Invalid predicted probabilities.")
    return prob


def probability_bins(prob):
    counts = pd.cut(prob, bins=PROB_BINS, right=False).value_counts().sort_index()
    return {str(interval): int(count) for interval, count in counts.items()}


def select_pseudo_labels(unlabeled, prob, lo=LO, hi=HI):
    keep = (prob <= lo) | (prob >= hi)
    pseudo = unlabeled.loc[keep, ["id", "text"]].copy()
    pseudo["label"] = (prob[keep] >= hi).astype(int)
    return pseudo, keep


def confusion(y_true, prob):
    return confusion_matrix(y_true, prob >= 0.5, labels=[0, 1]).tolist()


def run(train_path, unlabeled_path, output_dir=ROOT):
    df = load_data(train_path)
    train, val = load_split(df)
    unlabeled = load_unlabeled(unlabeled_path)
    leakage = assert_no_split_leakage(unlabeled, train, val)

    supervised = make_model()
    supervised.fit(train.text, train.label)
    val_prob = ai_probability(supervised, val.text)
    unlabeled_prob = ai_probability(supervised, unlabeled.text)
    pseudo, keep = select_pseudo_labels(unlabeled, unlabeled_prob)

    combined = pd.concat([train[["id", "text", "label"]], pseudo], ignore_index=True)
    semi = make_model()
    semi.fit(combined.text, combined.label)
    semi_val_prob = ai_probability(semi, val.text)

    supervised_auroc = float(roc_auc_score(val.label, val_prob))
    semi_auroc = float(roc_auc_score(val.label, semi_val_prob))
    metrics = {
        "method": "Confidence-based pseudo-labeling",
        "model": "Word TF-IDF + Logistic Regression",
        "human_if_p_ai_le": LO,
        "ai_if_p_ai_ge": HI,
        "decision_threshold": 0.5,
        "unlabeled_rows": len(unlabeled),
        "selected_rows": int(keep.sum()),
        "discarded_rows": int((~keep).sum()),
        "pseudo_label_counts": {
            str(label): int(count) for label, count in pseudo.label.value_counts().sort_index().items()
        },
        "unlabeled_probability_bins": probability_bins(unlabeled_prob),
        "leakage_checks": leakage,
        "supervised_validation_auroc": supervised_auroc,
        "with_unlabeled_validation_auroc": semi_auroc,
        "auroc_delta": semi_auroc - supervised_auroc,
        "supervised_confusion_matrix_labels_0_1": confusion(val.label, val_prob),
        "with_unlabeled_confusion_matrix_labels_0_1": confusion(val.label, semi_val_prob),
        "validation_protocol": "Shared stratified split, 8000 train / 2000 validation, seed 42",
        "leakage_rule": (
            "Validation rows are excluded from fitting, vocabulary construction, "
            "and pseudo-label selection. The confidence cutoffs were fixed before evaluation."
        ),
    }

    output_dir = Path(output_dir)
    results = output_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "q3_pseudo_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame([{
        "method": "Supervised only",
        "validation_auroc": supervised_auroc,
    }, {
        "method": "With unlabeled data",
        "validation_auroc": semi_auroc,
    }]).to_csv(results / "q3_pseudo_metrics.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--unlabeled", type=Path, default=ROOT / "data" / "unlabeled.csv")
    args = parser.parse_args()
    metrics = run(args.train, args.unlabeled)
    print(pd.read_csv(ROOT / "results" / "q3_pseudo_metrics.csv").to_string(index=False))
    print(
        f"selected {metrics['selected_rows']} / {metrics['unlabeled_rows']}; "
        f"pseudo labels {metrics['pseudo_label_counts']}"
    )


if __name__ == "__main__":
    main()
