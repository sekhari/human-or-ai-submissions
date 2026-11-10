"""Ablate structural writing cues for Team 35's character logistic model.

The cues are hypotheses inspired by Wikipedia's "Signs of AI writing" field guide;
they are not labels or rules for deciding authorship by themselves.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from train import make_pipeline


WORD = re.compile(r"\b[\w'-]+\b")
PATTERNS = {
    "not_but": re.compile(r"\bnot\b[^.!?;\n]{0,120}?\bbut\b", re.IGNORECASE),
    "rather_than": re.compile(r"\brather\s+than\b", re.IGNORECASE),
    "three_item_list": re.compile(
        r"\b(?:[a-z][a-z'-]*\s+){0,2}[a-z][a-z'-]*,\s+"
        r"(?:[a-z][a-z'-]*\s+){0,2}[a-z][a-z'-]*,?\s+"
        r"(?:and|or)\s+(?:[a-z][a-z'-]*\s+){0,2}[a-z][a-z'-]*\b",
        re.IGNORECASE,
    ),
}


def style_features(texts: np.ndarray) -> np.ndarray:
    """Return pattern counts per 100 words, in PATTERNS insertion order."""
    rows = []
    for text in texts:
        words = max(1, len(WORD.findall(text)))
        rows.append([100.0 * len(pattern.findall(text)) / words for pattern in PATTERNS.values()])
    return np.asarray(rows, dtype=np.float64)


def evaluate(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    return {
        "auc": float(roc_auc_score(y, probability)),
        "errors_at_0.5": int(np.sum((probability >= 0.5) != y)),
        "error_rate_at_0.5": float(np.mean((probability >= 0.5) != y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    team_dir = Path(__file__).resolve().parent
    parser.add_argument("--train", type=Path, default=team_dir / "data" / "train.csv")
    parser.add_argument("--output", type=Path, default=team_dir / "data" / "style" / "style_ablation.json")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--style-weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.C <= 0 or args.style_weight < 0:
        raise ValueError("C must be positive and style-weight must be nonnegative")

    frame = pd.read_csv(args.train)
    missing = {args.text_column, args.label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {list(frame.columns)}")
    if frame[[args.text_column, args.label_column]].isna().any().any():
        raise ValueError("Text and label columns must not contain missing values")
    texts = frame[args.text_column].astype(str).to_numpy()
    labels = pd.to_numeric(frame[args.label_column], errors="raise").to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected binary labels 0=human and 1=AI")
    train_idx, val_idx = train_test_split(
        np.arange(len(frame)), test_size=args.validation_size,
        stratify=labels, random_state=args.seed,
    )
    x_train, x_val = texts[train_idx], texts[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    baseline = make_pipeline(args.C, args.seed)
    start = time.perf_counter()
    baseline.fit(x_train, y_train)
    baseline_seconds = time.perf_counter() - start
    baseline_train = baseline.predict_proba(x_train)[:, 1]
    baseline_val = baseline.predict_proba(x_val)[:, 1]

    start = time.perf_counter()
    vectorizer = clone(baseline.named_steps["tfidf"])
    tfidf_train = vectorizer.fit_transform(x_train)  # Fit using training text only.
    raw_style_train = style_features(x_train)
    scaler = StandardScaler()
    scaled_style_train = scaler.fit_transform(raw_style_train)
    combined_train = sparse.hstack(
        [tfidf_train, sparse.csr_matrix(args.style_weight * scaled_style_train)],
        format="csr",
    )
    augmented = LogisticRegression(C=args.C, max_iter=1000, random_state=args.seed)
    augmented.fit(combined_train, y_train)
    augmented_seconds = time.perf_counter() - start
    raw_style_val = style_features(x_val)
    scaled_style_val = scaler.transform(raw_style_val)
    combined_val = sparse.hstack(
        [vectorizer.transform(x_val), sparse.csr_matrix(args.style_weight * scaled_style_val)],
        format="csr",
    )
    augmented_train = augmented.predict_proba(combined_train)[:, 1]
    augmented_val = augmented.predict_proba(combined_val)[:, 1]

    baseline_metrics = {
        "train": evaluate(y_train, baseline_train),
        "validation": evaluate(y_val, baseline_val),
        "fit_seconds": round(baseline_seconds, 3),
    }
    augmented_metrics = {
        "train": evaluate(y_train, augmented_train),
        "validation": evaluate(y_val, augmented_val),
        "fit_seconds": round(augmented_seconds, 3),
    }
    cue_prevalence = {
        name: {
            "human_fraction_with_cue": float(np.mean(raw_style_train[y_train == 0, i] > 0)),
            "ai_fraction_with_cue": float(np.mean(raw_style_train[y_train == 1, i] > 0)),
        }
        for i, name in enumerate(PATTERNS)
    }
    result = {
        "seed": args.seed,
        "train_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "C": args.C,
        "style_weight": args.style_weight,
        "cue_names": list(PATTERNS),
        "cue_definition": "Approximate regex counts per 100 words; three_item_list is not a full grammatical parser.",
        "cue_prevalence_on_training_partition": cue_prevalence,
        "baseline": baseline_metrics,
        "augmented": augmented_metrics,
        "delta_validation_auc": augmented_metrics["validation"]["auc"] - baseline_metrics["validation"]["auc"],
        "note": "The same fixed holdout is used for both; no Kaggle test labels or external text labels were used.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
