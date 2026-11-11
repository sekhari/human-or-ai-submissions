"""Group 7 Model 1: Word TF-IDF + Logistic Regression baseline.

Adapted from the team member's public handoff at
https://github.com/yx749-lab/cornell-tech-cs5785-model1.
Reproduces a single predeclared C=1 baseline, not a tuning search.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

SEED = 42


def read_data(path: Path, labeled: bool) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str, "text": str}, keep_default_na=False)
    expected = {"id", "text", "label"} if labeled else {"id", "text"}
    if set(df.columns) != expected:
        raise ValueError(f"Expected columns {expected} in {path}, got {set(df.columns)}")
    if df.id.duplicated().any() or df.text.str.strip().eq("").any():
        raise ValueError(f"Duplicate ID or empty text in {path}")
    if labeled and not df.label.isin([0, 1]).all():
        raise ValueError("Unexpected label: use 0=Human, 1=AI")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b", stop_words=None,
            min_df=2, max_df=1.0, max_features=50_000,
            sublinear_tf=True, norm="l2", use_idf=True, smooth_idf=True,
        )),
        ("classifier", LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000, tol=1e-4,
            random_state=SEED,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "submit"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.outdir}")
    args.outdir.mkdir(parents=True)
    df = read_data(args.data_dir / "train.csv", labeled=True)
    if len(df) != 10000 or df.label.value_counts().to_dict() != {0: 5000, 1: 5000}:
        raise ValueError("Expected the official balanced 10,000-row train.csv")
    model = build_pipeline()
    t0 = time.perf_counter()
    if args.command == "evaluate":
        # Preserve original Kaggle CSV row order before splitting.
        train_idx, val_idx = train_test_split(
            np.arange(len(df)), test_size=0.2,
            random_state=SEED, stratify=df.label,
        )
        train, val = df.iloc[train_idx], df.iloc[val_idx]
        if len(train) != 8000 or len(val) != 2000:
            raise ValueError("Unexpected validation split size")
        model.fit(train.text, train.label)
        fit_seconds = time.perf_counter() - t0
        metrics = {"train_rows": len(train), "validation_rows": len(val)}
        for subset_name, subset in [("train", train), ("validation", val)]:
            prob = model.predict_proba(subset.text)[:, 1]
            metrics[subset_name + "_auroc"] = float(roc_auc_score(subset.label, prob))
            pd.DataFrame({"id": subset.id, "label": subset.label,
                          "probability": prob}).to_csv(
                              args.outdir / f"{subset_name}_predictions.csv", index=False)
    else:
        test = read_data(args.data_dir / "test.csv", labeled=False)
        sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"id": str})
        if len(test) != 3000 or list(sample.columns) != ["id", "label"] or set(test.id) != set(sample.id):
            raise ValueError("Expected matching official test and sample_submission files")
        if set(df.id) & set(test.id):
            raise ValueError("Train/test IDs overlap")
        model.fit(df.text, df.label)
        fit_seconds = time.perf_counter() - t0
        prob = model.predict_proba(test.text)[:, 1]
        if not np.isfinite(prob).all() or not ((prob >= 0) & (prob <= 1)).all():
            raise ValueError("Invalid probabilities")
        pd.DataFrame({"id": test.id, "label": prob}).to_csv(
            args.outdir / "submission.csv", index=False)
        metrics = {"train_rows": len(df), "test_rows": len(test)}
    report = {"model": "word_tfidf_logistic_regression", "seed": SEED,
              "fit_seconds": fit_seconds, "metrics": metrics,
              "scikit_learn": sklearn.__version__,
              "source": "yx749-lab/cornell-tech-cs5785-model1"}
    (args.outdir / "run_info.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
