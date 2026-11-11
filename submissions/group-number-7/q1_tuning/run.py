"""Q1.3: retrospective training-only hyperparameter comparison for two models.

The fixed 2,000-row holdout is never used for choosing parameters. This
reproduces the six post-hoc CV comparisons reported in the team report; it
must not be described as the original selection process for Model 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42


def model_configs():
    for c in (0.1, 1.0, 10.0):
        yield "Model 1", f"word 1-2; C={c:g}", Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                      max_features=50000, sublinear_tf=True)),
            ("classifier", LogisticRegression(C=c, max_iter=1000, random_state=SEED)),
        ])
    for ngram, c in [((2, 4), 0.5), ((2, 4), 1.0), ((3, 5), 1.0)]:
        yield "Model 2", f"char {ngram[0]}-{ngram[1]}; C={c:g}", Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=ngram,
                                      min_df=2, max_features=100000, sublinear_tf=True)),
            ("classifier", LinearSVC(C=c, random_state=SEED)),
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"Use a new output directory: {args.outdir}")
    data_path = args.data_dir / "train.csv"
    if not data_path.is_file():
        parser.error(f"Missing official Kaggle file: {data_path}")
    df = pd.read_csv(data_path)
    if list(df.columns) != ["id", "text", "label"]:
        raise ValueError("Expected official train.csv columns: id,text,label")
    if len(df) != 10000 or df.id.duplicated().any() or df.label.value_counts().to_dict() != {0: 5000, 1: 5000}:
        raise ValueError("Expected unique IDs and exactly 5,000 rows of each label")
    if df.text.isna().any():
        raise ValueError("Missing text")
    # Numeric IDs 1..10000; keep the same sorting choice as the recorded Q1 run.
    df = df.sort_values("id").reset_index(drop=True)
    fit, holdout = train_test_split(df, test_size=2000, stratify=df.label,
                                    random_state=SEED)
    if len(fit) != 8000 or len(holdout) != 2000 or set(fit.id) & set(holdout.id):
        raise AssertionError("Split protocol failed")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    args.outdir.mkdir(parents=True)
    results = []
    for model, description, classifier in model_configs():
        start = perf_counter()
        scores = cross_validate(classifier, fit.text, fit.label, scoring="roc_auc",
                                cv=cv, n_jobs=1, return_train_score=False,
                                error_score="raise")["test_score"]
        row = {
            "Model": model,
            "Configuration": description,
            "Mean CV AUROC": float(scores.mean()),
            "Std CV AUROC": float(scores.std()),
            "CV wall-clock seconds": perf_counter() - start,
            **{f"Fold{i + 1}": float(v) for i, v in enumerate(scores)},
        }
        results.append(row)
        pd.DataFrame(results).to_csv(args.outdir / "q1_training_only_tuning.csv", index=False)
        print(f"{model}: {description}: mean CV AUROC={row['Mean CV AUROC']:.6f}", flush=True)
    info = {
        "purpose": "retrospective Q1.3 training-only comparison, not original model selection",
        "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "training_rows": len(fit), "holdout_rows_not_used_in_cv": len(holdout),
        "split": "official train.csv sorted by numeric id; stratified 8000/2000; seed 42",
        "cv": "StratifiedKFold(3, shuffle=True, random_state=42) on 8000 training rows",
        "model2_comparison": "uncalibrated LinearSVC decision-score AUROC; final model uses sigmoid calibration",
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "limitation": "one CV fold partition; no inference about held-out test performance",
    }
    (args.outdir / "protocol.json").write_text(json.dumps(info, indent=2) + "\n")
    print("Saved q1_training_only_tuning.csv and protocol.json")


if __name__ == "__main__":
    main()
