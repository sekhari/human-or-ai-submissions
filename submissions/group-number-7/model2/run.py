"""Group Number 7: reproduce Model 2 (character TF-IDF + calibrated LinearSVC).

This standalone adapter follows the selected Model 2 configuration and
train/validation protocol documented by Yuqi (Yokie) Zhang. It never uses
Kaggle test labels or unlabeled passages for model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


SETTINGS = {
    "analyzer": "char",
    "ngram_range": (2, 4),
    "min_df": 2,
    "max_features": 100_000,
    "lowercase": True,
    "sublinear_tf": True,
    "C": 1.0,
    "max_iter": 10_000,
    "calibration_folds": 3,
    "seed": 42,
}


def read_csv(path: Path, labeled: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str, "text": str}, keep_default_na=False)
    required = {"id", "text", "label"} if labeled else {"id", "text"}
    if set(frame.columns) != required:
        raise ValueError(f"{path}: required columns {sorted(required)}, got {list(frame.columns)}")
    if frame.empty or frame.id.duplicated().any() or frame.id.str.strip().eq("").any():
        raise ValueError(f"{path}: IDs must be nonempty and unique")
    if frame.text.str.strip().eq("").any():
        raise ValueError(f"{path}: blank text found")
    if labeled and (not frame.label.isin([0, 1]).all() or frame.label.nunique() != 2):
        raise ValueError(f"{path}: labels must be binary, including both 0 and 1")
    return frame


def pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=SETTINGS["analyzer"],
            ngram_range=SETTINGS["ngram_range"],
            min_df=SETTINGS["min_df"],
            max_features=SETTINGS["max_features"],
            lowercase=SETTINGS["lowercase"],
            sublinear_tf=SETTINGS["sublinear_tf"],
        )),
        ("svm", LinearSVC(
            C=SETTINGS["C"], dual="auto", max_iter=SETTINGS["max_iter"],
            random_state=SETTINGS["seed"],
        )),
    ])


def calibrated_pipeline() -> CalibratedClassifierCV:
    cv = StratifiedKFold(
        n_splits=SETTINGS["calibration_folds"],
        shuffle=True, random_state=SETTINGS["seed"],
    )
    # TF-IDF is inside the pipeline: no validation data enters vocabulary fitting.
    return CalibratedClassifierCV(
        pipeline(), method="sigmoid", cv=cv, ensemble=False, n_jobs=1,
    )


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Retain original CSV ordering, which matters for reproducing the team's split.
    _, val_indices = train_test_split(
        np.arange(len(frame)), test_size=0.2,
        stratify=frame.label, random_state=SETTINGS["seed"],
    )
    is_val = np.zeros(len(frame), dtype=bool)
    is_val[val_indices] = True
    train, val = frame.loc[~is_val], frame.loc[is_val]
    if len(train) != 8000 or len(val) != 2000:
        raise ValueError("Official labeled CSV must contain 10,000 rows (8,000/2,000 split)")
    if set(train.text) & set(val.text):
        raise ValueError("Exact duplicate text spans train/validation; review before fitting")
    return train, val


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["evaluate", "submit"])
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Local Kaggle data directory (NOT committed to GitHub)")
    p.add_argument("--outdir", type=Path, required=True,
                   help="Fresh results directory; never overwritten")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    t0 = perf_counter()
    warnings.filterwarnings("error", category=ConvergenceWarning)
    train_file = args.data_dir / "train.csv"
    all_labeled = read_csv(train_file, labeled=True)
    if len(all_labeled) != 10_000 or all_labeled.label.value_counts().to_dict() != {0: 5000, 1: 5000}:
        raise ValueError("Expected full official train.csv: 10,000 rows, balanced labels")
    metrics: dict[str, float | int] = {}
    if args.command == "evaluate":
        train, val = split(all_labeled)
        uncalibrated = pipeline().fit(train.text, train.label)
        calibrated = calibrated_pipeline().fit(train.text, train.label)
        for name, part in [("train", train), ("validation", val)]:
            margins = uncalibrated.decision_function(part.text)
            prob = calibrated.predict_proba(part.text)[:, 1]
            metrics[f"{name}_auroc"] = float(roc_auc_score(part.label, prob))
            metrics[f"{name}_margin_auroc"] = float(roc_auc_score(part.label, margins))
            pd.DataFrame({
                "id": part.id, "label": part.label, "margin": margins,
                "probability": prob, "word_count": part.text.str.split().str.len(),
            }).to_csv(args.outdir / f"{name}_predictions.csv", index=False)
        metrics.update(train_rows=len(train), validation_rows=len(val))
    else:
        test_file = args.data_dir / "test.csv"
        sample_file = args.data_dir / "sample_submission.csv"
        test = read_csv(test_file, labeled=False)
        sample = pd.read_csv(sample_file, dtype={"id": str}, keep_default_na=False)
        if len(test) != 3000 or set(sample.columns) != {"id", "label"}:
            raise ValueError("Expected 3,000 test rows and an id,label sample submission")
        if sample.id.duplicated().any() or set(sample.id) != set(test.id):
            raise ValueError("Sample submission IDs do not match test.csv")
        if set(test.id) & set(all_labeled.id):
            raise ValueError("Train/test IDs overlap")
        model = calibrated_pipeline().fit(all_labeled.text, all_labeled.label)
        prob = model.predict_proba(test.text)[:, 1]
        if not np.isfinite(prob).all() or not ((0 <= prob) & (prob <= 1)).all():
            raise ValueError("Invalid probabilities")
        pd.DataFrame({"id": test.id, "label": prob}).to_csv(
            args.outdir / "submission.csv", index=False,
        )
        metrics.update(train_rows=len(all_labeled), test_rows=len(test))
    report = {
        "command": args.command,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in SETTINGS.items()},
        "metrics": metrics,
        "elapsed_seconds": perf_counter() - t0,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "train_sha256": sha256(train_file),
    }
    (args.outdir / "run_info.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
