import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
TEAM_DIR = BASE_DIR.parent
SEED = 42


def build_model():
    word_features = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        max_features=50000,
        min_df=2,
        sublinear_tf=True,
        norm="l2"
    )

    character_features = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6),
        lowercase=False,
        max_features=200000,
        min_df=3,
        sublinear_tf=True,
        norm="l2"
    )

    return Pipeline([
        (
            "features",
            FeatureUnion([
                ("word", word_features),
                ("char", character_features)
            ])
        ),
        (
            "svm",
            LinearSVC(
                C=1.0,
                dual="auto",
                max_iter=10000,
                random_state=SEED
            )
        )
    ])


def build_calibrated_model():
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED
    )

    return CalibratedClassifierCV(
        estimator=build_model(),
        method="sigmoid",
        cv=cv,
        ensemble=False,
        n_jobs=1
    )


def normalize_text(text):
    return (
        text.str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the selected Round 2 SVM."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=TEAM_DIR / "train.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "svm_results"
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.train_csv)

    required_columns = {"id", "text", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Training CSV must contain id, text, and label columns."
        )

    if len(data) != 10000:
        raise ValueError("Expected the full 10,000-row training dataset.")

    if not data["id"].is_unique:
        raise ValueError("Training IDs must be unique.")

    if data[["id", "text", "label"]].isna().any().any():
        raise ValueError("Training data contains missing values.")

    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("Labels must be 0 (human) or 1 (AI).")

    if not data["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("All text values must be strings.")

    # Keep the original row order for a reproducible shared split.
    train_idx, val_idx = train_test_split(
        np.arange(len(data)),
        test_size=2000,
        random_state=SEED,
        stratify=data["label"]
    )

    train = data.iloc[train_idx]
    val = data.iloc[val_idx]

    overlap = (
        set(normalize_text(train["text"]))
        & set(normalize_text(val["text"]))
    )
    if overlap:
        raise ValueError(
            "Normalized exact duplicates cross the training/validation "
            "split. Resolve the shared split before continuing."
        )

    split = data[["id"]].copy()
    split["split"] = "train"
    split.loc[val_idx, "split"] = "validation"
    split.to_csv(output_dir / "split.csv", index=False)

    # Evaluate the selected configuration before probability calibration.
    uncalibrated_model = build_model()

    start = time.perf_counter()
    uncalibrated_model.fit(train["text"], train["label"])
    pipeline_fit_seconds = time.perf_counter() - start

    train_auc = roc_auc_score(
        train["label"],
        uncalibrated_model.decision_function(train["text"])
    )
    val_auc = roc_auc_score(
        val["label"],
        uncalibrated_model.decision_function(val["text"])
    )

    del uncalibrated_model

    # Calibration uses training-only cross-validation.
    # Each fold fits its own complete TF-IDF + SVM pipeline.
    model = build_calibrated_model()

    start = time.perf_counter()
    model.fit(train["text"], train["label"])
    calibration_fit_seconds = time.perf_counter() - start

    val_probabilities = model.predict_proba(val["text"])[:, 1]
    val_predictions = (val_probabilities >= 0.5).astype(int)

    calibrated_val_auc = roc_auc_score(
        val["label"],
        val_probabilities
    )

    validation_results = val[["id", "label"]].copy()
    validation_results["p_ai"] = val_probabilities
    validation_results["prediction"] = val_predictions
    validation_results.to_csv(
        output_dir / "validation_predictions.csv",
        index=False
    )

    cm = pd.DataFrame(
        confusion_matrix(
            val["label"],
            val_predictions,
            labels=[0, 1]
        ),
        index=["Actual: Human", "Actual: AI"],
        columns=["Predicted: Human", "Predicted: AI"]
    )
    cm.to_csv(output_dir / "confusion_matrix.csv")

    metrics = {
        "version": "Round 2",
        "C": 1.0,
        "seed": SEED,
        "training_rows": len(train),
        "validation_rows": len(val),
        "uncalibrated_train_auroc": float(train_auc),
        "uncalibrated_validation_auroc": float(val_auc),
        "calibrated_validation_auroc": float(calibrated_val_auc),
        "uncalibrated_pipeline_fit_seconds": pipeline_fit_seconds,
        "calibrated_pipeline_fit_seconds": calibration_fit_seconds,
        "decision_threshold": 0.5
    }

    print("Selected Model: Word + Character TF-IDF Linear SVM")
    print("C: 1")
    print(f"Training AUROC: {train_auc:.6f}")
    print(f"Validation AUROC: {val_auc:.6f}")
    print(
        f"Calibrated Validation AUROC: "
        f"{calibrated_val_auc:.6f}"
    )
    print(
        f"Uncalibrated Pipeline Fit Time: "
        f"{pipeline_fit_seconds:.2f} seconds"
    )
    print("\nValidation Confusion Matrix (threshold = 0.5)")
    print(cm.to_string())

    # Refit on all labeled examples only after validation.
    print("\nRefitting on all 10,000 labeled examples...", flush=True)

    start = time.perf_counter()
    model.fit(data["text"], data["label"])
    metrics["final_fit_seconds"] = time.perf_counter() - start
    metrics["final_training_rows"] = len(data)

    # Saves the vectorizers, SVM, and calibration together.
    model_path = output_dir / "svm_model.joblib"
    joblib.dump(model, model_path, compress=3)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Model saved: {model_path}")
    print(f"Results saved: {output_dir}")


if __name__ == "__main__":
    main()
