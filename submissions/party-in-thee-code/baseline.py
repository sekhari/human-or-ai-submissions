import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
SEED = 42


def load_data(path):
    df = pd.read_csv(path, dtype={"id": str})
    required = ["id", "text", "label"]
    if not set(required).issubset(df.columns):
        raise ValueError("Expected id, text and label columns.")
    df = df[required].copy()
    if df.isna().any().any() or not df.id.is_unique:
        raise ValueError("Missing values or repeated IDs in training data.")
    if not df.label.isin([0, 1]).all() or df.text.str.strip().eq("").any():
        raise ValueError("Invalid labels or empty text.")
    df["label"] = df.label.astype(int)
    if len(df) != 10000 or df.label.value_counts().to_dict() != {0: 5000, 1: 5000}:
        raise ValueError("Expected the full balanced 10,000-row train.csv.")
    return df


def data_fingerprint(df):
    # Independent of filename and CSV row order.
    rows = df.sort_values("id")[["id", "text", "label"]].values.tolist()
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_split(df, split_dir=ROOT / "splits"):
    split_dir = Path(split_dir)
    train_path = split_dir / "train_ids.csv"
    val_path = split_dir / "validation_ids.csv"
    manifest_path = split_dir / "split_manifest.json"
    exists = [p.exists() for p in (train_path, val_path, manifest_path)]
    if any(exists) and not all(exists):
        raise ValueError("Split files are incomplete; restore the shared split files.")

    if not any(exists):
        ids = df.sort_values("id")
        train_ids, val_ids = train_test_split(
            ids.id.to_numpy(), test_size=2000, random_state=SEED, stratify=ids.label
        )
        split_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": train_ids}).to_csv(train_path, index=False)
        pd.DataFrame({"id": val_ids}).to_csv(val_path, index=False)
        manifest = {
            "seed": SEED,
            "method": "train_test_split, stratified by label, input sorted lexicographically by ID",
            "training_rows": 8000,
            "validation_rows": 2000,
            "dataset_sha256": data_fingerprint(df),
            "label_mapping": {"0": "human", "1": "AI"},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    manifest = json.loads(manifest_path.read_text())
    if data_fingerprint(df) != manifest["dataset_sha256"]:
        raise ValueError("The labeled dataset differs from the shared baseline data.")
    train_ids = pd.read_csv(train_path, dtype={"id": str}).id.tolist()
    val_ids = pd.read_csv(val_path, dtype={"id": str}).id.tolist()
    if len(train_ids) != 8000 or len(val_ids) != 2000:
        raise ValueError("Unexpected split sizes.")
    if len(set(train_ids)) != 8000 or len(set(val_ids)) != 2000:
        raise ValueError("Repeated IDs in the split files.")
    if set(train_ids) & set(val_ids) or set(train_ids) | set(val_ids) != set(df.id):
        raise ValueError("The splits overlap or do not cover train.csv.")
    by_id = df.set_index("id", drop=False)
    train = by_id.loc[train_ids].reset_index(drop=True)
    val = by_id.loc[val_ids].reset_index(drop=True)
    if train.label.value_counts().to_dict() != {0: 4000, 1: 4000}:
        raise ValueError("Training split is not balanced.")
    if val.label.value_counts().to_dict() != {0: 1000, 1: 1000}:
        raise ValueError("Validation split is not balanced.")
    return train, val


def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 2),
            min_df=2,
            max_df=1.0,
            max_features=50000,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm="l2",
            stop_words=None,
            strip_accents=None,
            dtype=np.float64,
        )),
        ("lr", LogisticRegression(
            C=1.0, solver="liblinear", max_iter=2000,
            tol=1e-4, random_state=SEED, class_weight=None,
        )),
    ])


def audit_data(df, train, val):
    def normalize(text):
        return " ".join(text.lower().split())

    train_norm = set(train.text.map(normalize))
    val_norm = set(val.text.map(normalize))
    return {
        "labeled_rows": len(df),
        "unique_ids": int(df.id.nunique()),
        "missing_values": int(df.isna().sum().sum()),
        "repeated_exact_texts": int(df.text.duplicated().sum()),
        "repeated_case_whitespace_normalized_texts": int(df.text.map(normalize).duplicated().sum()),
        "exact_text_overlap_between_splits": len(set(train.text) & set(val.text)),
        "normalized_text_overlap_between_splits": len(train_norm & val_norm),
        "near_duplicate_check": "Not performed; source/domain/generator metadata are unavailable.",
        "train_label_counts": {str(k): int(v) for k, v in train.label.value_counts().items()},
        "validation_label_counts": {str(k): int(v) for k, v in val.label.value_counts().items()},
    }


def fit_and_evaluate(train, val):
    model = make_model()
    # fit() sees only the 8,000-row training partition.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        start = time.perf_counter()
        model.fit(train.text, train.label)
        fit_seconds = time.perf_counter() - start
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError("Logistic regression did not converge: " + convergence[0])
    ai_column = int(np.flatnonzero(model.classes_ == 1)[0])
    train_prob = model.predict_proba(train.text)[:, ai_column]
    val_prob = model.predict_proba(val.text)[:, ai_column]
    for prob in (train_prob, val_prob):
        if not np.isfinite(prob).all() or not ((prob >= 0) & (prob <= 1)).all():
            raise ValueError("Invalid predicted probabilities.")
    metrics = {
        "model": "Word TF-IDF + Logistic Regression",
        "seed": SEED,
        "training_rows": len(train),
        "validation_rows": len(val),
        "training_auroc": float(roc_auc_score(train.label, train_prob)),
        "validation_auroc": float(roc_auc_score(val.label, val_prob)),
        "training_time_seconds": fit_seconds,
        "timing_scope": "Pipeline.fit: TF-IDF fitting/transformation and logistic regression fitting; excludes CSV reading and evaluation",
        "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
        "lr_iterations": int(model.named_steps["lr"].n_iter_[0]),
        "convergence_warning": False,
        "decision_threshold": 0.5,
        "validation_confusion_matrix_labels_0_1": confusion_matrix(val.label, val_prob >= 0.5, labels=[0, 1]).tolist(),
    }
    return model, metrics, train_prob, val_prob


def prediction_frame(frame, prob):
    out = frame[["id", "label"]].rename(columns={"label": "true_label"}).copy()
    out["predicted_ai_probability"] = prob
    out["predicted_label"] = (prob >= 0.5).astype(int)
    out["word_count"] = frame.text.str.split().str.len().to_numpy()
    out["error_type"] = np.select(
        [(out.true_label == 0) & (out.predicted_label == 1),
         (out.true_label == 1) & (out.predicted_label == 0)],
        ["FP", "FN"], default="correct",
    )
    return out


def save_outputs(df, train, val, model, metrics, train_prob, val_prob, output_dir=ROOT):
    output_dir = Path(output_dir)
    results = output_dir / "results"
    models = output_dir / "models"
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    prediction_frame(train, train_prob).to_csv(results / "training_predictions.csv", index=False)
    prediction_frame(val, val_prob).to_csv(results / "validation_predictions.csv", index=False)
    pd.DataFrame([{
        "model": metrics["model"],
        "training_auroc": metrics["training_auroc"],
        "validation_auroc": metrics["validation_auroc"],
        "training_time_seconds": metrics["training_time_seconds"],
    }]).to_csv(results / "baseline_metrics.csv", index=False)
    (results / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (results / "data_audit.json").write_text(json.dumps(audit_data(df, train, val), indent=2) + "\n")
    params = model.get_params()
    config = {k: v for k, v in params.items() if "__" in k}
    config = {k: str(v) if isinstance(v, type) else v for k, v in config.items()}
    (output_dir / "baseline_config.json").write_text(json.dumps(config, indent=2) + "\n")
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
    }
    (results / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    joblib.dump(model, models / "baseline.joblib", compress=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    args = parser.parse_args()
    df = load_data(args.train)
    train, val = load_split(df)
    model, metrics, train_prob, val_prob = fit_and_evaluate(train, val)
    save_outputs(df, train, val, model, metrics, train_prob, val_prob)
    print(pd.read_csv(ROOT / "results" / "baseline_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
