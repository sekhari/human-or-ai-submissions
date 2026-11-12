"""Train/evaluate Alma's selected RF, or refit on all labeled rows."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from rf_features import StyleFeatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Refit using all labeled data; no validation score.")
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/reproduced"))
    args = parser.parse_args()
    data = pd.read_csv(args.train, dtype={"id": str})
    assert {"id", "text", "label"}.issubset(data.columns)
    assert len(data) == 10000 and data.id.is_unique
    assert data.label.notna().all() and set(data.label) == {0, 1}
    X = data.text.fillna("").astype(str)
    y = data.label.to_numpy()
    model = Pipeline([
        ("features", StyleFeatures()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=None,
             min_samples_leaf=2, max_features="sqrt", random_state=args.seed, n_jobs=-1)),
    ])
    metadata = {"seed": args.seed, "features": 21, "positive_class": 1,
                "mode": "full" if args.full else "validation",
                "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest()}
    if args.full:
        model.fit(X, y)
        metadata["training_rows"] = len(data)
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(data)), test_size=2000, random_state=args.seed, stratify=y)
        model.fit(X.iloc[train_idx], y[train_idx])
        positive = list(model.classes_).index(1)
        train_prob = model.predict_proba(X.iloc[train_idx])[:, positive]
        val_prob = model.predict_proba(X.iloc[val_idx])[:, positive]
        metadata.update(training_rows=8000, validation_rows=2000,
                        train_auroc=roc_auc_score(y[train_idx], train_prob),
                        val_auroc=roc_auc_score(y[val_idx], val_prob))
        args.results_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": data.iloc[val_idx].id.to_numpy(),
                      "y_true": y[val_idx], "p_ai": val_prob}).to_csv(
            args.results_dir / "validation_predictions.csv", index=False)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    (args.results_dir / ("full_fit.json" if args.full else "validation_metrics.json")).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("Model:", args.model_out)


if __name__ == "__main__":
    main()
