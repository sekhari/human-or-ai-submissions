"""Generate id,label Kaggle probabilities using a locally trained model."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    test = pd.read_csv(args.test, dtype={"id": str})
    submission = pd.read_csv(args.template, dtype={"id": str})
    assert list(submission.columns) == ["id", "label"]
    assert len(test) == len(submission) == 3000
    assert test.id.is_unique and submission.id.is_unique
    assert set(test.id) == set(submission.id)
    model = joblib.load(args.model)  # Only load the model you trained locally.
    probabilities = model.predict_proba(test.text.fillna("").astype(str))[:, list(model.classes_).index(1)]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    submission["label"] = submission.id.map(pd.Series(probabilities, index=test.id))
    assert submission.label.notna().all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print("Wrote", len(submission), "P(AI) predictions:", args.out)


if __name__ == "__main__":
    main()
