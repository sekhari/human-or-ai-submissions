import argparse
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TEAM_DIR = BASE_DIR.parent


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions using the saved Round 2 SVM."
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=TEAM_DIR / "test.csv"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=BASE_DIR / "svm_results" / "svm_model.joblib"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=BASE_DIR / "svm_results" / "submission_svm_round2.csv"
    )
    args = parser.parse_args()

    if not args.model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model_path}. "
            "Run train.py first."
        )

    test = pd.read_csv(args.test_csv)

    if not {"id", "text"}.issubset(test.columns):
        raise ValueError("Test CSV must contain id and text columns.")

    if test[["id", "text"]].isna().any().any():
        raise ValueError("Test data contains missing values.")

    if not test["id"].is_unique:
        raise ValueError("Test IDs must be unique.")

    if not test["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("All text values must be strings.")

    model = joblib.load(args.model_path)

    # Locate the AI class explicitly instead of assuming its position.
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The saved model does not contain AI class 1.")

    ai_class_index = classes.index(1)
    probabilities = model.predict_proba(test["text"])[:, ai_class_index]

    submission = pd.DataFrame({
        "id": test["id"].to_numpy(),
        "label": probabilities
    })

    if (
        submission["label"].isna().any()
        or not submission["label"].between(0, 1).all()
    ):
        raise ValueError("Predicted probabilities are invalid.")

    # Preserve the input test ID order.
    assert submission["id"].equals(test["id"].reset_index(drop=True))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False)

    print(f"Predictions saved: {len(submission):,} rows")
    print(f"Output file: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
