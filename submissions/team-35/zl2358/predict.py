"""Generate a Kaggle submission using a locally trained Human-or-AI model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    team_dir = Path(__file__).resolve().parent
    parser.add_argument("--test", type=Path, default=team_dir / "data" / "test.csv")
    parser.add_argument("--model", type=Path, default=team_dir / "data" / "q1" / "model.joblib")
    parser.add_argument("--out", type=Path, default=team_dir / "data" / "submission.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="id")
    args = parser.parse_args()

    frame = pd.read_csv(args.test)
    missing = {args.text_column, args.id_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {list(frame.columns)}")
    if frame[[args.text_column, args.id_column]].isna().any().any():
        raise ValueError("Text and ID columns must not contain missing values")
    if frame[args.id_column].duplicated().any():
        raise ValueError("Test IDs must be unique")

    model = joblib.load(args.model)  # Only load a model created locally by train.py.
    if list(model.classes_) != [0, 1]:
        raise ValueError(f"Unexpected model classes: {model.classes_}")
    probability = model.predict_proba(frame[args.text_column].astype(str))[:, 1]
    result = pd.DataFrame({"id": frame[args.id_column], "label": probability})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Wrote {len(result)} probabilities to {args.out}")


if __name__ == "__main__":
    main()
