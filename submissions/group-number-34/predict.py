from __future__ import annotations

import argparse
from pathlib import Path
import joblib

from src.data import load_data

def main() -> None:
  parser = argparse.ArgumentParser(description="Write Kaggle submission.csv")
  parser.add_argument("--test", type=Path, required=True, help="Path to test.csv")
  parser.add_argument("--model", type=Path, default=Path("model.joblib"))
  parser.add_argument("--out", type=Path, default=Path("submission.csv"))
  args = parser.parse_args()


  # load trained model and test passages
  model = joblib.load(args.model)
  test_df = load_data(args.test, has_labels=False)

  # predicted probability
  ai_class_index = list(model.classes_).index(1)
  probabilities = model.predict_proba(test_df["text"])[:, ai_class_index]

  # format for kaggle submission
  submission = test_df[["id"]].copy()
  submission["label"] = probabilities
  submission.to_csv(args.out, index=False)

  print(f"Wrote {args.out}: {len(submission)} rows")


if __name__ == "__main__":
  main()
