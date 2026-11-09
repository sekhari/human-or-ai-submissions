# final model's predictions on validation set saved for error analysis.
# usage: python save_val_predictions.py --train data/train.csv

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import SEED, TEAM_FOLDER, get_train_val, load_data
from train import MODELS

def main() -> None:
  parser = argparse.ArgumentParser(description="Save validation predictions")
  parser.add_argument("--train", type=Path, required=True, help="Path to train.csv")
  parser.add_argument("--model", default="word_char_lr", choices=MODELS.keys())
  parser.add_argument("--C", type=float, default=16.0)
  parser.add_argument("--out", type=Path, default=TEAM_FOLDER / "results" / "val_predictions.csv")
  args = parser.parse_args()

  # train on 8,000 samples only
  train_df, val_df = get_train_val(load_data(args.train))
  model = MODELS[args.model](SEED, C=args.C)
  model.fit(train_df["text"], train_df["label"])

  # one row per val passage
  predictions = val_df[["id", "label"]].copy()
  predictions["p_ai"] = model.predict_proba(val_df["text"])[:, 1]
  predictions["text_length"] = val_df["text"].str.split().str.len()

  predictions.to_csv(args.out, index=False)
  print(f"Saved {args.out}: {len(predictions)} rows")
  print(predictions.describe().round(3))


if __name__ == "__main__":
  main()