# trying several C values for each model on the shared validation split 

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.metrics import roc_auc_score

from src.data import SEED, TEAM_FOLDER, get_train_val, load_data
from train import MODELS

C_VALUES = [0.25, 1.0, 4.0, 16.0, 64.0]


def main() -> None:
  parser = argparse.ArgumentParser(description="Tune C for each model")
  parser.add_argument("--train", type=Path, required=True, help="Path to train.csv")
  parser.add_argument("--models", nargs="+", default=["tfidf_lr", "char_lr", "tfidf_svm"])
  args = parser.parse_args()

  train_df, val_df = get_train_val(load_data(args.train))
  all_results = []

  for model_name in args.models:
    for C in C_VALUES:
      model = MODELS[model_name](SEED, C=C)

      start = time.time()
      model.fit(train_df["text"], train_df["label"])
      seconds = time.time() - start

      train_probs = model.predict_proba(train_df["text"])[:, 1]
      val_probs = model.predict_proba(val_df["text"])[:, 1]

      row = {
        "model": model_name,
        "C": C,
        "train_auroc": round(roc_auc_score(train_df["label"], train_probs), 4),
        "val_auroc": round(roc_auc_score(val_df["label"], val_probs), 4),
        "train_seconds": round(seconds, 1),
      }
      all_results.append(row)
      print(row)

  # save every run so the report table comes from real numbers
  with open(TEAM_FOLDER / "results" / "tuning.json", "w") as f:
    json.dump(all_results, f, indent=2)


if __name__ == "__main__":
  main()