# train one model on increasing amounts of data and plot train/val AUROC

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import SEED, TEAM_FOLDER, get_train_val, load_data
from train import MODELS

TRAIN_SIZES = [500, 1000, 2000, 5000, 8000]


def main() -> None:
  parser = argparse.ArgumentParser(description="Learning curve for one model")
  parser.add_argument("--train", type=Path, required=True, help="Path to train.csv")
  parser.add_argument("--model", default="word_char_lr", choices=MODELS.keys())
  parser.add_argument("--C", type=float, default=16.0)
  args = parser.parse_args()

  train_df, val_df = get_train_val(load_data(args.train))
  results = []

  for size in TRAIN_SIZES:
    # take a stratified sample of `size` rows
    if size < len(train_df):
      subset, _ = train_test_split(
        train_df, train_size=size, stratify=train_df["label"], random_state=SEED
      )
    else:
      subset = train_df

    model = MODELS[args.model](SEED, C=args.C)
    start = time.time()
    model.fit(subset["text"], subset["label"])
    seconds = time.time() - start

    train_probs = model.predict_proba(subset["text"])[:, 1]
    val_probs = model.predict_proba(val_df["text"])[:, 1]

    row = {
      "train_rows": size,
      "train_auroc": round(roc_auc_score(subset["label"], train_probs), 4),
      "val_auroc": round(roc_auc_score(val_df["label"], val_probs), 4),
      "train_seconds": round(seconds, 1),
    }
    results.append(row)
    print(row)

  # save the numbers
  results_dir = TEAM_FOLDER / "results"
  with open(results_dir / "learning_curve.json", "w") as f:
    json.dump({"model": args.model, "C": args.C, "rows": results}, f, indent=2)

  # plot train and val AUROC vs training size
  sizes = [r["train_rows"] for r in results]
  plt.figure(figsize=(6, 4))
  plt.plot(sizes, [r["train_auroc"] for r in results], marker="o", label="Train")
  plt.plot(sizes, [r["val_auroc"] for r in results], marker="o", label="Validation")
  plt.xscale("log")
  plt.xticks(sizes, [str(s) for s in sizes])
  plt.minorticks_off()
  plt.xlabel("Training rows")
  plt.ylabel("AUROC")
  plt.title(f"Learning curve: {args.model} (C={args.C})")
  plt.legend()
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig(results_dir / "learning_curve.png", dpi=150)
  print(f"Saved {results_dir / 'learning_curve.png'}")


if __name__ == "__main__":
  main()