from __future__ import annotations

import argparse
from pathlib import Path

import json
import time
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from src.data import SEED, TEAM_FOLDER, get_train_val, load_data

# TF-IDF/LR baseline
def build_tfidf_lr(seed, C=4.0):
  vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=200_000,   
  )
  
  classifier = LogisticRegression(
    C=C, 
    max_iter=2000, 
    random_state=seed
  )

  return make_pipeline(vectorizer, classifier)

# character n-gram TF-IDF/LR: captures writing style rather than topic
def build_char_lr(seed, C=4.0):
  vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    sublinear_tf=True,
    max_features=300_000,
  )
  classifier = LogisticRegression(C=C, max_iter=2000, random_state=seed)
  return make_pipeline(vectorizer, classifier)

# word TF-IDF/linear SVM
def build_tfidf_svm(seed, C=4.0):
  vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=200_000,
  )
  svm = LinearSVC(C=C, random_state=seed)
  classifier = CalibratedClassifierCV(svm, cv=3)
  return make_pipeline(vectorizer, classifier)

# dict of all our models (add to this)
MODELS = {
  "tfidf_lr": build_tfidf_lr,
  "char_lr": build_char_lr,
  "tfidf_svm": build_tfidf_svm,
}


def main() -> None:
  parser = argparse.ArgumentParser(description="Train Human-or-AI detector")
  parser.add_argument("--train", type=Path, required=True, help="Path to train.csv")
  parser.add_argument("--seed", type=int, default=SEED)
  parser.add_argument("--model-out", type=Path, default=Path("model.joblib"))
  parser.add_argument("--model", default="tfidf_lr", choices=MODELS.keys(), help="which model to train")
  parser.add_argument("--full", action="store_true", help="train on all labeled rows and save the model") 
  args = parser.parse_args()

  # load data and build untrained model
  full_df = load_data(args.train)
  model = MODELS[args.model](args.seed)

  # final mode: train on all 10,000 rows and save
  if args.full:
    model.fit(full_df["text"], full_df["label"])
    joblib.dump(model, args.model_out)
    print(f"Trained {args.model} on {len(full_df)} rows, saved {args.model_out}")
    return

  # validation mode: train on 8,000 rows and score on the 2,000 held out (and records time)
  train_df, val_df = get_train_val(full_df)
  start = time.time()
  model.fit(train_df["text"], train_df["label"])
  train_seconds = time.time() - start

  # predicated probabilites
  train_probs = model.predict_proba(train_df["text"])[:,1]
  val_probs = model.predict_proba(val_df["text"])[:,1]

  # compute AUROC on train and val
  results = {
    "model": args.model,
    "seed": args.seed,
    "train_auroc": round(roc_auc_score(train_df["label"], train_probs), 4),
    "val_auroc": round(roc_auc_score(val_df["label"], val_probs), 4),
    "train_seconds": round(train_seconds, 1),
  }
  print(results)

  # save results to respective json file in results/ 
  results_file = TEAM_FOLDER / "results" / f"{args.model}.json"
  results_file.parent.mkdir(exist_ok=True)

  with open(results_file, "w") as f:
    json.dump(results, f, indent=2)


if __name__ == "__main__":
  main()
