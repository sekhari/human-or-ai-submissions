"""Write the Kaggle submission with the final Q1 model (Model A).

Model A = word 1-2-gram TF-IDF + logistic regression, with the settings chosen by
5-fold CV in Q1 (C=30, ngram_range=(1, 2), min_df=5). After model selection, the
competition rules allow retraining on all 10,000 labeled rows, so we do that here
(no validation holdout is needed at this point; no test labels are used).

Usage:
    python predict.py --train data/train.csv --test data/test.csv --out submission.csv
"""
import argparse

import numpy as np
import pandas as pd

from common import SEED
from q1_model_comparison import build_models

FINAL_PARAMS = {"vec__ngram_range": (1, 2), "vec__min_df": 5, "clf__C": 30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    np.random.seed(SEED)  # liblinear is deterministic here; seeded for safety
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    # Same pipeline object Q1 evaluated, with the CV-selected settings.
    _, model, _ = build_models()["A_word_tfidf_lr"]
    model.set_params(**FINAL_PARAMS)
    model.fit(train["text"].values, train["label"].values)

    # Column 1 of predict_proba = P(AI). Kaggle expects one probability per test id.
    p_ai = model.predict_proba(test["text"].values)[:, 1]
    pd.DataFrame({"id": test["id"], "label": p_ai}).to_csv(args.out, index=False)
    print(f"wrote {len(test)} rows to {args.out}  (mean P(AI) = {p_ai.mean():.3f})")


if __name__ == "__main__":
    main()
