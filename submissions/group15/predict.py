"""Predict test.csv with the final A+C model and write submission.csv.

Run from submissions/group15/ after train.py:  python predict.py

Each model's test scores are turned into ranks divided by the number of test passages,
and the final score is the equal-weight average of the two ranks. This is not a
calibrated probability, but AUROC only depends on the ordering.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import get_scores, load_data
import experiments.model_c_style  # noqa: F401  (needed to unpickle Model C's feature function)

MODEL_DIR = os.path.join(HERE, "models")


def to_rank(scores):
    return rankdata(scores) / len(scores)


def main():
    _, _, test = load_data(os.path.join(HERE, "data"))

    ranks = []
    for name in ["model_a", "model_c"]:
        model = joblib.load(os.path.join(MODEL_DIR, f"{name}.joblib"))
        ranks.append(to_rank(get_scores(model, test["text"])))
        print(f"{name}: scored {len(test)} test passages")

    final = np.mean(ranks, axis=0)
    out = os.path.join(HERE, "submission.csv")
    pd.DataFrame({"id": test["id"], "label": final}).to_csv(out, index=False)
    print(f"saved submission.csv ({len(final)} rows)")


if __name__ == "__main__":
    main()
