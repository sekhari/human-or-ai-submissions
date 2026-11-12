"""Train the final model (A+C) on all 10,000 labeled passages.

Run from submissions/group15/:  python train.py
Then:                           python predict.py

Model A: word 1-2 gram TF-IDF (case and punctuation kept) + logistic regression, C=10
Model C: 34 style features + random forest (300 trees, max_depth=None, min_samples_leaf=2)
Both settings were chosen on the fixed 2,000-row validation set (see report Q1.3, Q1.4).
The fitted models are saved to models/ (ignored by git).
"""

import os
import sys
import time

import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import load_data
from experiments.model_a import build_model as build_model_a
from experiments.model_c_style import make_model as build_model_c

MODEL_DIR = os.path.join(HERE, "models")


def main():
    train, _, _ = load_data(os.path.join(HERE, "data"))
    print(f"training on {len(train)} labeled passages")
    os.makedirs(MODEL_DIR, exist_ok=True)

    models = {"model_a": build_model_a(), "model_c": build_model_c()}
    for name, model in models.items():
        start = time.time()
        model.fit(train["text"], train["label"])
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        print(f"{name}: fitted in {time.time() - start:.1f}s -> models/{name}.joblib")


if __name__ == "__main__":
    main()
