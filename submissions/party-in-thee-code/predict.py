"""Final Kaggle pipeline: train on all 10,000 labeled rows and write submission.csv.

Models (all chosen on the shared validation split, see Q1 and Q5):
  newline_char       char 3-5 TF-IDF with newlines kept as "¶" + LinearSVC C=1,
                     3-fold sigmoid calibration on training rows (default, Q5 v4)
  newline_ensemble   newline_char rank-averaged with word 1-2 TF-IDF + LR C=10 (Q5 v5)
  char_svm           the original Q1 selection, calibrated the same way

    # sanity check: fit on the 8,000 training rows, report validation AUROC
    python predict.py --train data/train.csv --validate
    # final: fit on all 10,000 rows, write submission.csv (+ the two alternatives)
    python predict.py --train data/train.csv --test data/test.csv

Output probabilities are P(AI). The ensemble outputs an averaged rank score in
[0, 1], which Kaggle's AUROC treats the same way as a probability.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from baseline import ROOT, SEED, load_data, load_split
from q1 import make_q1_model
from q5_improvement import rank01
from q5_round2 import make_newline_char

DEFAULT = "newline_char"


def calibrated(pipeline):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    return CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=cv, ensemble=False, n_jobs=1)


def fit_predict(train_text, train_label, texts):
    """Fit every candidate once and return P(AI) (or rank score) for each."""
    newline = calibrated(make_newline_char()).fit(train_text, train_label)
    char = calibrated(make_q1_model("char_svm", 1.0)).fit(train_text, train_label)
    word = make_q1_model("word_lr", 10.0).fit(train_text, train_label)
    p_newline = newline.predict_proba(texts)[:, 1]
    p_char = char.predict_proba(texts)[:, 1]
    p_word = word.predict_proba(texts)[:, 1]
    return {
        "newline_char": p_newline,
        "newline_ensemble": (rank01(p_newline) + rank01(p_word)) / 2,
        "char_svm": p_char,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data" / "train.csv"))
    parser.add_argument("--test", default=str(ROOT / "data" / "test.csv"))
    parser.add_argument("--validate", action="store_true", help="fit on 8,000 rows and score validation only")
    parser.add_argument("--model", default=DEFAULT, choices=["newline_char", "newline_ensemble", "char_svm"])
    parser.add_argument("--out", default=str(ROOT / "submission.csv"))
    args = parser.parse_args()

    full = load_data(args.train)
    start = time.perf_counter()
    with threadpool_limits(1):
        if args.validate:
            train, val = load_split(full)
            scores = fit_predict(train.text, train.label, val.text)
            report = {k: round(float(roc_auc_score(val.label, v)), 6) for k, v in scores.items()}
            report["seconds"] = round(time.perf_counter() - start, 1)
            print(json.dumps(report, indent=2))
            out = ROOT / "results" / "final"
            out.mkdir(parents=True, exist_ok=True)
            (out / "validation_check.json").write_text(json.dumps(report, indent=2))
            return

        test = pd.read_csv(args.test, dtype={"id": str})
        if len(test) != 3000 or not test.id.is_unique or test.text.isna().any():
            raise ValueError("Expected the 3,000-row test.csv with unique IDs.")
        scores = fit_predict(full.text, full.label, test.text)

    out = Path(args.out)
    for name, p in scores.items():
        if not np.all((p >= 0) & (p <= 1)):
            raise ValueError(f"{name} produced values outside [0, 1].")
        path = out if name == args.model else out.with_name(f"submission_{name}.csv")
        pd.DataFrame({"id": test.id, "label": p}).to_csv(path, index=False)
        print(f"wrote {path} ({name})")
    print(f"done in {time.perf_counter() - start:.1f}s (trained on {len(full)} rows)")


if __name__ == "__main__":
    main()
