"""Train on all of train.csv and write a Kaggle submission.

    python src/predict.py --model tfidf_logreg
"""
import argparse

import pandas as pd

from data import ROOT, load_test, load_train
from models import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tfidf_logreg", choices=MODELS)
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    train, test = load_train(), load_test()
    model = MODELS[args.model]()
    model.fit(train["text"], train["label"])

    sub = pd.DataFrame({"id": test["id"],
                        "label": model.predict_proba(test["text"])[:, 1]})

    # Format checks from the competition rules
    assert len(sub) == 3000, f"expected 3000 rows, got {len(sub)}"
    assert sub["id"].is_unique
    assert sub["label"].between(0, 1).all()

    out = ROOT / args.out
    sub.to_csv(out, index=False)
    print(f"Wrote {out.relative_to(ROOT)} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
