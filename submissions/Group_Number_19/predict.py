"""Fit the LightGBM stack (style features + TF-IDF LR score) on all of train.csv
and write submission.csv with P(label = 1) for every row of test.csv.

This is the same recipe that lgbm_stack.py cross-validates, applied to the full
training set:
  1. Fit TF-IDF on the train text. LR scores for the train rows are out-of-fold
     (inner 5-fold); test rows are scored by an LR fit on all train rows.
  2. Fit LightGBM on [style features + LR score], early-stopping on a 10% holdout
     of the train rows, and predict the test rows.

--strip-layout removes newlines/markdown from the inputs, as in lgbm_stack.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline_tfidf import build_features
from lgbm_stack import fit_lgbm, lr_scores
from style_features import style_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("../../../data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("../../../data/test.csv"))
    parser.add_argument("--C", type=float, default=30.0, help="LR inverse L2 strength")
    parser.add_argument("--strip-layout", action="store_true",
                        help="remove newlines/markdown from the inputs")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    y = train["label"].to_numpy()

    layout = not args.strip_layout
    features = build_features(0, layout=layout)
    X_tr = features.fit_transform(train["text"])
    X_te = features.transform(test["text"])
    lr_tr, lr_te = lr_scores(X_tr, y, X_te, args.C)

    S_tr = style_features(train["text"], layout=layout).assign(lr_score=lr_tr)
    S_te = style_features(test["text"], layout=layout).assign(lr_score=lr_te)
    pred, model = fit_lgbm(S_tr, y, S_te)

    submission = pd.DataFrame({"id": test["id"], "label": pred})
    submission.to_csv(args.out, index=False)
    print(f"wrote {len(submission)} predictions to {args.out} "
          f"(LightGBM best iteration {model.best_iteration_}, "
          f"{(pred > 0.5).mean():.1%} predicted label 1)")


if __name__ == "__main__":
    main()
