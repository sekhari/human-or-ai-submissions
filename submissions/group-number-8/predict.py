"""Write a Kaggle submission from a model saved by train.py --mode submit."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from pipeline.data import load_split
from pipeline.features import flatten_text, format_features
from pipeline.models import rank_average, submission_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Kaggle submission.csv")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("model.joblib"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    test_df = load_split(args.test, labeled=False)
    tokenizer = Tokenizer.from_str(bundle["tokenizer_json"])
    flat = flatten_text(test_df["text"]).tolist()
    tokens = [encoding.tokens for encoding in tokenizer.encode_batch(flat)]
    word = bundle["vectorizer"].transform(tokens)
    layout = format_features(test_df["text"])
    sgd_scores = np.asarray(bundle["sgd"].decision_function(word), dtype=np.float64)
    gbm_scores = np.asarray(bundle["gbm"].decision_function(layout), dtype=np.float64)
    labels = rank_average([sgd_scores, gbm_scores])
    frame = submission_frame(test_df["id"], labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
