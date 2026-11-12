#!/usr/bin/env python3
"""Write the Kaggle ``submission.csv`` from a trained model artifact.

Each test row gets ``P(AI-generated)`` in the ``label`` column, matching
``data/sample_submission.csv`` (``id,label``).

Example
-------
    python predict.py --test data/test.csv --model model.joblib --out submission.csv

Hard 0/1 labels instead of probabilities:

    python predict.py --test data/test.csv --model model.joblib \
        --out submission.csv --hard --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline import (
    DEFAULT_BATCH_SIZE,
    EMBED_MODEL,
    default_cache_path,
    load_embeddings,
    predict_scores,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate submission.csv for the Human-or-AI competition"
    )
    parser.add_argument("--test", type=Path, required=True, help="Path to test.csv")
    parser.add_argument(
        "--model", type=Path, default=Path("model.joblib"), help="Model artifact from train.py"
    )
    parser.add_argument("--id-col", default="id", help="ID column (default: id)")
    parser.add_argument("--text-col", default="text", help="Text column (default: text)")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Embedding cache (.npy). Defaults to <test-stem>_embeddinggemma2_270m.npy",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore an existing embedding cache"
    )
    parser.add_argument(
        "--no-cache-write", action="store_true", help="Do not write a new embedding cache"
    )
    parser.add_argument("--embed-model", default=None, help="Override the model's embedding model")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--out", type=Path, default=Path("submission.csv"), help="Output CSV path"
    )
    parser.add_argument(
        "--hard", action="store_true", help="Write hard 0/1 labels instead of probabilities"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="P(AI) threshold for --hard (default: 0.5)"
    )
    parser.add_argument(
        "--summary-out", type=Path, default=None, help="Optional JSON summary of the predictions"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.test.exists():
        raise SystemExit(f"test file not found: {args.test}")
    if not args.model.exists():
        raise SystemExit(
            f"model artifact not found: {args.model} (run `python train.py ...` first)"
        )

    bundle = joblib.load(args.model)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    embed_model = args.embed_model or (
        bundle.get("embed_model", EMBED_MODEL) if isinstance(bundle, dict) else EMBED_MODEL
    )

    frame = pd.read_csv(args.test)
    if args.id_col not in frame.columns:
        raise SystemExit(
            f"column {args.id_col!r} not in {args.test} (columns: {list(frame.columns)})"
        )

    cache_path = args.embeddings or default_cache_path(args.test)
    X = load_embeddings(
        args.test,
        cache_path=cache_path,
        text_col=args.text_col,
        embed_model=embed_model,
        batch_size=args.batch_size,
        write_cache=not args.no_cache_write,
        use_cache=not args.no_cache,
    )
    if len(X) != len(frame):
        raise SystemExit(
            f"embedding/row mismatch: {len(X)} embeddings vs {len(frame)} test rows"
        )

    scores = predict_scores(model, X)
    if args.hard:
        values = (scores >= args.threshold).astype(int)
    else:
        values = scores

    submission = pd.DataFrame({args.id_col: frame[args.id_col].to_numpy(), "label": values})
    if submission["id"].duplicated().any():
        raise SystemExit("duplicate ids in test file")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)

    print(
        f"wrote {args.out} ({len(submission)} rows) | "
        f"mean P(AI)={float(scores.mean()):.4f} | predicted AI rate="
        f"{(scores >= 0.5).mean():.4f}"
    )

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(
                {
                    "test_file": str(args.test),
                    "model": str(args.model),
                    "rows": int(len(submission)),
                    "mean_score": float(scores.mean()),
                    "predicted_ai_rate_at_0.5": float((scores >= 0.5).mean()),
                    "hard_labels": bool(args.hard),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
