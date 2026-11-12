#!/usr/bin/env python3
"""Train and evaluate the embedding + SVC pipeline.

The labelled validation protocol is a stratified holdout of ``--train``:
the holdout rows are scored (accuracy / F1 / ROC-AUC) and then the final model
is refitted on all labelled rows before being saved.

The model written by this script is therefore **always trained on every
labelled row**, including when the holdout evaluation is skipped.

Examples
--------
Train, report holdout metrics, and save the final model (fit on all 10,000 rows):

    python train.py --train data/train.csv --holdout 2000 \
        --model-out model.joblib --metrics-out metrics.json

Skip evaluation entirely and fit on all labelled rows (what produced the
leaderboard submission):

    python train.py --train data/train.csv --holdout 0 --model-out model.joblib

Evaluate only (skip the final refit; faster when tuning):

    python train.py --train data/train.csv --holdout 2000 --skip-final-fit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pipeline import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_C,
    DEFAULT_KERNEL,
    EMBED_MODEL,
    RANDOM_SEED,
    build_model,
    default_cache_path,
    load_embeddings,
    predict_scores,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the Human-or-AI embedding + SVC classifier"
    )
    parser.add_argument("--train", type=Path, required=True, help="Path to train.csv")
    parser.add_argument("--id-col", default="id", help="ID column (default: id)")
    parser.add_argument("--text-col", default="text", help="Text column (default: text)")
    parser.add_argument("--label-col", default="label", help="Label column (default: label)")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Embedding cache (.npy). Defaults to <train-stem>_embeddinggemma2_270m.npy",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore an existing embedding cache"
    )
    parser.add_argument(
        "--no-cache-write", action="store_true", help="Do not write a new embedding cache"
    )
    parser.add_argument(
        "--embed-model", default=EMBED_MODEL, help=f"Ollama embedding model (default: {EMBED_MODEL})"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--holdout",
        type=int,
        default=2000,
        help="Held-out rows for evaluation; 0 disables evaluation (default: 2000)",
    )
    parser.add_argument("--C", type=float, default=DEFAULT_C, help="SVC C (default: 10.0)")
    parser.add_argument(
        "--kernel", default=DEFAULT_KERNEL, help=f"SVC kernel (default: {DEFAULT_KERNEL})"
    )
    parser.add_argument(
        "--skip-final-fit",
        action="store_true",
        help="Only run the holdout evaluation; do not fit/save the final model",
    )
    parser.add_argument(
        "--model-out", type=Path, default=Path("model.joblib"), help="Model artifact path"
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("metrics.json"),
        help="Where to write the metrics JSON",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.train.exists():
        raise SystemExit(f"training file not found: {args.train}")

    frame = pd.read_csv(args.train)
    for column in (args.text_col, args.label_col):
        if column not in frame.columns:
            raise SystemExit(
                f"column {column!r} not in {args.train} (columns: {list(frame.columns)})"
            )

    labels = frame[args.label_col].to_numpy()
    classes = sorted(np.unique(labels).tolist())
    if classes != [0, 1]:
        raise SystemExit(f"expected binary labels [0, 1], found {classes}")

    if args.holdout < 0:
        raise SystemExit("--holdout must be >= 0 (0 disables evaluation and trains on all rows)")
    if args.holdout >= len(frame):
        raise SystemExit(
            f"--holdout {args.holdout} must be smaller than the {len(frame)} training rows "
            "(use --holdout 0 to skip evaluation)"
        )
    if args.skip_final_fit and args.holdout <= 0:
        raise SystemExit(
            "--skip-final-fit with --holdout 0 would train nothing; "
            "drop one of the two flags"
        )

    cache_path = args.embeddings or default_cache_path(args.train)
    X = load_embeddings(
        args.train,
        cache_path=cache_path,
        text_col=args.text_col,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
        write_cache=not args.no_cache_write,
        use_cache=not args.no_cache,
    )
    if len(X) != len(frame):
        raise SystemExit(
            f"embedding/label mismatch: {len(X)} embeddings vs {len(frame)} labelled rows"
        )

    print(f"data: {X.shape[0]} rows x {X.shape[1]} dims, labels {np.bincount(labels).tolist()}")

    report: dict = {
        "config": {
            "train_file": str(args.train),
            "n_rows": int(len(frame)),
            "embedding_model": args.embed_model,
            "embedding_dims": int(X.shape[1]),
            "seed": args.seed,
            "holdout": args.holdout,
            "model": {"type": "SVC", "kernel": args.kernel, "C": args.C, "probability": True},
            "sklearn_version": sklearn.__version__,
        }
    }

    holdout_metrics = None
    if args.holdout == 0:
        print(f"holdout disabled (--holdout 0): fitting on all {len(frame)} labelled rows")
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            labels,
            test_size=args.holdout,
            stratify=labels,
            random_state=args.seed,
        )
        start = time.time()
        eval_model = build_model(C=args.C, kernel=args.kernel, seed=args.seed).fit(X_tr, y_tr)
        fit_seconds = time.time() - start

        scores = predict_scores(eval_model, X_val)
        predictions = (scores >= 0.5).astype(int)
        holdout_metrics = {
            "n_train": int(len(X_tr)),
            "n_holdout": int(len(X_val)),
            "accuracy": float(accuracy_score(y_val, predictions)),
            "f1_ai": float(f1_score(y_val, predictions, pos_label=1)),
            "roc_auc": float(roc_auc_score(y_val, scores)),
            "fit_seconds": round(fit_seconds, 1),
        }
        report["holdout"] = holdout_metrics

        print(
            f"holdout ({len(X_val)} rows, seed={args.seed}): "
            f"accuracy={holdout_metrics['accuracy']:.4f} "
            f"f1(AI)={holdout_metrics['f1_ai']:.4f} "
            f"roc_auc={holdout_metrics['roc_auc']:.4f} "
            f"[{fit_seconds:.1f}s]"
        )

    if args.skip_final_fit:
        report["final_model"] = None
        _write_json(args.metrics_out, report)
        print(f"skipped final fit; metrics -> {args.metrics_out}")
        return 0

    start = time.time()
    model = build_model(C=args.C, kernel=args.kernel, seed=args.seed).fit(X, labels)
    fit_seconds = time.time() - start

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "embed_model": args.embed_model,
        "seed": args.seed,
        "kernel": args.kernel,
        "C": args.C,
        "text_col": args.text_col,
        "id_col": args.id_col,
        "label_col": args.label_col,
        "train_rows": int(len(frame)),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, args.model_out)

    report["final_model"] = {
        "path": str(args.model_out),
        "trained_on": int(len(frame)),
        "trained_on_full_data": True,
        "fit_seconds": round(fit_seconds, 1),
    }
    _write_json(args.metrics_out, report)

    print(f"final model trained on {len(frame)} rows in {fit_seconds:.1f}s -> {args.model_out}")
    print(f"metrics -> {args.metrics_out}")
    return 0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
