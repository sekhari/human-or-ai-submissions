"""Fit one configuration and evaluate on the shared stratified holdout."""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from data_io import digest, labels_from, read_texts, split_rows


def bootstrap_auc(labels, scores, seed):
    """Percentile interval from 1,000 class-stratified validation resamples."""
    rng = np.random.default_rng(seed)
    positive, negative = (np.flatnonzero(labels == value) for value in (1, 0))
    aucs = []
    for _ in range(1000):
        rows = np.concatenate([
            rng.choice(positive, len(positive)),
            rng.choice(negative, len(negative)),
        ])
        aucs.append(roc_auc_score(labels[rows], scores[rows]))
    return np.percentile(aucs, [2.5, 97.5]).tolist()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--model", choices=("linear", "tree", "forest", "boost", "mlp", "deberta"),
                        required=True)
    parser.add_argument("--label-col")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--full", action="store_true", help="Refit on all labeled data after selection")
    size.add_argument("--train-size", type=int, help="Subsample the training partition")
    parser.add_argument("--backbone", default="microsoft/deberta-v3-base")
    parser.add_argument("--revision", default="main", help="Model revision or immutable commit hash")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    if min(args.jobs, args.batch_size, args.accumulate, args.epochs) < 1:
        parser.error("jobs, batch size, accumulation and epochs must be positive")
    if not 4 <= args.max_length <= 512 or not 0 < args.lr:
        parser.error("Use max length from 4 to 512 and a positive learning rate")
    return args


def main():
    args = parse_args()
    frame = read_texts(args.train, args.id_col, args.text_col)
    labels, label_column = labels_from(frame, args.label_col)
    train, valid, eligible = split_rows(frame, labels, args.text_col, args.seed)
    fit = np.arange(len(frame)) if args.full else eligible
    if args.train_size is not None:
        if not 2 <= args.train_size <= len(eligible):
            raise ValueError(f"--train-size must be between 2 and {len(eligible)}")
        if args.train_size < len(eligible):
            fit, _ = train_test_split(
                eligible, train_size=args.train_size, stratify=labels[eligible], random_state=args.seed
            )
    texts = frame[args.text_col]
    folder = args.run_dir
    folder.mkdir(parents=True, exist_ok=False)
    versions = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]
    revision = None
    device_name = "cpu"
    with threadpool_limits(limits=args.jobs):
        start = perf_counter()
        if args.model == "deberta":
            import neural

            neural.torch.set_num_threads(args.jobs)
            model, tokenizer, device, revision = neural.fit(
                texts.iloc[fit], labels[fit], args, folder / "transformer"
            )
            device_name = str(device)
            versions += ["torch", "transformers", "safetensors"]
            if importlib.util.find_spec("sentencepiece") is not None:
                versions.append("sentencepiece")

            def score(rows):
                return neural.scores(model, tokenizer, texts.iloc[rows],
                                     args.max_length, args.batch_size, device)
        else:
            from classical import make_model

            model = make_model(args.model, args.seed, args.jobs)
            model.fit(texts.iloc[fit], labels[fit])
            joblib.dump(model, folder / "model.joblib")

            def score(rows):
                return model.predict_proba(texts.iloc[rows])[:, 1]

        metrics = {"training_and_save_seconds": perf_counter() - start, "fit_rows": len(fit)}
        if not args.full:
            validation_scores = score(valid)
            metrics.update(
                train_auroc=roc_auc_score(labels[fit], score(fit)),
                validation_auroc=roc_auc_score(labels[valid], validation_scores),
                validation_ci95=bootstrap_auc(labels[valid], validation_scores, args.seed),
                validation_rows=len(valid),
            )
            pd.DataFrame({
                "row_index": valid,
                "id": frame[args.id_col].iloc[valid].to_numpy(),
                "label": labels[valid],
                "p_ai": validation_scores,
            }).to_csv(folder / "validation_predictions.csv", index=False)

    split = np.full(len(frame), "train", dtype=object)
    split[valid] = "validation"
    pd.DataFrame({
        "row_index": np.arange(len(frame)), "id": frame[args.id_col],
        "label": labels, "split": split,
        "used_to_fit": np.isin(np.arange(len(frame)), fit),
    }).to_csv(folder / "split.csv", index=False)
    run = {
        "config": vars(args),
        "label_column": label_column,
        "train_sha256": digest(args.train),
        "cross_split_training_duplicates": len(train) - len(eligible),
        "device": device_name,
        "resolved_revision": revision,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in versions},
        "metrics": metrics,
    }
    (folder / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    print(json.dumps(metrics, indent=2))
    if len(train) != len(eligible) and not args.full:
        print(f"Excluded {len(train) - len(eligible)} training rows matching validation text.")


if __name__ == "__main__":
    main()
