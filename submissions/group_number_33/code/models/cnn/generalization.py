"""Compare raw and cleaned TextCNNs on random and held-out topic splits.

Topic groups approximate domain shift; they do not identify hidden generators.
Only ``prepare`` fits the label-independent grouping representation on all labeled
texts. Every predictive model fits its vocabulary and weights on its training fold.
Neither test.csv nor unlabeled.csv is used to construct validation splits.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import normalize

from .neural import (
    CNNConfig,
    load_predictor,
    refit,
    train_with_early_stopping,
    validate_vector_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_NAMES = ("cnn_raw", "cnn_clean")


def read_data(path, labeled=True):
    frame = pd.read_csv(path)
    required = ["id", "text"] + (["label"] if labeled else [])
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{path} must contain {required}")
    if frame[required].isna().any().any() or not frame.id.is_unique:
        raise ValueError("Required values must be present and IDs must be unique")
    if not frame.text.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Every passage must be a nonempty string")
    if labeled and set(frame.label.unique()) != {0, 1}:
        raise ValueError("Expected both binary labels, 0 and 1")
    return frame


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assign_splits(frame, groups, folds=5, seed=12):
    """Keep the existing random split and hold every topic out exactly once."""
    groups = np.asarray(groups)
    if len(groups) != len(frame) or len(np.unique(groups)) < folds:
        raise ValueError("Need one group per passage and at least one group per fold")
    indices = np.arange(len(frame))
    _, validation = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=frame.label
    )
    random_split = np.full(len(frame), "train", dtype=object)
    random_split[validation] = "validation"
    topic_fold = np.full(len(frame), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train, val) in enumerate(splitter.split(indices, frame.label, groups)):
        if frame.label.iloc[train].nunique() != 2 or frame.label.iloc[val].nunique() != 2:
            raise ValueError("Each fold needs both labels; try fewer topics or folds")
        topic_fold[val] = fold
    return pd.DataFrame({
        "id": frame.id.to_numpy(),
        "random_split": random_split,
        "topic_group": groups,
        "topic_fold": topic_fold,
    })


def prepare(args):
    frame = read_data(args.train)
    if not 2 <= args.folds <= args.topics < len(frame):
        raise ValueError("Require 2 <= folds <= topics < number of passages")
    print("Building topic proxies from labeled texts only; labels are not clustering features.", flush=True)
    vectorizer = TfidfVectorizer(
        stop_words="english", min_df=2, max_df=0.95,
        max_features=20000, sublinear_tf=True, dtype=np.float32,
    )
    features = vectorizer.fit_transform(frame.text)
    dimensions = min(64, features.shape[0] - 1, features.shape[1] - 1)
    if dimensions < 2:
        raise ValueError("Not enough vocabulary to construct topic groups")
    embedding = TruncatedSVD(n_components=dimensions, random_state=args.seed).fit_transform(features)
    groups = MiniBatchKMeans(
        n_clusters=args.topics, n_init=10, batch_size=1024, random_state=args.seed,
    ).fit_predict(normalize(embedding))
    splits = assign_splits(frame, groups, args.folds, args.seed)
    vocabulary = vectorizer.get_feature_names_out()
    topics = []
    for group in np.unique(groups):
        mask = groups == group
        weights = np.asarray(features[mask].mean(axis=0)).ravel()
        topics.append({
            "topic_group": int(group), "rows": int(mask.sum()),
            "human_rows": int((frame.label.to_numpy()[mask] == 0).sum()),
            "ai_rows": int((frame.label.to_numpy()[mask] == 1).sum()),
            "keywords": ", ".join(vocabulary[np.argsort(weights)[-8:][::-1]]),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.output_dir / "splits.csv", index=False)
    pd.DataFrame(topics).to_csv(args.output_dir / "topics.csv", index=False)
    metadata = {
        "train_sha256": file_hash(args.train), "seed": args.seed,
        "folds": args.folds, "requested_topics": args.topics,
        "actual_topics": len(topics), "sklearn_version": sklearn.__version__,
        "grouping": "word TF-IDF (English stop words), SVD, normalized MiniBatchKMeans",
        "limitation": "Unsupervised topic proxies, not verified domains or AI generators",
    }
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(splits.groupby("topic_fold").size().to_string(), flush=True)
    print(f"Saved split IDs and topic descriptions to {args.output_dir}")


def load_splits(train_path, output_dir):
    frame = read_data(train_path)
    metadata = json.loads((output_dir / "split_metadata.json").read_text())
    if file_hash(train_path) != metadata["train_sha256"]:
        raise ValueError("Training file changed since prepare; regenerate splits explicitly")
    if metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError("Use the scikit-learn version recorded in split_metadata.json")
    splits = pd.read_csv(output_dir / "splits.csv")
    if not splits.id.is_unique or set(splits.id) != set(frame.id):
        raise ValueError("Split IDs do not match training IDs")
    splits = splits.set_index("id").loc[frame.id].reset_index()
    if splits.isna().any().any() or set(splits.random_split) != {"train", "validation"}:
        raise ValueError("Invalid split assignments")
    if set(splits.topic_fold) != set(range(metadata["folds"])):
        raise ValueError("Invalid topic fold assignments")
    if (splits.groupby("topic_group").topic_fold.nunique() != 1).any():
        raise ValueError("A topic appears in multiple held-out folds")
    return frame, splits, metadata


def topic_scores(labels, probabilities, groups, minimum_per_class=20):
    data = pd.DataFrame({"label": np.asarray(labels), "probability": probabilities, "topic_group": groups})
    rows = []
    for group, part in data.groupby("topic_group"):
        counts = part.label.value_counts()
        eligible = counts.get(0, 0) >= minimum_per_class and counts.get(1, 0) >= minimum_per_class
        rows.append({
            "topic_group": group, "rows": len(part),
            "human_rows": int(counts.get(0, 0)), "ai_rows": int(counts.get(1, 0)),
            "auroc": roc_auc_score(part.label, part.probability) if len(counts) == 2 else np.nan,
            "eligible_for_summary": eligible,
        })
    return pd.DataFrame(rows)


def config_from_args(args, embedding_mode="random"):
    return CNNConfig(
        max_vocab=args.max_vocab, max_length=args.max_length,
        embedding_dim=args.embedding_dim, embedding_mode=embedding_mode,
        embedding_path=str(args.embeddings.resolve()) if args.embeddings else None,
        embedding_lowercase_fallback=args.lowercase_embedding_fallback,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        learning_rate=args.learning_rate, seed=args.seed, device=args.device,
    )


def write_results(destination, metrics, predictions, group_metrics):
    results = pd.DataFrame(metrics)
    results.to_csv(destination / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(destination / "validation_predictions.csv", index=False)
    pd.concat(group_metrics, ignore_index=True).to_csv(destination / "topic_metrics.csv", index=False)
    summary = results.groupby(["model", "embedding_mode", "protocol"]).agg(
        mean_auroc=("auroc", "mean"), std_auroc=("auroc", "std"),
        minimum_fold_auroc=("auroc", "min"), evaluations=("auroc", "size"),
        median_selected_epoch=("selected_epoch", "median"),
        mean_training_auroc=("training_auroc", "mean"),
        mean_training_seconds=("training_seconds", "mean"),
    ).reset_index()
    summary.to_csv(destination / "summary.csv", index=False)
    return summary


def benchmark(args):
    frame, splits, metadata = load_splits(args.train, args.output_dir)
    configs = {mode: config_from_args(args, mode) for mode in args.embedding_modes}
    if any(mode != "random" for mode in configs):
        validate_vector_file(args.embeddings, args.embedding_dim)
    if args.fold == "all":
        folds = list(range(metadata["folds"]))
    else:
        folds = [int(args.fold)]
        if folds[0] not in range(metadata["folds"]):
            raise ValueError("Requested fold is outside the prepared range")
    evaluations = []
    if args.protocol in ("both", "random"):
        evaluations.append(("random", -1, (splits.random_split == "validation").to_numpy()))
    if args.protocol in ("both", "topic"):
        evaluations.extend(("topic", fold, (splits.topic_fold == fold).to_numpy()) for fold in folds)
    metrics, predictions, group_metrics = [], [], []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output_dir / f"cnn_benchmark_{args.protocol}_{args.fold}_{run_id}"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "run_metadata.json").write_text(json.dumps({
        "split_metadata": metadata, "configs": {mode: asdict(config) for mode, config in configs.items()},
        "splits_sha256": file_hash(args.output_dir / "splits.csv"),
        "models": args.models, "embedding_modes": args.embedding_modes,
        "folds": folds, "protocol": args.protocol,
        "epoch_selection": "early stopping on the fixed validation partition; no refit",
    }, indent=2) + "\n")
    for protocol, fold, validation in evaluations:
        outer_train = frame.loc[~validation].reset_index(drop=True)
        for name, embedding_mode in product(args.models, args.embedding_modes):
            config = configs[embedding_mode]
            print(f"Training {name}/{embedding_mode}: {protocol}, fold {fold}, {int((~validation).sum())} rows", flush=True)
            model_dir = destination / f"{protocol}_{fold}_{name}_{embedding_mode}"
            model_dir.mkdir(parents=True, exist_ok=True)
            training_started = perf_counter()
            predictor, selected_epoch = train_with_early_stopping(
                outer_train.text,
                outer_train.label,
                frame.text[validation],
                frame.label[validation],
                name,
                config,
                model_dir,
            )
            training_seconds = perf_counter() - training_started
            training_probabilities = predictor.predict(outer_train.text)
            probabilities = predictor.predict(frame.text[validation])
            labels = frame.label[validation]
            by_topic = topic_scores(labels, probabilities, splits.topic_group.to_numpy()[validation])
            supported = by_topic.loc[by_topic.eligible_for_summary, "auroc"]
            row = {
                "model": name, "embedding_mode": embedding_mode, "protocol": protocol, "fold": fold,
                "training_rows": int((~validation).sum()), "validation_rows": int(validation.sum()),
                "training_auroc": roc_auc_score(outer_train.label, training_probabilities),
                "training_seconds": training_seconds,
                "auroc": roc_auc_score(labels, probabilities),
                "selected_epoch": selected_epoch,
                "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
                "topic_macro_auroc": supported.mean(), "topic_worst_auroc": supported.min(),
                "eligible_topics": len(supported),
            }
            metrics.append(row)
            predictions.append(pd.DataFrame({
                "model": name, "embedding_mode": embedding_mode, "protocol": protocol, "fold": fold,
                "id": frame.id[validation].to_numpy(), "label": labels.to_numpy(),
                "probability": probabilities, "topic_group": splits.topic_group.to_numpy()[validation],
            }))
            group_metrics.append(by_topic.assign(model=name, embedding_mode=embedding_mode, protocol=protocol, fold=fold))
            # Preserve completed evaluations if the terminal run is interrupted.
            summary = write_results(destination, metrics, predictions, group_metrics)
            print(
                f"  Train AUROC={row['training_auroc']:.4f}, "
                f"validation AUROC={row['auroc']:.4f}, "
                f"training time={row['training_seconds']:.1f}s",
                flush=True,
            )
            del predictor
    print(summary.to_string(index=False))
    print(f"Saved results to {destination}")


def save_submission(predictor, test, test_path, checkpoint, destination, training_metadata=None):
    """Write probabilities and provenance for either a loaded or newly fitted CNN."""
    config = predictor.config
    model_name = predictor.encoder.mode
    probabilities = predictor.predict(test.text)
    if len(probabilities) != len(test) or not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("Invalid predicted probabilities")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"submission_{model_name}.csv"
    pd.DataFrame({"id": test.id, "label": probabilities}).to_csv(path, index=False)
    metadata = {
        "model": model_name, "config": asdict(config), "inference_only": training_metadata is None,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_hash(checkpoint),
        "test_sha256": file_hash(test_path), "prediction_rows": len(test),
        "sklearn_version": sklearn.__version__,
    }
    if training_metadata is not None:
        metadata["training"] = training_metadata
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(test)} probabilities in original test order to {path}")
    return path


def submit(args):
    """Generate test probabilities from saved weights, without any training."""
    test = read_data(args.test, labeled=False)
    predictor = load_predictor(args.checkpoint, device=args.device)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output_dir / f"cnn_prediction_{predictor.encoder.mode}_{predictor.config.embedding_mode}_{run_id}"
    print(f"Predicting from saved checkpoint: {args.checkpoint}", flush=True)
    return save_submission(predictor, test, args.test, args.checkpoint, destination)


def train_full(args):
    """Refit a selected configuration on all labeled rows for a fixed epoch count."""
    train = read_data(args.train)
    test = read_data(args.test, labeled=False)
    if set(train.id) & set(test.id):
        raise ValueError("Training and test IDs must not overlap")
    config = config_from_args(args, args.embedding_mode)
    if config.embedding_mode != "random":
        validate_vector_file(args.embeddings, config.embedding_dim)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output_dir / f"cnn_full_{args.model}_{args.embedding_mode}_{run_id}"
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model, "config": asdict(config),
        "training_rows": len(train), "train_sha256": file_hash(args.train),
        "test_sha256": file_hash(args.test), "epochs": args.epochs,
        "validation_auroc": None,
        "epoch_selection": "fixed --epochs, chosen before full-data refit",
        "sklearn_version": sklearn.__version__,
    }
    (destination / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    train[["id"]].to_csv(destination / "training_ids.csv", index=False)
    print(f"Training {args.model}/{args.embedding_mode} on all {len(train)} labeled rows for {args.epochs} epochs", flush=True)
    training_started = perf_counter()
    predictor = refit(train.text, train.label, args.model, config, args.epochs, destination)
    metadata["training_seconds"] = perf_counter() - training_started
    (destination / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return save_submission(predictor, test, args.test, destination / "model.pt", destination, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in [("prepare", prepare), ("benchmark", benchmark), ("train-full", train_full), ("submit", submit)]:
        command = commands.add_parser(name)
        command.set_defaults(handler=handler)
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for splits and results (default: %(default)s)")
        if name != "submit":
            command.add_argument("--train", type=Path, default=ROOT / "data/train.csv")
            command.add_argument("--seed", type=int, default=12)
        if name == "prepare":
            command.add_argument("--topics", type=int, default=25)
            command.add_argument("--folds", type=int, default=5)
        elif name == "benchmark":
            command.add_argument("--fold", default="0", help="held-out topic fold number, or all")
            command.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["cnn_raw"])
            command.add_argument("--protocol", choices=("both", "random", "topic"), default="random", help="random preserves the required shared 2,000-row holdout; topic/both are supplementary")
            command.add_argument("--embedding-modes", nargs="+", choices=("random", "frozen", "finetune"), default=["random"])
        elif name == "train-full":
            command.add_argument("--test", type=Path, default=ROOT / "data/test.csv")
            command.add_argument("--model", choices=MODEL_NAMES, default="cnn_raw")
            command.add_argument("--embedding-mode", choices=("random", "frozen", "finetune"), default="random")
        else:
            command.add_argument("--test", type=Path, default=ROOT / "data/test.csv")
            command.add_argument("--checkpoint", type=Path, required=True, help="saved model.pt; inference only, no training")
            command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        if name in ("benchmark", "train-full"):
            command.add_argument("--epochs", type=int, default=30 if name == "benchmark" else None, required=name == "train-full", help="selection cap for benchmark; fixed training duration for train-full")
            command.add_argument("--patience", type=int, default=3)
            command.add_argument("--max-vocab", type=int, default=30000)
            command.add_argument("--embedding-dim", type=int, default=100)
            command.add_argument("--embeddings", type=Path, help="local GloVe/plain word-vector .txt file")
            command.add_argument("--lowercase-embedding-fallback", action="store_true", help="try lowercase vector lookup after an exact match; token IDs keep their case")
            command.add_argument("--max-length", type=int, default=768)
            command.add_argument("--batch-size", type=int, default=64)
            command.add_argument("--learning-rate", type=float, default=0.001)
            command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
