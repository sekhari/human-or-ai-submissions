#!/usr/bin/env python3
"""Person 1: reproducible word TF-IDF + logistic regression.

Every vectorizer is fitted only on the rows used to fit its classifier. The
shared split is immutable and tied to the exact bytes of the original CSV.
Run ``python logistic_model.py --help`` for the three CLI stages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


SCHEMA_VERSION = 1


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_frame(path: str | Path, text_col: str, id_col: str | None,
               label_col: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing CSV: {path}. Download the competition files first.")
    # Strings preserve ID formatting and make the positive-label mapping explicit.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = [text_col] + ([label_col] if label_col else []) + ([id_col] if id_col else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; available: {list(frame.columns)}")
    if len(frame) == 0:
        raise ValueError(f"{path}: CSV has no rows.")
    if frame[text_col].str.strip().eq("").any():
        raise ValueError(f"{path}: missing/blank text; resolve the source data before modeling.")
    if id_col and (frame[id_col].eq("").any() or frame[id_col].duplicated().any()):
        raise ValueError(f"{path}: IDs must be nonempty and unique.")
    return frame


def labels_from(frame: pd.DataFrame, label_col: str, positive_label: str):
    labels = sorted(frame[label_col].unique().tolist())
    if len(labels) != 2 or "" in labels:
        raise ValueError(f"Expected two nonempty labels, observed {labels!r}.")
    if positive_label not in labels:
        raise ValueError(f"Positive label {positive_label!r} is absent; observed {labels!r}.")
    negative_label = next(label for label in labels if label != positive_label)
    y = frame[label_col].eq(positive_label).to_numpy(dtype=np.int64)
    return y, negative_label


def load_train(args):
    frame = read_frame(args.train, args.text_col, args.id_col, args.label_col)
    y, negative_label = labels_from(frame, args.label_col, args.positive_label)
    return frame, y, negative_label


def validate_manifest(manifest: dict, args, frame: pd.DataFrame, y: np.ndarray,
                      negative_label: str):
    expected = {"schema_version": SCHEMA_VERSION,
                "train_csv_sha256": sha256(args.train), "total_rows": len(frame),
                "text_col": args.text_col, "label_col": args.label_col,
                "id_col": args.id_col, "positive_label": args.positive_label,
                "negative_label": negative_label}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Split manifest mismatch for {key}: expected {value!r}, "
                             f"found {manifest.get(key)!r}. Use the original CSV and schema.")
    parts = []
    for key in ("train_indices", "validation_indices"):
        raw = manifest.get(key)
        if not isinstance(raw, list) or not raw or any(type(x) is not int for x in raw):
            raise ValueError(f"Manifest {key} must be a nonempty integer list.")
        indices = np.asarray(raw, dtype=np.int64)
        if indices.min() < 0 or indices.max() >= len(frame):
            raise ValueError(f"Manifest {key} contains out-of-range indices.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Manifest {key} contains duplicate indices.")
        if len(np.unique(y[indices])) != 2:
            raise ValueError(f"Manifest {key} must contain both classes for AUROC.")
        parts.append(indices)
    train_indices, validation_indices = parts
    joined = np.concatenate(parts)
    if len(joined) != len(frame) or len(np.unique(joined)) != len(frame):
        raise ValueError("Manifest partitions must be disjoint and cover every row exactly once.")
    if manifest.get("validation_rows") != len(validation_indices):
        raise ValueError("Manifest validation_rows disagrees with validation_indices.")
    if args.id_col:
        for name, indices in zip(("train_ids", "validation_ids"), parts):
            if manifest.get(name) != frame.iloc[indices][args.id_col].tolist():
                raise ValueError(f"Manifest {name} does not match source IDs at its row indices.")
    return train_indices, validation_indices


def read_manifest(args, frame, y, negative_label):
    path = Path(args.manifest)
    if not path.is_file():
        raise ValueError(f"Missing shared split {path}; run the split command first.")
    manifest = json.loads(path.read_text())
    train_indices, validation_indices = validate_manifest(manifest, args, frame, y, negative_label)
    return manifest, train_indices, validation_indices


def split_command(args):
    frame, y, negative_label = load_train(args)
    path = Path(args.manifest)
    if path.exists():
        manifest, train_indices, validation_indices = read_manifest(args, frame, y, negative_label)
        if len(validation_indices) != args.validation_size:
            raise ValueError("Existing split uses a different validation size; it will not be overwritten.")
        if not args.validation_ids and manifest.get("seed") != args.seed:
            raise ValueError("Existing split uses a different seed; it will not be overwritten.")
        if args.validation_ids:
            requested = read_validation_ids(args.validation_ids, args.id_col)
            if set(requested) != set(frame.iloc[validation_indices][args.id_col]):
                raise ValueError("Existing manifest differs from supplied team validation IDs; no overwrite.")
        print(f"Verified existing immutable split: {len(train_indices)} train / "
              f"{len(validation_indices)} validation rows.", flush=True)
        return
    if args.validation_size < 2 or args.validation_size > len(frame) - 2:
        raise ValueError("Validation size must leave at least two rows in each partition.")
    if args.validation_ids:
        requested = read_validation_ids(args.validation_ids, args.id_col)
        missing = sorted(set(requested) - set(frame[args.id_col]))
        if missing:
            raise ValueError(f"Team validation IDs absent from train.csv: {missing[:5]}")
        if len(requested) != args.validation_size:
            raise ValueError("Number of supplied validation IDs does not equal --validation-size.")
        mask = frame[args.id_col].isin(requested).to_numpy()
        validation_indices = np.flatnonzero(mask)
        train_indices = np.flatnonzero(~mask)
        origin = "imported_team_validation_ids"
    else:
        train_indices, validation_indices = train_test_split(
            np.arange(len(frame)), test_size=args.validation_size,
            random_state=args.seed, stratify=y)
        train_indices = np.sort(train_indices)
        validation_indices = np.sort(validation_indices)
        origin = "stratified_train_test_split"
    manifest = {"schema_version": SCHEMA_VERSION, "created_utc": utc_now(),
                "train_csv_sha256": sha256(args.train), "total_rows": len(frame),
                "text_col": args.text_col, "label_col": args.label_col, "id_col": args.id_col,
                "positive_label": args.positive_label, "negative_label": negative_label,
                "seed": args.seed, "origin": origin, "validation_rows": len(validation_indices),
                "train_indices": train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
                "train_class_counts": {str(k): int((y[train_indices] == k).sum()) for k in (0, 1)},
                "validation_class_counts": {str(k): int((y[validation_indices] == k).sum()) for k in (0, 1)}}
    if args.id_col:
        manifest["train_ids"] = frame.iloc[train_indices][args.id_col].tolist()
        manifest["validation_ids"] = frame.iloc[validation_indices][args.id_col].tolist()
    validate_manifest(manifest, args, frame, y, negative_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an accidental reshuffle or a concurrent overwrite.
    with path.open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Created immutable split: {len(train_indices)} train / {len(validation_indices)} "
          f"validation rows at {path}", flush=True)


def read_validation_ids(path, id_col):
    if not id_col:
        raise ValueError("--validation-ids requires --id-col.")
    ids = pd.read_csv(path, dtype=str, keep_default_na=False)
    if id_col not in ids.columns:
        raise ValueError(f"Validation ID CSV must have an {id_col!r} header.")
    values = ids[id_col].tolist()
    if not values or "" in values or len(values) != len(set(values)):
        raise ValueError("Validation IDs must be nonempty and unique.")
    return values


def model_params(c_value: float, max_features: int, seed: int, max_iter: int) -> dict:
    return {"C": float(c_value), "max_features": int(max_features),
            "ngram_range": [1, 2], "solver": "liblinear", "max_iter": int(max_iter),
            "random_state": int(seed)}


def make_pipeline(params: dict) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=tuple(params["ngram_range"]),
                                 max_features=params["max_features"], lowercase=True,
                                 dtype=np.float64)),
        ("logistic", LogisticRegression(C=params["C"], solver=params["solver"],
                                        max_iter=params["max_iter"],
                                        random_state=params["random_state"]))])


def probability(model, texts):
    positive_positions = np.flatnonzero(model.classes_ == 1)
    if len(positive_positions) != 1:
        raise ValueError("Fitted classifier lacks mapped positive class 1.")
    probabilities = model.predict_proba(texts)[:, int(positive_positions[0])]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Classifier returned invalid probabilities.")
    return probabilities


def fit_and_score(params, train_texts, train_y, val_texts, val_y):
    model = make_pipeline(params)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_texts, train_y)
    fit_seconds = time.perf_counter() - start
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in recorded)
    for item in recorded:
        if not issubclass(item.category, ConvergenceWarning):
            warnings.warn(str(item.message), item.category)
    train_probability = probability(model, train_texts)
    val_probability = probability(model, val_texts)
    train_auc = float(roc_auc_score(train_y, train_probability))
    val_auc = float(roc_auc_score(val_y, val_probability))
    result = {"train_auroc": train_auc, "val_auroc": val_auc,
              "auroc_gap": train_auc - val_auc, "train_time_seconds": fit_seconds,
              "n_features": len(model.named_steps["tfidf"].vocabulary_), "converged": converged}
    return model, result, val_probability


def nested_subsets(train_indices, y, sizes, seed):
    """Classwise shuffled prefixes; binary proportional allocation is monotone."""
    sizes = sorted(set(sizes))
    if any(size < 2 or size > len(train_indices) for size in sizes):
        raise ValueError(f"Every requested training size must be between 2 and {len(train_indices)}.")
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(train_indices[y[train_indices] == label]) for label in (0, 1)]
    fraction_positive = len(groups[1]) / len(train_indices)
    result = {}
    for size in sizes:
        n_positive = int(np.floor(size * fraction_positive + 0.5))
        n_positive = max(1, min(n_positive, size - 1, len(groups[1])))
        n_negative = size - n_positive
        if n_negative > len(groups[0]):
            n_negative = len(groups[0])
            n_positive = size - n_negative
        result[size] = np.sort(np.concatenate([groups[0][:n_negative], groups[1][:n_positive]]))
    previous = set()
    for subset in result.values():
        current = set(subset.tolist())
        if not previous.issubset(current):
            raise ValueError("Internal error: learning-curve subsets are not nested.")
        previous = current
    return result


def word_lengths(texts):
    return np.asarray([len(text.split()) for text in texts], dtype=np.int64)


def length_analysis(train_texts, val_texts, val_y, val_probability):
    train_lengths = word_lengths(train_texts)
    val_lengths = word_lengths(val_texts)
    thresholds = np.unique(np.quantile(train_lengths, [0.25, 0.5, 0.75])).tolist()
    edges = [-np.inf] + thresholds + [np.inf]
    records = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        train_mask = (train_lengths > lower) & (train_lengths <= upper)
        val_mask = (val_lengths > lower) & (val_lengths <= upper)
        if np.isneginf(lower):
            label = f"≤ {upper:g}"
        elif np.isposinf(upper):
            label = f"> {lower:g}"
        else:
            label = f"({lower:g}, {upper:g}]"
        n_positive = int((val_y[val_mask] == 1).sum())
        n_negative = int((val_y[val_mask] == 0).sum())
        auc = (float(roc_auc_score(val_y[val_mask], val_probability[val_mask]))
               if n_positive and n_negative else None)
        records.append({"length_bin": label,
                        "lower_words_exclusive": None if np.isneginf(lower) else lower,
                        "upper_words_inclusive": None if np.isposinf(upper) else upper,
                        "train_rows": int(train_mask.sum()), "val_rows": int(val_mask.sum()),
                        "val_negative_rows": n_negative, "val_positive_rows": n_positive,
                        "val_auroc": auc})
    return pd.DataFrame(records), thresholds


def plot_results(learning, bins, output_dir):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.plot(learning.train_rows, learning.train_auroc, "o-", label="Training AUROC", color="#4866a8")
    ax.plot(learning.train_rows, learning.val_auroc, "o-", label="Fixed validation AUROC", color="#d28237")
    ax.set(xlabel="Labeled training rows", ylabel="AUROC",
           title="Logistic regression: learning curve", ylim=(0.5, 1.01))
    ax.set_xticks(learning.train_rows)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"learning_curve.{suffix}", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    values = bins.val_auroc.to_numpy(dtype=float)
    positions = np.arange(len(bins))
    ax.bar(positions, np.nan_to_num(values, nan=0), color="#4866a8", width=0.65)
    ax.set_xticks(positions, bins.length_bin.tolist())
    ax.set(xlabel="Whitespace-separated words (training quartile thresholds)", ylabel="Validation AUROC",
           title="Logistic regression: performance by text length", ylim=(0, 1.12))
    for index, row in bins.iterrows():
        label = "undefined" if pd.isna(row.val_auroc) else f"{row.val_auroc:.4f}"
        height = 0 if pd.isna(row.val_auroc) else row.val_auroc
        ax.text(index, height + 0.02, f"{label}\nn={row.val_rows}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"length_bins.{suffix}", dpi=200)
    plt.close(fig)


def experiment_command(args):
    frame, y, negative_label = load_train(args)
    manifest, train_indices, val_indices = read_manifest(args, frame, y, negative_label)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metrics.json").exists() and not args.overwrite_results:
        raise ValueError(f"Completed results already exist at {output_dir}; use a new directory or --overwrite-results.")
    if any(c <= 0 for c in args.c_values) or any(n <= 0 for n in args.max_features):
        raise ValueError("C and max-features values must be positive.")
    subsets = nested_subsets(train_indices, y, args.train_sizes, manifest["seed"])
    texts = frame[args.text_col].to_numpy()
    train_texts, val_texts = texts[train_indices], texts[val_indices]
    train_y, val_y = y[train_indices], y[val_indices]
    records, best_model, best_result, best_probability, best_params = [], None, None, None, None
    tuning_start = time.perf_counter()
    for c_value in sorted(set(args.c_values)):
        for features in sorted(set(args.max_features)):
            params = model_params(c_value, features, manifest["seed"], args.max_iter)
            model, result, val_probability = fit_and_score(params, train_texts, train_y, val_texts, val_y)
            record = {"C": c_value, "max_features": features, **result}
            records.append(record)
            # Iteration order breaks exact AUROC ties in favor of smaller C, then vocabulary cap.
            if result["converged"] and (best_result is None or result["val_auroc"] > best_result["val_auroc"]):
                best_model, best_result, best_probability, best_params = model, result, val_probability, params
            print(f"C={c_value:g}, max_features={features}: train={result['train_auroc']:.6f}, "
                  f"val={result['val_auroc']:.6f}, fit={result['train_time_seconds']:.2f}s, "
                  f"converged={result['converged']}", flush=True)
            pd.DataFrame(records).to_csv(output_dir / "tuning_results.csv", index=False)
    tuning_total = time.perf_counter() - tuning_start
    if best_model is None:
        raise ValueError("No tuning candidate converged. Increase --max-iter; no final result has been selected.")
    for record in records:
        record["selected"] = record["C"] == best_params["C"] and record["max_features"] == best_params["max_features"]
    pd.DataFrame(records).to_csv(output_dir / "tuning_results.csv", index=False)
    selected_artifact = {"schema_version": SCHEMA_VERSION, "created_utc": utc_now(),
                         "train_csv_sha256": sha256(args.train),
                         "split_manifest_sha256": sha256(args.manifest),
                         "positive_label": args.positive_label, "negative_label": negative_label,
                         "text_col": args.text_col, "label_col": args.label_col, "id_col": args.id_col,
                         "selected_params": best_params,
                         "selection_rule": "Highest validation AUROC among converged fits; exact ties: smaller C, then max_features."}
    write_json(output_dir / "selected_params.json", selected_artifact)
    joblib.dump(best_model, output_dir / "model.joblib")
    predictions = pd.DataFrame({"row_index": val_indices,
                                "id": frame.iloc[val_indices][args.id_col].to_numpy() if args.id_col else val_indices,
                                "true_label": val_y, "p_positive": best_probability,
                                "p_negative": 1 - best_probability,
                                "word_count": word_lengths(val_texts)})
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    learning_records = []
    for size, subset in subsets.items():
        if size == len(train_indices):
            # This already-fitted model is precisely the full training subset.
            result = best_result.copy()
        else:
            _, result, _ = fit_and_score(best_params, texts[subset], y[subset], val_texts, val_y)
        if not result["converged"]:
            raise ValueError(f"Learning-curve fit at {size} rows did not converge. Increase --max-iter.")
        learning_records.append({"train_rows": size, **result,
                                 "class_0_rows": int((y[subset] == 0).sum()),
                                 "class_1_rows": int((y[subset] == 1).sum())})
        print(f"Learning curve {size}: train={result['train_auroc']:.6f}, val={result['val_auroc']:.6f}", flush=True)
    learning = pd.DataFrame(learning_records)
    learning.to_csv(output_dir / "learning_curve.csv", index=False)
    write_json(output_dir / "learning_curve_subsets.json",
               {"seed": manifest["seed"], "split_manifest_sha256": sha256(args.manifest),
                "row_indices_by_size": {str(size): subset.tolist() for size, subset in subsets.items()}})
    bins, thresholds = length_analysis(train_texts, val_texts, val_y, best_probability)
    bins.to_csv(output_dir / "length_bins.csv", index=False)
    plot_results(learning, bins, output_dir)
    metrics = {"schema_version": SCHEMA_VERSION, "created_utc": utc_now(),
               "model_name": "Word TF-IDF (1–2 grams) + logistic regression",
               "train_rows": len(train_indices), "validation_rows": len(val_indices),
               "total_rows": len(frame), "positive_label": args.positive_label,
               "negative_label": negative_label, "selected_params": best_params, **best_result,
               "tuning_total_seconds": tuning_total,
               "train_csv_sha256": sha256(args.train), "split_manifest_sha256": sha256(args.manifest),
               "learning_curve_rows": sorted(subsets), "length_thresholds_words": thresholds,
               "train_time_definition": "Wall-clock pipeline.fit only, including TF-IDF fit/transform and logistic fit; excludes predictions and metric calculation.",
               "learning_curve_design": "Nested stratified training subsets; fixed selected hyperparameters; freshly fitted vocabulary except full-training reuse; same validation rows throughout.",
               "validation_caveat": "This validation set selected hyperparameters; reported validation scores are not an unbiased final test estimate.",
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                            "matplotlib": matplotlib.__version__}}
    write_json(output_dir / "metrics.json", metrics)
    print(f"Saved completed experiment to {output_dir}. Selected C={best_params['C']}, "
          f"max_features={best_params['max_features']}; validation AUROC={best_result['val_auroc']:.6f}", flush=True)


def aligned_submission(sample: pd.DataFrame, test: pd.DataFrame, id_col: str,
                       prediction_col: str, probabilities: np.ndarray):
    if id_col == prediction_col:
        raise ValueError("Prediction column and ID column must differ.")
    if id_col not in sample or prediction_col not in sample:
        raise ValueError("Sample submission must contain both --id-col and --prediction-col.")
    if sample[id_col].eq("").any() or sample[id_col].duplicated().any():
        raise ValueError("Sample-submission IDs must be nonempty and unique.")
    if len(sample) != len(test) or set(sample[id_col]) != set(test[id_col]):
        raise ValueError("Sample-submission IDs must match test IDs exactly.")
    if len(probabilities) != len(test) or not np.isfinite(probabilities).all():
        raise ValueError("Predictions must be finite and have one value for each test row.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predictions must lie within [0, 1].")
    result = sample.copy()
    mapping = pd.Series(probabilities, index=test[id_col].to_numpy())
    result[prediction_col] = result[id_col].map(mapping).to_numpy()
    return result


def submit_command(args):
    if not args.id_col:
        raise ValueError("Submission generation requires --id-col to verify row alignment.")
    frame, y, negative_label = load_train(args)
    read_manifest(args, frame, y, negative_label)
    selected_path = Path(args.results_dir) / "selected_params.json"
    if not selected_path.is_file():
        raise ValueError(f"Missing selected model parameters: {selected_path}; run experiment first.")
    selected = json.loads(selected_path.read_text())
    expected = {"train_csv_sha256": sha256(args.train),
                "split_manifest_sha256": sha256(args.manifest), "positive_label": args.positive_label,
                "negative_label": negative_label, "text_col": args.text_col,
                "label_col": args.label_col, "id_col": args.id_col}
    for key, value in expected.items():
        if selected.get(key) != value:
            raise ValueError(f"Selected-parameter artifact does not match current {key}.")
    output = Path(args.output)
    if output.exists() and not args.overwrite_output:
        raise ValueError(f"Submission exists: {output}; choose a new name or --overwrite-output.")
    test = read_frame(args.test, args.text_col, args.id_col)
    if set(test[args.id_col]) & set(frame[args.id_col]):
        raise ValueError("Training IDs and test IDs overlap; verify the source files.")
    sample = pd.read_csv(args.sample_submission, dtype=str, keep_default_na=False)
    # Check alignment before spending time on a full-data fit.
    aligned_submission(sample, test, args.id_col, args.prediction_col, np.zeros(len(test)))
    model = make_pipeline(selected["selected_params"])
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(frame[args.text_col].to_numpy(), y)
    seconds = time.perf_counter() - start
    probabilities = probability(model, test[args.text_col].to_numpy())
    submission = aligned_submission(sample, test, args.id_col, args.prediction_col, probabilities)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False, float_format="%.15g")
    round_trip = pd.read_csv(output, dtype={args.id_col: str}, keep_default_na=False)
    if list(round_trip.columns) != list(sample.columns) or round_trip[args.id_col].tolist() != sample[args.id_col].tolist():
        raise ValueError("Written submission does not preserve the sample header and ID order.")
    written = round_trip[args.prediction_col].to_numpy(dtype=float)
    if not np.isfinite(written).all() or ((written < 0) | (written > 1)).any():
        raise ValueError("Written submission contains invalid probabilities.")
    joblib.dump(model, output.with_suffix(".model.joblib"))
    metadata = {"schema_version": SCHEMA_VERSION, "created_utc": utc_now(),
                "status": "candidate_only" if args.candidate_only else "team_selection_confirmed",
                "uploaded_to_kaggle": False, "train_rows": len(frame), "test_rows": len(test),
                "positive_label": args.positive_label, "prediction_col": args.prediction_col,
                "selected_params": selected["selected_params"], "train_time_seconds": seconds,
                "train_csv_sha256": sha256(args.train), "test_csv_sha256": sha256(args.test),
                "sample_submission_sha256": sha256(args.sample_submission),
                "submission_sha256": sha256(output), "split_manifest_sha256": sha256(args.manifest),
                "min_probability": float(written.min()), "max_probability": float(written.max()),
                "mean_probability": float(written.mean()),
                "note": "Full-data refit is for test predictions only; validation metrics belong to the 8k training model."}
    write_json(output.with_suffix(".metadata.json"), metadata)
    print(f"Saved {metadata['status']} submission: {output} ({len(test)} rows). No upload performed.", flush=True)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--train", required=True, help="Original labeled training CSV")
    common.add_argument("--text-col", required=True)
    common.add_argument("--label-col", required=True)
    common.add_argument("--id-col")
    common.add_argument("--positive-label", default="1", help="Exact CSV label representing AI/positive class")
    common.add_argument("--manifest", required=True, help="Immutable shared JSON split manifest")
    split = commands.add_parser("split", parents=[common], help="Create or verify the fixed shared split")
    split.add_argument("--validation-size", type=int, default=2000)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--validation-ids", help="Import team validation IDs from a CSV with the --id-col header")
    split.set_defaults(function=split_command)
    experiment = commands.add_parser("experiment", parents=[common], help="Tune and analyze Model A")
    experiment.add_argument("--output-dir", required=True)
    experiment.add_argument("--c-values", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    experiment.add_argument("--max-features", type=int, nargs="+", default=[20000, 50000, 100000])
    experiment.add_argument("--train-sizes", type=int, nargs="+", default=[500, 1000, 2000, 5000, 8000])
    experiment.add_argument("--max-iter", type=int, default=2000)
    experiment.add_argument("--overwrite-results", action="store_true")
    experiment.set_defaults(function=experiment_command)
    submit = commands.add_parser("submit", parents=[common], help="Refit selected Model A on all labels and write a local submission")
    submit.add_argument("--test", required=True)
    submit.add_argument("--sample-submission", required=True)
    submit.add_argument("--prediction-col", required=True)
    submit.add_argument("--results-dir", required=True)
    submit.add_argument("--output", required=True)
    gate = submit.add_mutually_exclusive_group(required=True)
    gate.add_argument("--candidate-only", action="store_true", help="Prepare a Model A candidate before team model selection")
    gate.add_argument("--confirm-team-selection", action="store_true", help="Team has selected Model A using validation comparisons")
    submit.add_argument("--overwrite-output", action="store_true")
    submit.set_defaults(function=submit_command)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        args.function(args)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, json.JSONDecodeError,
            ConvergenceWarning) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
