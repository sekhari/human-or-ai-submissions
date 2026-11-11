#!/usr/bin/env python3
"""Reproduce the team's classical models on the existing fixed 2,000 IDs.

Run with --train PATH --output-dir PACKAGE_DIR --seed 42. Original team
sources must be in PACKAGE_DIR/code, or supplied with --code-dir.
No estimator parameters are changed. No models are downloaded or saved.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

C_VALUES = [0.25, 1.0, 4.0, 16.0, 64.0]
TUNED_MODELS = ["tfidf_lr", "char_lr", "tfidf_svm"]
TRAIN_SIZES = [500, 1000, 2000, 5000, 8000]
TIE_TOLERANCE = 0.001


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def feature_metadata(model):
    features = model.steps[0][1]
    if hasattr(features, "transformer_list"):
        counts = {name: len(transformer.vocabulary_)
                  for name, transformer in features.transformer_list}
    else:
        counts = {model.steps[0][0]: len(features.vocabulary_)}
    return {"feature_count": sum(counts.values()), "feature_counts": counts}


def iteration_metadata(model):
    classifier = model.steps[-1][1]
    if hasattr(classifier, "calibrated_classifiers_"):
        entries = []
        for index, calibrated in enumerate(classifier.calibrated_classifiers_):
            estimator = getattr(calibrated, "estimator", None)
            if estimator is None:
                estimator = getattr(calibrated, "base_estimator", None)
            entries.append({"calibration_fold": index,
                            "n_iter": getattr(estimator, "n_iter_", None),
                            "max_iter": getattr(estimator, "max_iter", None)})
    else:
        entries = [{"n_iter": getattr(classifier, "n_iter_", None),
                    "max_iter": getattr(classifier, "max_iter", None)}]
    for entry in entries:
        for key in ("n_iter", "max_iter"):
            value = entry[key]
            if hasattr(value, "tolist"):
                entry[key] = value.tolist()
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--code-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed != 42:
        parser.error("This predeclared experiment requires --seed 42.")
    output = args.output_dir.resolve()
    code = (args.code_dir or output / "code").resolve()
    train_path = args.train.resolve()
    if not (code / "train.py").is_file():
        raise FileNotFoundError(f"Original team sources not found: {code / 'train.py'}")
    sys.path.insert(0, str(code))
    import numpy as np
    import pandas as pd
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from src.data import SEED, SPLIT_FILE, get_train_val, load_data
    from train import MODELS

    if SEED != args.seed:
        raise ValueError(f"Team source SEED is {SEED}, expected {args.seed}")
    if set(TUNED_MODELS + ["word_char_lr"]) - set(MODELS):
        raise ValueError("The original four model builders are required.")
    training_sources = [code / "train.py", code / "src" / "data.py", code / "src" / "__init__.py"]
    source_files = {str(path.relative_to(code)): sha256_file(path)
                    for path in training_sources}
    split_path = Path(SPLIT_FILE).resolve()
    split = json.loads(split_path.read_text())
    if split.get("seed") != args.seed:
        raise ValueError("Saved split seed must be 42.")
    if len(split["val_ids"]) != 2000 or len(set(split["val_ids"])) != 2000:
        raise ValueError("Saved validation IDs must contain 2,000 unique IDs.")
    full = load_data(train_path)
    if len(full) != 10000 or full["id"].isna().any() or not full["id"].is_unique:
        raise ValueError("Expected 10,000 rows with unique non-null IDs.")
    if set(full["label"].unique()) != {0, 1}:
        raise ValueError("Labels must be exactly 0 and 1.")
    train, validation = get_train_val(full)
    if train["label"].value_counts().to_dict() != {0: 4000, 1: 4000}:
        raise ValueError("Expected 4,000 training rows per class.")
    if validation["label"].value_counts().to_dict() != {0: 1000, 1: 1000}:
        raise ValueError("Expected 1,000 validation rows per class.")
    if set(train["id"]) & set(validation["id"]):
        raise ValueError("Training and validation IDs overlap.")
    if set(validation["id"]) != set(split["val_ids"]):
        raise ValueError("Validation membership differs from the saved IDs.")
    protocol = {
        "schema_version": 1,
        "upstream_commit": "263644720413b4b8b82bc999f358da04ddba78d0",
        "seed": args.seed, "input_sha256": sha256_file(train_path),
        "source_sha256": source_files,
        "runner_sha256": sha256_file(Path(__file__)),
        "split_sha256": sha256_file(split_path),
        "validation_ids_sorted_sha256": json_digest(sorted(validation["id"].tolist())),
        "training_ids_order_sha256": json_digest(train["id"].tolist()),
        "validation_ids_order_sha256": json_digest(validation["id"].tolist()),
        "train_rows": len(train), "validation_rows": len(validation),
        "train_class_counts": {str(k): int(v) for k, v in train["label"].value_counts().items()},
        "validation_class_counts": {str(k): int(v) for k, v in validation["label"].value_counts().items()},
        "tuned_models": TUNED_MODELS, "C_values": C_VALUES,
        "initial_C": {name: 4.0 for name in TUNED_MODELS} | {"word_char_lr": 16.0},
        "selection_rule": "Choose smallest C with best validation AUC minus candidate AUC strictly less than 0.001.",
        "tie_tolerance": TIE_TOLERANCE,
        "learning_curve": {"model": "word_char_lr", "C": 16.0, "sizes": TRAIN_SIZES,
                           "sampling": "Independent stratified train_test_split from the fixed 8000, random_state=42; full 8000 reused."},
        "expected_unique_fits": 20,
        "estimator_parameters": "Use original train.MODELS builders unchanged; supply only seed and C.",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": {name: importlib.metadata.version(name)
                                 for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
                    "thread_environment": {name: os.environ.get(name) for name in
                                           ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]}},
    }
    fingerprint = json_digest(protocol)
    manifest_path = output / "protocol.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != protocol:
            raise RuntimeError("Existing protocol differs in source, runner, data, split, configuration, or runtime. Use a new output directory; cache reuse refused.")
    else:
        if any((output / "runs").glob("*/result.json")):
            raise RuntimeError("Run results exist without a protocol; cache reuse refused.")
        write_json(manifest_path, protocol)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    progress = {"protocol_fingerprint": fingerprint, "status": "running", "events": []}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError("Progress protocol fingerprint mismatch.")
    all_results = {}

    def event(kind, **fields):
        entry = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "event": kind, **fields}
        progress["events"].append(entry)
        progress["status"] = "failed" if kind == "failed" else ("complete" if kind == "complete" else "running")
        progress["completed_unique_runs"] = len(all_results)
        write_json(progress_path, progress)
        with open(output / "events.jsonl", "a") as handle:
            handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(entry, sort_keys=True, allow_nan=False), flush=True)

    def fit_one(name, C, subset):
        run_id = f"{name}_C{C:g}_n{len(subset)}"
        if run_id in all_results:
            return all_results[run_id]
        directory = output / "runs" / run_id
        config = {"protocol_fingerprint": fingerprint, "model": name, "C": C,
                  "seed": args.seed, "train_rows": len(subset),
                  "train_ids_order_sha256": json_digest(subset["id"].tolist())}
        result_path = directory / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("config") != config or not result.get("complete"):
                raise RuntimeError(f"Cached run configuration mismatch: {run_id}")
            payload = {key: value for key, value in result.items() if key != "result_payload_sha256"}
            if result.get("result_payload_sha256") != json_digest(payload):
                raise RuntimeError(f"Cached result checksum mismatch: {run_id}")
            for filename, digest in result["prediction_sha256"].items():
                path = directory / filename
                if not path.is_file() or sha256_file(path) != digest:
                    raise RuntimeError(f"Cached prediction checksum mismatch: {path}")
            all_results[run_id] = result
            event("cache_hit", run_id=run_id)
            return result
        event("fit_started", run_id=run_id, config=config)
        model = None
        train_probs = val_probs = None
        try:
            model = MODELS[name](args.seed, C=C)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                model.fit(subset["text"], subset["label"])
                fit_seconds = time.perf_counter() - start
                classes = list(model.classes_)
                positive_column = classes.index(1)
                start = time.perf_counter()
                train_probs = model.predict_proba(subset["text"])[:, positive_column]
                train_predict_seconds = time.perf_counter() - start
                start = time.perf_counter()
                val_probs = model.predict_proba(validation["text"])[:, positive_column]
                val_predict_seconds = time.perf_counter() - start
            for probabilities in [train_probs, val_probs]:
                if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                    raise ValueError(f"Invalid probabilities in {run_id}")
            warning_records = [{"category": item.category.__name__, "message": str(item.message),
                                "filename": item.filename, "lineno": item.lineno} for item in caught]
            result = {"complete": True, "run_id": run_id, "config": config,
                      "model": name, "C": C, "seed": args.seed,
                      "train_rows": len(subset), "validation_rows": len(validation),
                      "train_auroc": float(roc_auc_score(subset["label"], train_probs)),
                      "val_auroc": float(roc_auc_score(validation["label"], val_probs)),
                      "fit_seconds": fit_seconds, "train_predict_seconds": train_predict_seconds,
                      "validation_predict_seconds": val_predict_seconds,
                      "predict_seconds": train_predict_seconds + val_predict_seconds,
                      "warnings": warning_records,
                      "convergence_warning_count": sum(issubclass(item.category, ConvergenceWarning) for item in caught),
                      "iterations": iteration_metadata(model), **feature_metadata(model)}
            result["prediction_sha256"] = {}
            for filename, frame, probabilities in [("train_predictions.csv", subset, train_probs),
                                                     ("val_predictions.csv", validation, val_probs)]:
                predictions = frame[["id", "label"]].copy()
                predictions["probability"] = probabilities
                write_csv(directory / filename, predictions)
                result["prediction_sha256"][filename] = sha256_file(directory / filename)
            result["result_payload_sha256"] = json_digest(result)
            write_json(result_path, result)
            all_results[run_id] = result
            event("fit_completed", run_id=run_id, train_auroc=result["train_auroc"],
                  val_auroc=result["val_auroc"], fit_seconds=fit_seconds,
                  convergence_warning_count=result["convergence_warning_count"])
            return result
        finally:
            del model, train_probs, val_probs
            gc.collect()

    columns = ["run_id", "model", "C", "seed", "train_rows", "validation_rows", "train_auroc",
               "val_auroc", "fit_seconds", "train_predict_seconds", "validation_predict_seconds",
               "predict_seconds", "feature_count", "convergence_warning_count"]

    def table(filename, results):
        write_csv(output / filename, pd.DataFrame([{key: item[key] for key in columns} for item in results]))

    try:
        event("started", expected_unique_fits=20)
        initial = [fit_one(name, 4.0, train) for name in TUNED_MODELS]
        combined = fit_one("word_char_lr", 16.0, train)
        initial.append(combined)
        table("comparison_initial.csv", initial)
        tuning, selected = [], []
        selections = {}
        for name in TUNED_MODELS:
            candidates = [fit_one(name, C, train) for C in C_VALUES]
            tuning.extend(candidates)
            best_auc = max(item["val_auroc"] for item in candidates)
            choice = min((item for item in candidates if best_auc - item["val_auroc"] < TIE_TOLERANCE),
                         key=lambda item: item["C"])
            selections[name] = {"selected_C": choice["C"], "selected_val_auroc": choice["val_auroc"],
                                "best_val_auroc": best_auc, "best_minus_selected_auc": best_auc - choice["val_auroc"]}
            selected.append(choice)
            table("tuning.csv", tuning)
        selected.append(combined)
        table("comparison_selected.csv", selected)
        curve = []
        for size in TRAIN_SIZES:
            if size == len(train):
                result = combined
            else:
                subset, _ = train_test_split(train, train_size=size, stratify=train["label"], random_state=args.seed)
                result = fit_one("word_char_lr", 16.0, subset)
            curve.append(result)
            table("learning_curve.csv", curve)
            write_json(output / "learning_curve.json", {"model": "word_char_lr", "C": 16.0,
                                                         "seed": args.seed, "rows": curve})
        summary = {"complete": True, "protocol_fingerprint": fingerprint, "protocol": protocol,
                   "unique_fits": len(all_results), "selection": selections,
                   "comparison_initial": initial, "comparison_selected": selected,
                   "tuning": tuning, "learning_curve": curve,
                   "total_fit_seconds": sum(item["fit_seconds"] for item in all_results.values()),
                   "total_predict_seconds": sum(item["predict_seconds"] for item in all_results.values()),
                   "interpretation": "Selected scores reuse the hyperparameter-selection validation set and are not an independent final test estimate."}
        if len(all_results) != 20:
            raise RuntimeError(f"Expected 20 unique fits, found {len(all_results)}")
        write_json(output / "summary.json", summary)
        event("complete", summary_path=str(output / "summary.json"), unique_fits=len(all_results))
    except BaseException as error:
        event("failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
