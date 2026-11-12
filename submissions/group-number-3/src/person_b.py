"""Reproducible Person B baseline, learning curves and one-round self-training.

Run from the team directory: python src/person_b.py {verify,baseline,q2,q3}
Existing completed stages are never overwritten. A different --output-dir
creates an independent reproduction. No command generates shared splits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from data import ROOT, get_split
from models import tfidf_logreg

REPO = ROOT.parent.parent
DEFAULT_CONFIG = ROOT / "config/person_b.json"
DEFAULT_OUTPUT = ROOT / "results/person_b"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default) + "\n")


def id_hash(values):
    content = "\n".join(sorted(str(value) for value in values)) + "\n"
    return hashlib.sha256(content.encode()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def validate_split_data(train, splits):
    required = {"id", "label", "random_fold", "cluster", "ood_fold"}
    if not required.issubset(splits.columns):
        raise ValueError("Shared split is missing required columns.")
    if train["id"].isna().any() or not train["id"].is_unique:
        raise ValueError("Training IDs must be present and unique.")
    if splits[list(required)].isna().any().any() or not splits["id"].is_unique:
        raise ValueError("Split values must be complete and IDs unique.")
    if set(train["id"]) != set(splits["id"]):
        raise ValueError("Split IDs do not exactly match the official training IDs.")
    aligned = train[["id", "label"]].merge(
        splits[["id", "label"]], on="id", validate="one_to_one", suffixes=("_a", "_b")
    )
    if not aligned["label_a"].eq(aligned["label_b"]).all():
        raise ValueError("Split labels disagree with official training labels.")
    for fold in ("random_fold", "ood_fold"):
        if set(splits[fold]) != {"train", "val"}:
            raise ValueError(f"Invalid values in {fold}.")


def nested_stratified_indices(labels, sizes, seed):
    labels = np.asarray(labels)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected binary labels 0 and 1.")
    rng = np.random.default_rng(seed)
    permutations = {label: rng.permutation(np.flatnonzero(labels == label))
                    for label in (0, 1)}
    subsets = {}
    for size in sizes:
        if size <= 0 or size % 2 or size // 2 > min(map(len, permutations.values())):
            raise ValueError("Balanced subsets require a valid even sample count.")
        # Sorting preserves the pool's original order, including at size 8,000.
        subsets[size] = np.sort(np.concatenate(
            [permutations[label][:size // 2] for label in (0, 1)]
        ))
    return subsets


def select_pseudo_labels(probabilities, low, high):
    probabilities = np.asarray(probabilities, dtype=float)
    if not 0 <= low < 0.5 < high <= 1:
        raise ValueError("Thresholds must bracket 0.5 and lie in [0, 1].")
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be a finite one-dimensional array.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1].")
    selected = (probabilities <= low) | (probabilities >= high)
    return selected, (probabilities[selected] >= high).astype(int)


def positive_probabilities(classifier, matrix):
    if set(classifier.classes_) != {0, 1}:
        raise ValueError("Unexpected classifier classes.")
    column = np.flatnonzero(classifier.classes_ == 1).item()
    probabilities = classifier.predict_proba(matrix)[:, column]
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0) & (probabilities <= 1)
    ).all():
        raise ValueError("Invalid predicted AI probabilities.")
    return probabilities


def model_parameters(config):
    model = tfidf_logreg(C=config["C"])
    return {
        "classifier": model.named_steps["clf"].get_params(),
        "vectorizers": {name: vectorizer.get_params()
                        for name, vectorizer in model.named_steps["features"].transformer_list},
    }


def feature_fingerprint(features):
    """Hash learned representation, excluding process-local analyzer caches.

    Vectorizer.transform() refreshes _stop_words_id after deserialization.
    That Python object ID is not vocabulary, IDF or a transformation parameter.
    A whole-estimator pickle hash would falsely report a representation change.
    """
    vectorizers = []
    for name, vectorizer in features.transformer_list:
        idf = np.asarray(vectorizer.idf_, dtype="<f8")
        vectorizers.append({
            "name": name, "parameters": vectorizer.get_params(deep=False),
            "vocabulary": sorted((word, int(index))
                                 for word, index in vectorizer.vocabulary_.items()),
            "idf_shape": idf.shape,
            "idf_sha256": hashlib.sha256(idf.tobytes(order="C")).hexdigest(),
        })
    payload = {"transformer_weights": features.transformer_weights,
               "vectorizers": vectorizers}
    canonical = json.dumps(payload, sort_keys=True, default=json_default)
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_context(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    if config["model"] != "tfidf_logreg":
        raise ValueError("This protocol requires the unchanged team TF-IDF baseline.")
    if config["pseudo_label_rounds"] != 1 or config["pseudo_label_weight"] != 1.0:
        raise ValueError("This implementation is for the declared one-round, unit-weight study.")
    if config["validation_fold"] != "random_fold":
        raise ValueError("Q2/Q3 require the fixed random validation fold.")
    if config["q3_feature_policy"] != "freeze_vectorizers_fitted_on_labeled_training_only":
        raise ValueError("Unsupported feature policy.")
    actual_hashes = {name: sha256(ROOT / name) for name in config["expected_sha256"]}
    if actual_hashes != config["expected_sha256"]:
        raise ValueError("Input hash mismatch; do not silently change data or split.")
    train = pd.read_csv(ROOT / "data/train.csv", dtype={"id": "string"})
    splits = pd.read_csv(ROOT / "splits/splits.csv", dtype={"id": "string"})
    validate_split_data(train, splits)
    pool, validation = get_split(train, splits, config["validation_fold"])
    if (len(pool), len(validation)) != (
        config["expected_random_train"], config["expected_random_validation"]
    ):
        raise ValueError("Unexpected random split sizes.")
    if set(pool["id"]) & set(validation["id"]):
        raise ValueError("Train and validation IDs overlap.")
    packages = {name: metadata.version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"
    )}
    code_files = [Path(__file__).resolve(), ROOT / "src/models.py", ROOT / "src/data.py"]
    protocol = {
        "config": config, "config_sha256": sha256(config_path),
        "input_sha256": actual_hashes,
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_files},
        "python": platform.python_version(), "packages": packages,
        "model_parameters": model_parameters(config),
        "random_training_ids_sha256": id_hash(pool["id"]),
        "random_validation_ids_sha256": id_hash(validation["id"]),
    }
    canonical = json.dumps(protocol, sort_keys=True, default=json_default)
    signature = hashlib.sha256(canonical.encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["protocol_signature"] != signature:
            raise ValueError("Existing output uses a different protocol. Choose a new output directory.")
    else:
        write_json(manifest_path, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "originating_code_commit": git("rev-parse", "HEAD"),
            "protocol_signature": signature, "protocol": protocol,
            "test_csv_used": False,
        })
    cache = ROOT / ".person_b_cache" / signature
    cache.mkdir(parents=True, exist_ok=True)
    return dict(config=config, output=output, cache=cache, signature=signature,
                train=train, splits=splits, pool=pool, validation=validation)


@contextmanager
def stage(ctx, name):
    completed = ctx["output"] / f"{name}_completed.json"
    if completed.exists():
        raise FileExistsError(f"{name} is already complete; refusing to overwrite results.")
    started = time.perf_counter()
    print(f"START {name}: {datetime.now(timezone.utc).isoformat()}", flush=True)
    record = {"stage": name, "code_commit": git("rev-parse", "HEAD"),
              "protocol_signature": ctx["signature"]}
    try:
        yield
    except Exception:
        record.update(status="failed", elapsed_seconds=time.perf_counter() - started,
                      traceback=traceback.format_exc())
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        write_json(ctx["output"] / f"{name}_failure_{stamp}.json", record)
        raise
    else:
        record.update(status="completed", elapsed_seconds=time.perf_counter() - started)
        write_json(completed, record)
        print(f"COMPLETE {name}: {record['elapsed_seconds']:.1f}s", flush=True)


def fit_and_measure(training, validation, config):
    model = tfidf_logreg(C=config["C"])
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        print(f"  Fit TF-IDF on {len(training):,} labeled training rows only", flush=True)
        x_train = model.named_steps["features"].fit_transform(training["text"])
        if not sparse.issparse(x_train):
            raise ValueError("Text features must remain sparse.")
        feature_seconds = time.perf_counter() - started
        print(f"  Sparse features {x_train.shape}, nnz={x_train.nnz:,}; fit classifier", flush=True)
        fit_start = time.perf_counter()
        model.named_steps["clf"].fit(x_train, training["label"].to_numpy())
        classifier_seconds = time.perf_counter() - fit_start
        fit_seconds = time.perf_counter() - started
        p_train = positive_probabilities(model.named_steps["clf"], x_train)
        del x_train
        x_validation = model.named_steps["features"].transform(validation["text"])
        p_validation = positive_probabilities(model.named_steps["clf"], x_validation)
        result = {
            "training_rows": len(training), "validation_rows": len(validation),
            "training_auroc": float(roc_auc_score(training["label"], p_train)),
            "validation_auroc": float(roc_auc_score(validation["label"], p_validation)),
            "fit_seconds": fit_seconds, "feature_fit_seconds": feature_seconds,
            "classifier_fit_seconds": classifier_seconds,
            "feature_count": int(x_validation.shape[1]),
            "classifier_iterations": model.named_steps["clf"].n_iter_.tolist(),
            "training_ids_sha256": id_hash(training["id"]),
            "validation_ids_sha256": id_hash(validation["id"]),
            "warnings": [{"category": item.category.__name__, "message": str(item.message)}
                         for item in captured],
        }
        del x_validation
    print(f"  Train AUROC={result['training_auroc']:.6f}, "
          f"validation AUROC={result['validation_auroc']:.6f}, fit={fit_seconds:.1f}s",
          flush=True)
    return model, result, p_validation


def save_predictions(path, validation, probabilities):
    pd.DataFrame({"id": validation["id"].to_numpy(),
                  "true_label": validation["label"].to_numpy(),
                  "ai_probability": probabilities}).to_csv(path, index=False, float_format="%.12g")


def run_baseline(ctx):
    results = {}
    for fold in ("random_fold", "ood_fold"):
        training, validation = get_split(ctx["train"], ctx["splits"], fold)
        model, result, probabilities = fit_and_measure(training, validation, ctx["config"])
        result["fold"] = fold
        save_predictions(ctx["output"] / f"baseline_{fold}_predictions.csv",
                         validation, probabilities)
        if fold == "random_fold":
            cache_file = ctx["cache"] / "supervised_random.joblib"
            joblib.dump(model, cache_file, compress=3)
            result["cache_sha256"] = sha256(cache_file)
            result["feature_fingerprint"] = feature_fingerprint(model.named_steps["features"])
        results[fold] = result
        del model
        gc.collect()
    write_json(ctx["output"] / "baseline_results.json", results)
    pd.DataFrame([{key: value for key, value in result.items()
                   if key not in ("warnings", "classifier_iterations")}
                  for result in results.values()]).to_csv(
                      ctx["output"] / "baseline_results.csv", index=False, float_format="%.12g")


def load_baseline(ctx):
    if not (ctx["output"] / "baseline_completed.json").exists():
        raise ValueError("Run baseline successfully before Q2 or Q3.")
    results = json.loads((ctx["output"] / "baseline_results.json").read_text())
    baseline = results["random_fold"]
    path = ctx["cache"] / "supervised_random.joblib"
    if sha256(path) != baseline["cache_sha256"]:
        raise ValueError("Locally generated baseline cache has changed.")
    # Only our own freshly generated, hash-checked local model cache is loaded.
    model = joblib.load(path)
    return model, baseline


def plot_learning_curve(frame, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    axis.plot(frame["training_rows"], frame["training_auroc"],
              marker="o", color="#2563eb", label="Training AUROC", linewidth=2)
    axis.plot(frame["training_rows"], frame["validation_auroc"],
              marker="s", color="#d97706", label="Validation AUROC", linewidth=2)
    axis.set(title="Learning curves: TF-IDF + Logistic Regression",
             xlabel="Labeled training examples", ylabel="AUROC")
    axis.set_xticks(frame["training_rows"])
    lower = min(frame["training_auroc"].min(), frame["validation_auroc"].min())
    axis.set_ylim(max(0.45, lower - 0.025), 1.005)
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    figure.suptitle("Fixed validation set: 2,000 examples | C = 4.0 | sampling seed = 42",
                   fontsize=10, color="#475569")
    figure.savefig(destination / "learning_curve.png", dpi=240)
    figure.savefig(destination / "learning_curve.svg")
    # Matplotlib emits harmless trailing spaces in SVG path data; keep diffs clean.
    svg = destination / "learning_curve.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(figure)


def run_q2(ctx):
    model, baseline = load_baseline(ctx)
    subsets = nested_stratified_indices(
        ctx["pool"]["label"], ctx["config"]["training_sizes"],
        ctx["config"]["sampling_seed"]
    )
    selections, records = {}, []
    for size in ctx["config"]["training_sizes"]:
        training = ctx["pool"].iloc[subsets[size]]
        if set(training["id"]) & set(ctx["validation"]["id"]):
            raise ValueError("Learning subset overlaps validation.")
        selections[str(size)] = training["id"].tolist()
        if size == len(ctx["pool"]):
            if id_hash(training["id"]) != baseline["training_ids_sha256"]:
                raise ValueError("Full Q2 subset differs from the baseline training pool.")
            result = dict(baseline)
            probabilities = positive_probabilities(
                model.named_steps["clf"],
                model.named_steps["features"].transform(ctx["validation"]["text"])
            )
            result["result_source"] = "exact_full_pool_baseline_reused"
        else:
            fitted, result, probabilities = fit_and_measure(
                training, ctx["validation"], ctx["config"])
            result["result_source"] = "independent_fit_on_this_subset"
            del fitted
        records.append(result)
        save_predictions(ctx["output"] / f"q2_n{size}_validation_predictions.csv",
                         ctx["validation"], probabilities)
        pd.DataFrame([{key: value for key, value in record.items()
                       if key not in ("warnings", "classifier_iterations")}
                      for record in records]).to_csv(
                          ctx["output"] / "learning_curve_results.csv",
                          index=False, float_format="%.12g")
        gc.collect()
    write_json(ctx["output"] / "q2_subset_ids.json", selections)
    write_json(ctx["output"] / "q2_run_details.json", records)
    plot_learning_curve(pd.DataFrame(records), ctx["output"])


def run_q3(ctx):
    model, baseline = load_baseline(ctx)
    unlabeled = pd.read_csv(ROOT / "data/unlabeled.csv", dtype={"id": "string"})
    if len(unlabeled) != 3000 or not unlabeled["id"].is_unique:
        raise ValueError("Unexpected unlabeled dataset.")
    if set(unlabeled["id"]) & set(ctx["train"]["id"]):
        raise ValueError("Unlabeled IDs overlap labeled data.")
    features = model.named_steps["features"]
    feature_before = feature_fingerprint(features)
    started = time.perf_counter()
    x_unlabeled = features.transform(unlabeled["text"])
    probabilities = positive_probabilities(model.named_steps["clf"], x_unlabeled)
    selected, pseudo_labels = select_pseudo_labels(
        probabilities, ctx["config"]["pseudo_label_low"], ctx["config"]["pseudo_label_high"])
    count = int(selected.sum())
    print(f"  Fixed thresholds selected {count:,}/{len(unlabeled):,} unlabeled rows; "
          f"pseudo-human={int((pseudo_labels == 0).sum())}, "
          f"pseudo-AI={int((pseudo_labels == 1).sum())}", flush=True)
    selected_frame = pd.DataFrame({
        "id": unlabeled.loc[selected, "id"].to_numpy(),
        "pseudo_label": pseudo_labels, "initial_ai_probability": probabilities[selected],
    })
    selected_frame.to_csv(ctx["output"] / "q3_selected_pseudo_labels.csv",
                          index=False, float_format="%.12g")
    classifier = clone(model.named_steps["clf"])
    captured_records = []
    classifier_fit_seconds = 0.0
    if count:
        x_labeled = features.transform(ctx["pool"]["text"])
        x_augmented = sparse.vstack([x_labeled, x_unlabeled[selected]], format="csr")
        del x_labeled, x_unlabeled
        y_augmented = np.concatenate([ctx["pool"]["label"].to_numpy(), pseudo_labels])
        print(f"  Refit classifier only on {len(y_augmented):,} sparse rows; TF-IDF frozen",
              flush=True)
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            classifier.fit(x_augmented, y_augmented)
        classifier_fit_seconds = time.perf_counter() - fit_started
        captured_records = [{"category": item.category.__name__, "message": str(item.message)}
                            for item in captured]
        del x_augmented, y_augmented
    else:
        classifier = model.named_steps["clf"]
        del x_unlabeled
    x_validation = features.transform(ctx["validation"]["text"])
    semi_probabilities = positive_probabilities(classifier, x_validation)
    semi_auc = float(roc_auc_score(ctx["validation"]["label"], semi_probabilities))
    feature_after = feature_fingerprint(features)
    if feature_before != feature_after or feature_after != baseline["feature_fingerprint"]:
        raise ValueError("The Q3 experiment changed the frozen TF-IDF representation.")
    delta = semi_auc - baseline["validation_auroc"]
    details = {
        "low_threshold": ctx["config"]["pseudo_label_low"],
        "high_threshold": ctx["config"]["pseudo_label_high"],
        "rounds": 1, "pseudo_label_weight": 1.0,
        "labeled_training_rows": len(ctx["pool"]),
        "unlabeled_rows": len(unlabeled), "selected_rows": count,
        "pseudo_human_rows": int((pseudo_labels == 0).sum()),
        "pseudo_ai_rows": int((pseudo_labels == 1).sum()),
        "selected_unlabeled_ids_sha256": id_hash(selected_frame["id"]),
        "validation_rows": len(ctx["validation"]),
        "validation_ids_sha256": id_hash(ctx["validation"]["id"]),
        "supervised_validation_auroc": baseline["validation_auroc"],
        "with_unlabeled_validation_auroc": semi_auc,
        "delta_validation_auroc": delta,
        "classifier_fit_seconds": classifier_fit_seconds,
        "total_q3_seconds": time.perf_counter() - started,
        "classifier_iterations": classifier.n_iter_.tolist(),
        "feature_fingerprint_before": feature_before,
        "feature_fingerprint_after": feature_after,
        "features_unchanged": feature_before == feature_after,
        "initial_probability_quantiles": np.quantile(probabilities, [0, .05, .25, .5, .75, .95, 1]),
        "warnings": captured_records,
        "no_selection_action": "baseline retained" if not count else None,
    }
    write_json(ctx["output"] / "q3_run_details.json", details)
    pd.DataFrame([
        {"method": "Supervised only", "labeled_rows": 8000, "pseudo_rows": 0,
         "validation_auroc": baseline["validation_auroc"], "delta_vs_supervised": 0.0},
        {"method": "With unlabeled data", "labeled_rows": 8000, "pseudo_rows": count,
         "validation_auroc": semi_auc, "delta_vs_supervised": delta},
    ]).to_csv(ctx["output"] / "pseudo_label_results.csv", index=False, float_format="%.12g")
    save_predictions(ctx["output"] / "q3_validation_predictions.csv",
                     ctx["validation"], semi_probabilities)
    print(f"  Q3 validation AUROC={semi_auc:.6f}, delta={delta:+.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["verify", "baseline", "q2", "q3"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cache_root = ROOT / ".person_b_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "cache"))
    ctx = load_context(args.config, args.output_dir)
    if args.stage == "verify":
        print(json.dumps({"protocol_signature": ctx["signature"],
                          "training_rows": len(ctx["pool"]),
                          "validation_rows": len(ctx["validation"]),
                          "model_training_run": False}, indent=2))
        return
    with threadpool_limits(limits=ctx["config"]["threads"]):
        with stage(ctx, args.stage):
            {"baseline": run_baseline, "q2": run_q2, "q3": run_q3}[args.stage](ctx)


if __name__ == "__main__":
    main()
