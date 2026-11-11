"""Run the frozen four-condition Q5 ablation with atomic resumable checkpoints.

No API calls, network access, Kaggle uploads, or test-set fitting are used.
Run with a new --prefix; --resume only continues the identical recorded study.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_info, threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, SPLIT_DIR, load_partitions, read_csv, sha256
from .run_q1 import clean_params, id_hash, predict_ai, score

HERE = Path(__file__).resolve().parent
MODEL_KEYS = ("w60", "c60", "wc120", "w120")
SOURCE_PATHS = (
    "experiments/supervised/run_q5.py", "experiments/supervised/q5_plan.json",
    "experiments/supervised/q1_plan.json", "experiments/supervised/run_q1.py",
    "experiments/supervised/requirements.txt", "human_ai/data.py",
    "human_ai/check_predictions.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path, fields, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_plan():
    return json.loads((HERE / "q5_plan.json").read_text())


def feature_transformer(model_key, plan):
    settings = plan["models"][model_key]
    blocks = []
    for kind in ("word", "char"):
        key = f"{kind}_max_features"
        if key in settings:
            params = dict(plan[f"{kind}_representation"])
            params.update(max_features=settings[key], dtype=np.float64)
            params["ngram_range"] = tuple(params["ngram_range"])
            blocks.append((kind, TfidfVectorizer(**params)))
    if len(blocks) == 1:
        return blocks[0][1]
    return Pipeline([("union", FeatureUnion(blocks, n_jobs=1)),
                     ("normalize", Normalizer(norm="l2", copy=False))])


def make_model(model_key, plan, calibrated=True):
    classifier = LinearSVC(random_state=plan["seeds"]["model"], **plan["classifier"])
    model = Pipeline([("features", feature_transformer(model_key, plan)),
                      ("classifier", classifier)])
    if not calibrated:
        return model
    cv = StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["calibration"])
    return CalibratedClassifierCV(model, method="sigmoid", cv=cv, ensemble=False, n_jobs=1)


def feature_counts(fitted_features):
    if isinstance(fitted_features, TfidfVectorizer):
        kind = fitted_features.analyzer
        count = len(fitted_features.vocabulary_)
        return {"total": count, "word": count if kind == "word" else 0,
                "char": count if kind == "char" else 0}
    counts = {name: len(vec.vocabulary_) for name, vec in
              fitted_features.named_steps["union"].transformer_list}
    return {"total": sum(counts.values()), **counts}


def fit_record(model, text, labels):
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(text, labels)
    notes = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
    if any(issubclass(w.category, ConvergenceWarning) for w in captured):
        raise RuntimeError("Convergence warning: do not silently treat an unfinished fit as evidence")
    return time.perf_counter() - started, notes


def record_status(study, stage, **details):
    old = json.loads((study / "status.json").read_text()) if (study / "status.json").exists() else {}
    old.update(pid=os.getpid(), stage=stage, updated_at_utc=utc_now(), **details)
    atomic_json(study / "status.json", old)
    print(f"[{utc_now()}] {stage}", flush=True)


def provenance():
    from importlib.metadata import version
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {
        "started_at_utc": utc_now(), "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")), "git_status": git("status", "--porcelain"),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_PATHS},
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {name: version(name) for name in
                         ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "matplotlib")},
        "thread_limit": 1, "threadpools": threadpool_info(),
        "pid_at_launch": os.getpid(),
    }


def initialize(prefix, resume):
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in prefix):
        raise ValueError("Use a nonempty alphanumeric/underscore/hyphen prefix")
    study = ROOT / "results" / "supervised" / f"{prefix}_study"
    plan = load_plan()
    original = json.loads((HERE / "q1_plan.json").read_text())
    if plan["word_representation"] != original["representation"]:
        raise ValueError("Q5 word baseline does not match the recorded Q1 representation")
    if study.exists():
        if not resume:
            raise FileExistsError("Study already exists; use --resume only for this identical study")
        if json.loads((study / "q5_plan.json").read_text()) != plan:
            raise ValueError("Cannot resume a changed plan")
        context = json.loads((study / "provenance.json").read_text())
        for name, expected in context["source_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Cannot resume changed source: {name}")
        current = provenance()
        for field in ("python", "dependencies"):
            if current[field] != context[field]:
                raise ValueError(f"Cannot resume a changed environment: {field}")
    else:
        if resume:
            raise FileNotFoundError("No existing study to resume")
        if any((study.parent / f"{prefix}_{key}").exists() for key in MODEL_KEYS):
            raise FileExistsError("One or more model output directories already exist")
        study.mkdir(parents=True)
        context = provenance()
        atomic_json(study / "provenance.json", context)
        for name in SOURCE_PATHS:
            target = study / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            if name.startswith("experiments/supervised/"):
                shutil.copy2(ROOT / name, study / Path(name).name)
        for name in ("train_ids.csv", "validation_ids.csv", "manifest.json"):
            shutil.copy2(SPLIT_DIR / name, study / ("split_manifest.json" if name == "manifest.json" else name))
    split = json.loads((SPLIT_DIR / "manifest.json").read_text())
    if json.loads((study / "split_manifest.json").read_text()) != split:
        raise ValueError("Cannot resume a changed split")
    return study, plan, context, split


def fit_final(study, prefix, key, plan, context, split, text, labels):
    folder = study.parent / f"{prefix}_{key}"
    folder.mkdir(exist_ok=True)
    record_path = folder / "model_fit.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if sha256(folder / "model.joblib") != record["model_sha256"]:
            raise ValueError(f"Changed checkpoint model: {key}")
        return record
    record_status(study, f"fitting_{key}", active_model=key)
    model = make_model(key, plan)
    seconds, notes = fit_record(model, text, labels)
    temporary = folder / "model.joblib.tmp"
    joblib.dump(model, temporary)
    os.replace(temporary, folder / "model.joblib")
    record = {"fit_seconds": seconds, "warnings": notes, "model_sha256": sha256(folder / "model.joblib"),
              "completed_at_utc": utc_now()}
    config = {"run_id": folder.name, "study_id": study.name, "model": key,
              "settings": plan["models"][key], "plan": plan, "split": split,
              "estimator_parameters": clean_params(model), "provenance": context,
              "train_ids_file": f"../{study.name}/train_ids.csv",
              "validation_ids_file": f"../{study.name}/validation_ids.csv"}
    atomic_json(folder / "config.json", config)
    atomic_json(record_path, record)
    print(f"Saved fitted {key}: {seconds:.2f}s", flush=True)
    return record


def baseline_gate(study, prefix, plan, validation):
    path = study / "baseline_gate.json"
    reference_path = ROOT / plan["baseline_reference"]
    if path.exists():
        gate = json.loads(path.read_text())
        if gate["reference_sha256"] != sha256(reference_path) or not gate["passed"]:
            raise ValueError("Changed or failed baseline reference")
        return
    record_status(study, "checking_q1_baseline")
    reference = read_csv(reference_path)
    if [(r["id"], r["label"]) for r in reference] != [(r["id"], r["label"]) for r in validation]:
        raise ValueError("Baseline reference IDs/labels differ")
    model = joblib.load(study.parent / f"{prefix}_w60" / "model.joblib")
    probabilities = predict_ai(model, [r["text"] for r in validation])
    saved = np.array([float(r["p_ai"]) for r in reference])
    error = float(np.max(np.abs(probabilities - saved)))
    if error > plan["evaluation"]["baseline_tolerance"]:
        raise ValueError(f"W60 differs from Q1 by {error}; stop before new comparisons")
    atomic_json(path, {"passed": True, "reference_predictions": plan["baseline_reference"],
                      "reference_sha256": sha256(reference_path), "max_probability_error": error,
                      "validation_auroc": float(roc_auc_score([int(r["label"]) for r in validation], probabilities)),
                      "completed_at_utc": utc_now()})
    print(f"Baseline gate passed; max Q1 probability error={error:g}", flush=True)


def run_cv(study, plan, train):
    text = [r["text"] for r in train]
    labels = np.array([int(r["label"]) for r in train])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["inner_cv"]).split(text, labels))
    assignment = np.empty(len(train), dtype=int)
    for fold, (_, held) in enumerate(folds):
        assignment[held] = fold
    atomic_csv(study / "inner_folds.csv", ["id", "inner_validation_fold"],
               ({"id": row["id"], "inner_validation_fold": int(assignment[i])} for i, row in enumerate(train)))
    path = study / "cv_progress.json"
    observations = json.loads(path.read_text()) if path.exists() else []
    done = {(r["model"], r["fold"]) for r in observations}
    if len(done) != len(observations):
        raise ValueError("Duplicate CV checkpoints")
    for row in observations:
        if row["model"] not in MODEL_KEYS or row["fold"] not in range(3):
            raise ValueError("Unknown model/fold in CV checkpoint")
        fit, held = folds[row["fold"]]
        if (row["fit_ids_sha256"] != id_hash([train[i]["id"] for i in fit]) or
                row["held_ids_sha256"] != id_hash([train[i]["id"] for i in held]) or
                not np.isfinite(row["auroc"]) or not 0 <= row["auroc"] <= 1):
            raise ValueError("Invalid fitting IDs or score in CV checkpoint")
    for key in MODEL_KEYS:
        for fold, (fit, held) in enumerate(folds):
            if (key, fold) in done:
                continue
            record_status(study, f"cv_{key}_fold_{fold}", completed_cv_fits=len(observations))
            model = make_model(key, plan, calibrated=False)
            seconds, notes = fit_record(model, [text[i] for i in fit], labels[fit])
            auc = float(roc_auc_score(labels[held], model.decision_function([text[i] for i in held])))
            observations.append({"model": key, "fold": fold, "auroc": auc,
                                 "fit_seconds": seconds, "n_features": feature_counts(model.named_steps["features"])["total"],
                                 "fit_ids_sha256": id_hash([train[i]["id"] for i in fit]),
                                 "held_ids_sha256": id_hash([train[i]["id"] for i in held]),
                                 "warning_count": len(notes), "warnings": notes})
            atomic_json(path, observations)
            print(f"CV checkpoint {len(observations)}/12: {key} fold={fold} AUROC={auc:.6f}", flush=True)
            del model
    summary = []
    for key in MODEL_KEYS:
        scores = [next(r["auroc"] for r in observations if r["model"] == key and r["fold"] == f) for f in range(3)]
        summary.append({"model": key, "mean_cv_auroc": float(np.mean(scores)), "std_cv_auroc": float(np.std(scores, ddof=1)),
                        **{f"fold_{fold}_auroc": value for fold, value in enumerate(scores)}})
    atomic_csv(study / "cv_results.csv", list(summary[0]), summary)
    return {row["model"]: row for row in summary}


def evaluate(study, prefix, plan, train, validation, cv):
    record_status(study, "evaluating_frozen_models")
    x, z = [r["text"] for r in train], [r["text"] for r in validation]
    y, vy = np.array([int(r["label"]) for r in train]), np.array([int(r["label"]) for r in validation])
    comparison, predictions = [], {}
    for key in MODEL_KEYS:
        folder = study.parent / f"{prefix}_{key}"
        fit = json.loads((folder / "model_fit.json").read_text())
        complete = folder / "completed.json"
        if complete.exists():
            receipt = json.loads(complete.read_text())
            for name, expected in receipt["artifact_sha256"].items():
                if sha256(folder / name) != expected:
                    raise ValueError(f"Changed completed artifact: {folder / name}")
            metrics = json.loads((folder / "metrics.json").read_text())
            probability = np.array([float(r["p_ai"]) for r in read_csv(folder / "validation_predictions.csv")])
        else:
            model = joblib.load(folder / "model.joblib")
            started = time.perf_counter()
            train_probability, probability = predict_ai(model, x), predict_ai(model, z)
            prediction_seconds = time.perf_counter() - started
            fitted_features = model.calibrated_classifiers_[0].estimator.named_steps["features"]
            counts = feature_counts(fitted_features)
            transformed = fitted_features.transform(x[:64])
            norms = np.sqrt(transformed.multiply(transformed).sum(axis=1)).A1
            if not np.allclose(norms[norms > 0], 1, atol=1e-12):
                raise ValueError(f"Unexpected feature scaling for {key}")
            metrics = {"train": {"n": len(train), **score(y, train_probability)},
                       "validation": {"n": len(validation), **score(vy, probability)},
                       "fit_seconds": fit["fit_seconds"], "prediction_seconds_train_plus_validation": prediction_seconds,
                       "timing_definition": "one CPU thread; full feature fitting and sigmoid calibration; CV excluded",
                       "n_features": counts["total"], "feature_counts": counts,
                       "warnings": fit["warnings"], "norm_check_max_abs_error": float(np.max(abs(norms - 1))),
                       "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                       "memory_note": "cumulative process high-water mark, not isolated model memory"}
            atomic_json(folder / "metrics.json", metrics)
            atomic_csv(folder / "validation_predictions.csv", ["id", "label", "p_ai"],
                       ({"id": r["id"], "label": r["label"], "p_ai": format(float(p), ".17g")}
                        for r, p in zip(validation, probability, strict=True)))
            check(folder / "validation_predictions.csv", "validation")
            names = ("config.json", "metrics.json", "validation_predictions.csv", "model.joblib", "model_fit.json")
            atomic_json(complete, {"completed_at_utc": utc_now(),
                                   "artifact_sha256": {name: sha256(folder / name) for name in names}})
            del model
        predictions[key] = probability
        comparison.append({"model": key, "name": plan["models"][key]["name"], "run_id": folder.name,
                           "representation": plan["models"][key]["representation"], "n_features": metrics["n_features"],
                           "train_auroc": metrics["train"]["auroc"], "validation_auroc": metrics["validation"]["auroc"],
                           "fit_seconds": metrics["fit_seconds"], "cv_mean_auroc": cv[key]["mean_cv_auroc"],
                           "cv_std_auroc": cv[key]["std_cv_auroc"]})
        print(f"FROZEN VALIDATION {key}: AUROC={metrics['validation']['auroc']:.6f}", flush=True)
    atomic_csv(study / "comparison.csv", list(comparison[0]), comparison)
    return vy, predictions, comparison


def bootstrap(study, plan, labels, predictions):
    path = study / "bootstrap.json"
    prefix = study.name.removesuffix("_study")
    prediction_hashes = {key: sha256(study.parent / f"{prefix}_{key}" / "validation_predictions.csv") for key in MODEL_KEYS}
    if path.exists():
        if json.loads(path.read_text())["prediction_sha256"] != prediction_hashes:
            raise ValueError("Bootstrap checkpoint belongs to different predictions")
        return
    record_status(study, "paired_bootstrap")
    rng = np.random.default_rng(plan["seeds"]["bootstrap"])
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    samples = {key: [] for key in MODEL_KEYS}
    for _ in range(plan["evaluation"]["bootstrap_replicates"]):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for key in MODEL_KEYS:
            samples[key].append(float(roc_auc_score(labels[indices], predictions[key][indices])))
    differences = {key: np.array(samples[key]) - samples["w60"] for key in MODEL_KEYS}
    primary = float(roc_auc_score(labels, predictions["wc120"]) - roc_auc_score(labels, predictions["w60"]))
    result = {"method": "stratified paired percentile bootstrap", "seed": plan["seeds"]["bootstrap"],
              "prediction_sha256": prediction_hashes,
              "replicates": plan["evaluation"]["bootstrap_replicates"],
              "primary_comparison": ["wc120", "w60"], "primary_delta": primary,
              "primary_delta_95_interval": np.quantile(differences["wc120"], [0.025, 0.975]).tolist(),
              "auroc_95_intervals": {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in samples.items()},
              "deltas_vs_w60_95_intervals": {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in differences.items()},
              "wc120_minus_w120_95_interval": np.quantile(np.array(samples["wc120"]) - samples["w120"], [0.025, 0.975]).tolist(),
              "limitations": "development validation; conditional on fitted models and split; no training variation or model-selection adjustment"}
    atomic_json(path, result)


def error_diagnostics(study, plan, train, validation, labels, predictions):
    threshold = plan["evaluation"]["error_threshold"]
    lengths = np.array([len(r["text"].split()) for r in train])
    cuts = np.quantile(lengths, [1/3, 2/3]).tolist()
    val_lengths = np.array([len(r["text"].split()) for r in validation])
    bins = np.digitize(val_lengths, cuts, right=True)
    subgroup = []
    for b in range(3):
        mask = bins == b
        for key in MODEL_KEYS:
            subgroup.append({"length_bin": b, "model": key, "n": int(mask.sum()),
                             "n_human": int(np.sum(labels[mask] == 0)), "n_ai": int(np.sum(labels[mask] == 1)),
                             "validation_auroc": float(roc_auc_score(labels[mask], predictions[key][mask])) if len(set(labels[mask])) == 2 else None})
    atomic_csv(study / "length_subgroups.csv", list(subgroup[0]), subgroup)
    old_correct = (predictions["w60"] >= threshold) == labels
    new_correct = (predictions["wc120"] >= threshold) == labels
    rows = []
    for i, row in enumerate(validation):
        if old_correct[i] == new_correct[i]:
            continue
        rows.append({"id": row["id"], "label": row["label"],
                     "w60_p_ai": format(float(predictions["w60"][i]), ".17g"),
                     "wc120_p_ai": format(float(predictions["wc120"][i]), ".17g"),
                     "change": "corrected" if new_correct[i] else "new_error",
                     "error_type": "FP" if labels[i] == 0 else "FN", "word_count": int(val_lengths[i]),
                     "length_bin": int(bins[i])})
    fields = ["id", "label", "w60_p_ai", "wc120_p_ai", "change", "error_type", "word_count", "length_bin"]
    atomic_csv(study / "changed_errors.csv", fields, rows)
    atomic_json(study / "error_change_summary.json", {
        "threshold": threshold, "length_cutpoints_from_training_only": cuts,
        "length_bin_definition": "bin 0: length <= cut0; bin 1: cut0 < length <= cut1; bin 2: length > cut1",
        "w60_errors": int((~old_correct).sum()), "wc120_errors": int((~new_correct).sum()),
        "corrected": int(np.sum(~old_correct & new_correct)), "new_errors": int(np.sum(old_correct & ~new_correct)),
        "counts_by_change_and_error_type": {f"{change}_{kind}": sum(r["change"] == change and r["error_type"] == kind for r in rows)
                                             for change in ("corrected", "new_error") for kind in ("FP", "FN")},
        "interpretation": "Descriptive diagnostics only; passage-level failure hypotheses require human review. No causal or source claims are inferred."
    })


def run(prefix, resume=False):
    study, plan, context, split = initialize(prefix, resume)
    previous_status = json.loads((study / "status.json").read_text()) if (study / "status.json").exists() else {}
    if previous_status.get("stage") == "complete":
        print("Study already complete; no fits or artifact overwrites performed", flush=True)
        return
    started = time.perf_counter()
    record_status(study, "loading_canonical_data", started_at_utc=previous_status.get("started_at_utc", utc_now()))
    try:
        train, validation = load_partitions()
        x, y = [r["text"] for r in train], np.array([int(r["label"]) for r in train])
        with threadpool_limits(limits=1):
            fit_final(study, prefix, "w60", plan, context, split, x, y)
            baseline_gate(study, prefix, plan, validation)
            cv = run_cv(study, plan, train)
            for key in MODEL_KEYS[1:]:
                fit_final(study, prefix, key, plan, context, split, x, y)
            labels, predictions, comparison = evaluate(study, prefix, plan, train, validation, cv)
            bootstrap(study, plan, labels, predictions)
            error_diagnostics(study, plan, train, validation, labels, predictions)
        atomic_json(study / "selection.json", {
            "primary_comparison": "wc120 minus w60", "nominal_highest_validation_model": max(comparison, key=lambda r: r["validation_auroc"])["model"],
            "scope": "Q5 development evidence; team-wide final choice requires cost, uncertainty and Q3 evidence",
            "test_or_leaderboard_used": False, "new_full_data_candidate_generated": False,
            "all_four_predeclared_conditions_completed": True,
            "wall_seconds_this_invocation": time.perf_counter() - started,
            "validation_previously_inspected_in_q1": True,
        })
        record_status(study, "complete", completed_cv_fits=12, completed_models=list(MODEL_KEYS),
                      completed_at_utc=utc_now(), wall_seconds_this_invocation=time.perf_counter() - started)
    except BaseException as exc:
        record_status(study, "failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.prefix):
        parser.error("Invalid run prefix")
    lock_directory = ROOT / "tmp"
    lock_directory.mkdir(exist_ok=True)
    with (lock_directory / f"{args.prefix}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This study already has a running worker") from exc
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        run(args.prefix, args.resume)


if __name__ == "__main__":
    main()
