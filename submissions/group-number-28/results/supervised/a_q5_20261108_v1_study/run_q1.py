"""Reproduce Q1 without fitting any component to the held-out partition.

Run from the repository root: python -m experiments.supervised.run_q1 --prefix ...
Each prefix is immutable. The full experiment (including its code) is snapshotted.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
import warnings

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_info, threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, SPLIT_DIR, load_partitions, sha256, write_csv

HERE = Path(__file__).resolve().parent
FAMILIES = ("lr", "nb", "svm")
NAMES = {"lr": "Logistic regression", "nb": "Multinomial Naive Bayes", "svm": "Calibrated linear SVM"}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def vectorizer(plan):
    settings = dict(plan["representation"])
    settings["ngram_range"] = tuple(settings["ngram_range"])
    settings["dtype"] = np.float64
    return TfidfVectorizer(**settings)


def estimator(family, value, seed):
    if family == "lr":
        return LogisticRegression(C=value, solver="liblinear", penalty="l2", tol=1e-5,
                                  max_iter=2000, random_state=seed)
    if family == "nb":
        return MultinomialNB(alpha=value, force_alpha=True)
    return LinearSVC(C=value, dual="auto", tol=1e-5, max_iter=10000, random_state=seed)


def pipeline(plan, family, value):
    return Pipeline([("tfidf", vectorizer(plan)),
                     ("classifier", estimator(family, value, plan["seeds"]["model"]))])


def make_model(plan, family, value):
    base = pipeline(plan, family, value)
    if family != "svm":
        return base
    cv = StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["calibration"])
    return CalibratedClassifierCV(base, method="sigmoid", cv=cv, ensemble=False, n_jobs=1)


def clean_params(model):
    """Record every sklearn default as well as intentional settings."""
    result = {}
    for key, value in model.get_params(deep=True).items():
        try:
            json.dumps(value)
            result[key] = value
        except TypeError:
            result[key] = repr(value)
    return result


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    paths = [HERE / "run_q1.py", HERE / "q1_plan.json", HERE / "requirements.txt",
             ROOT / "human_ai/data.py", ROOT / "human_ai/check_predictions.py"]
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "git_status": git("status", "--porcelain"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "thread_limit": 1,
        "dependencies": {p: importlib.metadata.version(p) for p in
                         ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "matplotlib")},
        "threadpools": threadpool_info(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def id_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def predict_ai(model, text):
    index = list(model.classes_).index(1)
    return model.predict_proba(text)[:, index]


def score(y, probabilities):
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "accuracy_at_0_5": float(accuracy_score(y, probabilities >= 0.5)),
        "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities)),
    }


def tune(plan, train, study):
    text = [r["text"] for r in train]
    y = np.array([int(r["label"]) for r in train])
    splitter = StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["inner_cv"])
    folds = list(splitter.split(text, y))
    fold_of_row = np.empty(len(train), dtype=int)
    for fold, (_, held) in enumerate(folds):
        fold_of_row[held] = fold
    write_csv(study / "inner_folds.csv", ["id", "inner_validation_fold"],
              ({"id": row["id"], "inner_validation_fold": int(fold_of_row[i])}
               for i, row in enumerate(train)))
    observations = []
    for fold, (fit, held) in enumerate(folds):
        tfidf = vectorizer(plan)
        start = time.perf_counter()
        x_fit = tfidf.fit_transform([text[i] for i in fit])
        feature_seconds = time.perf_counter() - start
        x_held = tfidf.transform([text[i] for i in held])
        for family in FAMILIES:
            for value in plan["models"][family]["candidates"]:
                model = estimator(family, value, plan["seeds"]["model"])
                start = time.perf_counter()
                model.fit(x_fit, y[fit])
                seconds = time.perf_counter() - start
                values = (model.decision_function(x_held) if family != "nb"
                          else predict_ai(model, x_held))
                auc = float(roc_auc_score(y[held], values))
                observations.append({"model": family, "parameter_value": value, "fold": fold,
                                     "auroc": auc, "classifier_fit_seconds": seconds,
                                     "feature_fit_seconds": feature_seconds,
                                     "n_features": len(tfidf.vocabulary_),
                                     "fit_ids_sha256": id_hash([train[i]["id"] for i in fit]),
                                     "held_ids_sha256": id_hash([train[i]["id"] for i in held])})
                save_json(study / "cv_progress.json", observations)
                print(f"inner fold={fold} {family} value={value:g} AUROC={auc:.6f}", flush=True)
    results, chosen = [], {}
    for family in FAMILIES:
        candidates = []
        for value in plan["models"][family]["candidates"]:
            rows = [r for r in observations if r["model"] == family and r["parameter_value"] == value]
            aucs = [r["auroc"] for r in rows]
            candidates.append({"model": family, "parameter": plan["models"][family]["parameter"],
                               "value": value, "mean_cv_auroc": float(np.mean(aucs)),
                               "std_cv_auroc": float(np.std(aucs, ddof=1)),
                               **{f"fold_{i}_auroc": auc for i, auc in enumerate(aucs)}})
        chosen[family] = max(candidates, key=lambda r: r["mean_cv_auroc"])["value"]
        results.extend(candidates)
    write_csv(study / "cv_results.csv", list(results[0]), results)
    save_json(study / "chosen_parameters.json", chosen)
    return chosen


def bootstrap(y, predictions, winner, seed):
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(y == value) for value in (0, 1)]
    samples = {family: [] for family in predictions}
    for _ in range(1000):
        ix = np.concatenate([rng.choice(group, len(group), replace=True) for group in indices])
        for family, p in predictions.items():
            samples[family].append(float(roc_auc_score(y[ix], p[ix])))
    result = {"seed": seed, "replicates": 1000, "method": "stratified paired percentile bootstrap",
              "limitations": "conditional on these fitted models and split; not adjusted for model selection; no training variability",
              "auroc_95_intervals": {}, "winner_minus_other_95_intervals": {}}
    for family in predictions:
        result["auroc_95_intervals"][family] = np.quantile(samples[family], [0.025, 0.975]).tolist()
        if family != winner:
            delta = np.array(samples[winner]) - samples[family]
            result["winner_minus_other_95_intervals"][family] = np.quantile(delta, [0.025, 0.975]).tolist()
    return result


def run(prefix):
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in prefix):
        raise ValueError("Prefix must contain only letters, numbers, '_' and '-'")
    plan = json.loads((HERE / "q1_plan.json").read_text())
    base = ROOT / "results/supervised"
    study = base / f"{prefix}_study"
    paths = [study] + [base / f"{prefix}_{family}" for family in FAMILIES]
    if any(p.exists() for p in paths):
        raise FileExistsError("Run prefix already exists; choose a NEW prefix, never overwrite evidence")
    study.mkdir(parents=True)
    for name in ("run_q1.py", "q1_plan.json", "requirements.txt"):
        shutil.copy2(HERE / name, study / name)
    context = provenance()
    save_json(study / "provenance.json", context)
    train, validation = load_partitions()
    split = json.loads((SPLIT_DIR / "manifest.json").read_text())
    # Copy IDs (never passages) so artifacts specify every actual fitting row.
    shutil.copy2(SPLIT_DIR / "train_ids.csv", study / "train_ids.csv")
    shutil.copy2(SPLIT_DIR / "validation_ids.csv", study / "validation_ids.csv")
    save_json(study / "split_manifest.json", split)
    started = time.perf_counter()
    chosen = tune(plan, train, study)
    tuning_seconds = time.perf_counter() - started
    train_text, val_text = [r["text"] for r in train], [r["text"] for r in validation]
    train_y = np.array([int(r["label"]) for r in train])
    val_y = np.array([int(r["label"]) for r in validation])
    comparison, predictions = [], {}
    for family in FAMILIES:
        output = base / f"{prefix}_{family}"
        output.mkdir()
        model = make_model(plan, family, chosen[family])
        config = {"run_id": output.name, "study_id": study.name, "family": family,
                  "chosen_value": chosen[family], "plan": plan, "split": split,
                  "train_ids_file": f"../{study.name}/train_ids.csv",
                  "validation_ids_file": f"../{study.name}/validation_ids.csv",
                  "estimator_parameters": clean_params(model), "provenance": context}
        save_json(output / "config.json", config)
        start = time.perf_counter()
        model.fit(train_text, train_y)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        train_p = predict_ai(model, train_text)
        val_p = predict_ai(model, val_text)
        prediction_seconds = time.perf_counter() - start
        predictions[family] = val_p
        base_model = (model.calibrated_classifiers_[0].estimator if family == "svm" else model)
        metrics = {"train": {"n": len(train), **score(train_y, train_p)},
                   "validation": {"n": len(validation), **score(val_y, val_p)},
                   "fit_seconds": fit_seconds, "prediction_seconds_train_plus_validation": prediction_seconds,
                   "timing_definition": "wall seconds; fit includes full TF-IDF and any probability calibration, excludes inner tuning",
                   "n_features": len(base_model.named_steps["tfidf"].vocabulary_),
                   "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                       (1 if platform.system() == "Darwin" else 1024),
                   "memory_note": "cumulative process high-water mark, not isolated per model"}
        save_json(output / "metrics.json", metrics)
        write_csv(output / "validation_predictions.csv", ["id", "label", "p_ai"],
                  ({"id": r["id"], "label": r["label"], "p_ai": format(float(p), ".17g")}
                   for r, p in zip(validation, val_p, strict=True)))
        check(output / "validation_predictions.csv", "validation")
        joblib.dump(model, output / "model.joblib")
        comparison.append({"model": family, "name": NAMES[family], "run_id": output.name,
                           "train_auroc": metrics["train"]["auroc"],
                           "validation_auroc": metrics["validation"]["auroc"],
                           "fit_seconds": fit_seconds, "chosen_value": chosen[family],
                           "n_features": metrics["n_features"]})
        print(f"HELD-OUT {family}: {comparison[-1]}", flush=True)
    write_csv(study / "comparison.csv", list(comparison[0]), comparison)
    winner = max(comparison, key=lambda r: r["validation_auroc"])
    save_json(study / "selection.json", {
        "winner": winner["model"], "winner_run_id": winner["run_id"],
        "criterion": plan["selection"]["family_selection"],
        "scope": plan["selection"]["scope"], "tuning_seconds_all_families": tuning_seconds,
        "total_experiment_wall_seconds": time.perf_counter() - started,
        "test_or_leaderboard_used": False, "full_10000_refit_performed": False,
    })
    save_json(study / "bootstrap.json", bootstrap(val_y, predictions, winner["model"], plan["seeds"]["bootstrap"]))
    print(f"Finished {study.name}; Q1 candidate={winner['run_id']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    # A failed convergence check stops the experiment instead of silently producing scores.
    warnings.filterwarnings("error", category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        run(args.prefix)


if __name__ == "__main__":
    main()
