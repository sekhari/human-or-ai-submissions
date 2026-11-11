"""Execute the fixed weak-crop protocol with stage gates and no uploads.

First run --action prepare. Supply two blinded review JSON files to --action run
--resume. Scientific settings are read from q5_crop_plan.json and never searched.
"""
import argparse
from datetime import datetime, timezone
from functools import wraps
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import warnings
from unittest.mock import patch

import joblib
import numpy as np
from scipy.sparse import vstack
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, SPLIT_DIR, load_partitions, raw_path, read_csv, sha256
from .crop_model import CropAugmentedClassifier, text_sha256, weighted_training_rows
from .run_q5 import atomic_csv, atomic_json, feature_transformer


SOURCE_FILES = (
    "experiments/supervised/run_q5_crop.py", "experiments/supervised/crop_model.py",
    "experiments/supervised/crop_topics.py", "experiments/supervised/crop_evaluation.py",
    "experiments/supervised/test_q5_crop.py", "experiments/supervised/q5_crop_plan.json",
    "experiments/supervised/run_q5.py", "experiments/supervised/run_q1.py",
    "experiments/supervised/q5_plan.json", "experiments/supervised/q1_plan.json",
    "experiments/supervised/q5_sensitivity.py", "experiments/supervised/requirements.txt",
    "human_ai/data.py", "human_ai/check_predictions.py",
    "docs/next_round_crop_protocol_20261109.md",
)
PRIMARY = ("clean", "quote_and_dash", "digit_runs_to_zero", "whitespace_only", "prefix_100")
SEALED = ("suffix_100", "middle_100")


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(study, stage, **fields):
    record = {"stage": stage, "updated_at_utc": now(), **fields}
    atomic_json(study / "status.json", record)
    print(json.dumps(record), flush=True)


def save_model(path, model):
    temporary = path.with_suffix(".joblib.tmp")
    joblib.dump(model, temporary)
    temporary.replace(path)


def initialize(prefix, resume):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError("Unsafe run prefix")
    plan_path = ROOT / "experiments/supervised/q5_crop_plan.json"
    plan = json.loads(plan_path.read_text())
    for name, expected in plan["expected_reference_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Changed reference: {name}")
    study = ROOT / "results/supervised" / f"{prefix}_study"
    hashes = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    if study.exists():
        if not resume:
            raise FileExistsError("Use a new prefix or identical --resume")
        old = json.loads((study / "provenance.json").read_text())
        if old["source_sha256"] != hashes or json.loads((study / "plan.json").read_text()) != plan:
            raise ValueError("Source/plan changed; do not silently resume")
    else:
        study.mkdir(parents=True)
        provenance = {
            "started_at_utc": now(), "python": platform.python_version(),
            "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
            "source_sha256": hashes, "plan_sha256": sha256(plan_path),
            "user_authorized_execution": True, "threads": 1,
            "synthetic_checks": {"tests_passed": 5, "svc_fits": 7, "maximum_originals_per_fit": 12,
                                 "command": "/opt/anaconda3/bin/python -m unittest experiments.supervised.test_q5_crop -v"},
            "uploads_performed": 0,
        }
        atomic_json(study / "plan.json", plan)
        atomic_json(study / "provenance.json", provenance)
        atomic_json(study / "fit_receipts.json", [])
        for name in SOURCE_FILES:
            target = study / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        for name in ("manifest.json", "train_ids.csv", "validation_ids.csv"):
            shutil.copy2(SPLIT_DIR / name, study / ("split_manifest.json" if name == "manifest.json" else name))
    return study, plan, json.loads((ROOT / plan["reference_plan"]).read_text())


def counted_fit(study, tag, fit_function, details):
    path = study / "fit_receipts.json"
    entries = json.loads(path.read_text())
    if len(entries) >= 24:
        raise RuntimeError("Competition-data SVC fit budget exhausted")
    position = len(entries)
    entries.append({"fit_number": position + 1, "tag": tag, "state": "started", "started_at_utc": now(), **details})
    atomic_json(path, entries)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fitted = fit_function()
        except Exception as error:
            entries[position].update(state="failed", error=repr(error), fit_seconds=time.perf_counter() - started)
            atomic_json(path, entries)
            raise
    notes = [{"category": note.category.__name__, "message": str(note.message)} for note in caught]
    entries[position].update(state="completed", fit_seconds=time.perf_counter() - started, warnings=notes,
                             completed_at_utc=now())
    if hasattr(fitted, "training_weight_receipt_"):
        entries[position]["weight_receipt"] = fitted.training_weight_receipt_
    invalid_convergence = any(issubclass(note.category, ConvergenceWarning) for note in caught)
    if invalid_convergence:
        entries[position]["state"] = "invalid_convergence"
    atomic_json(path, entries)
    if invalid_convergence:
        raise RuntimeError("Convergence warning invalidates this run")
    return fitted


def condition_record(study, folder, train, held_indices, scheme, fold, include_sealed=False):
    from .crop_evaluation import make_conditions, evaluate_margins
    held = [train[i] for i in held_indices]
    labels = np.array([int(row["label"]) for row in held])
    stem = "sealed" if include_sealed else "primary"
    pred_path, metrics_path = folder / f"{stem}_predictions.csv", folder / f"{stem}_metrics.json"
    if not pred_path.exists():
        features = joblib.load(folder / "features.joblib")
        baseline = joblib.load(folder / "baseline.joblib")
        weak = joblib.load(folder / "weak_crop.joblib")
        rows, values = [], {}
        for name, (indices, texts) in make_conditions([r["text"] for r in held], include_sealed=include_sealed).items():
            if name not in (SEALED if include_sealed else PRIMARY):
                continue
            if len(indices):
                matrix = features.transform(texts)
                before, after = baseline.decision_function(matrix), weak.decision_function(matrix)
            else:
                before, after = np.empty(0), np.empty(0)
            values[name] = (indices, before, after)
            rows.extend({"scheme": scheme, "fold": fold, "condition": name, "id": held[i]["id"],
                         "label": int(labels[i]), "baseline_margin_not_probability": format(float(before[j]), ".17g"),
                         "weak_crop_margin_not_probability": format(float(after[j]), ".17g")}
                        for j, i in enumerate(indices))
        columns = ["scheme", "fold", "condition", "id", "label", "baseline_margin_not_probability", "weak_crop_margin_not_probability"]
        atomic_csv(pred_path, columns, rows)
        metrics = evaluate_margins(scheme, fold, labels, values, minimum_per_class=50)
        atomic_json(metrics_path, metrics)
        atomic_json(folder / f"{stem}_receipt.json", {"prediction_sha256": sha256(pred_path),
                                                     "metrics_sha256": sha256(metrics_path)})
    else:
        receipt = json.loads((folder / f"{stem}_receipt.json").read_text())
        if receipt["prediction_sha256"] != sha256(pred_path) or receipt["metrics_sha256"] != sha256(metrics_path):
            raise ValueError("Changed saved predictions")
    by_id = {row["id"]: i for i, row in enumerate(held)}
    grouped = {}
    for row in read_csv(pred_path):
        grouped.setdefault(row["condition"], []).append(row)
    conditions = {}
    for name in (SEALED if include_sealed else PRIMARY):
        rows = grouped.get(name, [])
        conditions[name] = (np.array([by_id[r["id"]] for r in rows], dtype=int),
                            np.array([float(r["baseline_margin_not_probability"]) for r in rows]),
                            np.array([float(r["weak_crop_margin_not_probability"]) for r in rows]))
    return {"fold": fold, "labels": labels, "conditions": conditions}, json.loads(metrics_path.read_text())


def raw_stage(study, train, assignment, scheme, plan, reference, topics=None):
    from .crop_evaluation import aggregate_screen, bootstrap_cv
    all_rows, records = [], []
    labels = np.array([int(r["label"]) for r in train])
    for fold in range(3):
        emit(study, f"{scheme}_fold_{fold}")
        fitting, held = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        folder = study / "folds" / f"{scheme}_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        fit_text = [train[i]["text"] for i in fitting]
        fit_y = labels[fitting]
        expanded, expanded_y, weights, parents = weighted_training_rows(fit_text, fit_y, .1, 150, 100)
        parent_weights = np.bincount(parents, weights=weights, minlength=len(fitting))
        np.testing.assert_allclose(parent_weights, np.ones(len(fitting)), atol=1e-14, rtol=0)
        if not (folder / "feature_fit.json").exists():
            started = time.perf_counter()
            features = feature_transformer("wc120", reference)
            original_x = features.fit_transform(fit_text)
            seconds = time.perf_counter() - started
            save_model(folder / "features.joblib", features)
            atomic_json(folder / "feature_fit.json", {"fit_seconds": seconds, "n_originals": len(fitting),
                        "n_features": original_x.shape[1], "model_sha256": sha256(folder / "features.joblib"),
                        "fitting_ids": [train[i]["id"] for i in fitting],
                        "held_ids": [train[i]["id"] for i in held]})
        else:
            receipt = json.loads((folder / "feature_fit.json").read_text())
            if sha256(folder / "features.joblib") != receipt["model_sha256"]:
                raise ValueError("Changed fold features")
            features = joblib.load(folder / "features.joblib")
            original_x = None
        for name in ("baseline", "weak_crop"):
            model_path, receipt_path = folder / f"{name}.joblib", folder / f"{name}_fit.json"
            if receipt_path.exists():
                if sha256(model_path) != json.loads(receipt_path.read_text())["model_sha256"]:
                    raise ValueError("Changed classifier checkpoint")
                continue
            if original_x is None:
                original_x = features.transform(fit_text)
            if name == "baseline":
                matrix, target, sample_weight = original_x, fit_y, np.ones(len(fitting))
            else:
                matrix = (vstack([original_x, features.transform(expanded[len(fitting):])], format="csr")
                          if len(expanded) > len(fitting) else original_x)
                target, sample_weight = expanded_y, weights
            classifier = LinearSVC(random_state=reference["seeds"]["model"], **reference["classifier"])
            details = {"stage": 1 if scheme == "random" else 2, "scheme": scheme, "fold": fold, "arm": name,
                       "n_originals": len(fitting), "n_training_rows": matrix.shape[0], "n_features": matrix.shape[1],
                       "sample_weight_sum": float(sample_weight.sum()),
                       "fitting_id_sha256": text_sha256("\n".join(train[i]["id"] for i in fitting))}
            fitted = counted_fit(study, f"{scheme}_{fold}_{name}",
                                 lambda: classifier.fit(matrix, target, sample_weight=sample_weight), details)
            save_model(model_path, fitted)
            atomic_json(receipt_path, {"model_sha256": sha256(model_path), **details})
        record, rows = condition_record(study, folder, train, held, scheme, fold)
        if topics is not None:
            record["topics"] = np.asarray(topics)[held]
        records.append(record)
        all_rows.extend(rows)
        print(f"Completed {scheme} fold {fold}: " + json.dumps(rows), flush=True)
    atomic_json(study / f"{scheme}_metrics.json", all_rows)
    if all_rows:
        atomic_csv(study / f"{scheme}_per_fold_metrics.csv", list(all_rows[0]), all_rows)
    gate = aggregate_screen(all_rows, scheme, plan)
    atomic_json(study / f"{scheme}_gate.json", gate)
    emit(study, f"{scheme}_bootstrap")
    intervals = bootstrap_cv(records, scheme, plan)
    atomic_json(study / f"{scheme}_bootstrap.json", intervals)
    return records, all_rows, gate


def terminal(study, reason, **fields):
    entries = json.loads((study / "fit_receipts.json").read_text())
    record = {"completed_at_utc": now(), "outcome": "stopped_no_export", "reason": reason,
              "competition_svc_fits": len(entries), "synthetic_svc_fits": 7,
              "test_text_accessed": False, "new_candidates": 0, "uploads": 0,
              "incumbent_retained": True, **fields}
    atomic_json(study / "completion.json", record)
    emit(study, "complete_negative", **record)


def fit_calibrated(study, folder, originals, crop_weight, reference, plan, stage):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.joblib"
    if (folder / "fit_complete.json").exists():
        receipt = json.loads((folder / "fit_complete.json").read_text())
        if receipt["model_sha256"] != sha256(path):
            raise ValueError("Changed calibrated model")
        return joblib.load(path)
    texts = [r["text"] for r in originals]
    labels = np.array([int(r["label"]) for r in originals])
    allowed = {text_sha256(r["text"]): r["id"] for r in originals}
    cv = StratifiedKFold(3, shuffle=True, random_state=242)
    model = CalibratedClassifierCV(CropAugmentedClassifier(crop_weight, reference), method="sigmoid",
                                   cv=cv, ensemble=False, n_jobs=1)
    original_fit, seen = CropAugmentedClassifier.fit, []

    @wraps(original_fit)
    def tracked_fit(estimator, X, y, sample_weight=None):
        documents = list(X)
        identifiers = [allowed[text_sha256(text)] for text in documents]
        seen.append(identifiers)
        return counted_fit(study, f"{folder.name}_base_{len(seen)}",
                           lambda: original_fit(estimator, documents, y, sample_weight),
                           {"stage": stage, "arm": "weak_crop" if crop_weight else "baseline",
                            "n_originals": len(documents), "fitting_ids": identifiers,
                            "crop_weight": crop_weight})

    with patch.object(CropAugmentedClassifier, "fit", tracked_fit):
        model.fit(texts, labels)
    expected = [{originals[i]["id"] for i in fitting} for fitting, _ in cv.split(texts, labels)] + [{r["id"] for r in originals}]
    if len(seen) != 4 or [set(ids) for ids in seen] != expected:
        raise ValueError("Calibration fitting boundaries do not match original-ID folds")
    save_model(path, model)
    atomic_json(folder / "fit_complete.json", {"model_sha256": sha256(path), "n_originals": len(originals),
                                               "base_fits": 4, "crop_weight": crop_weight})
    atomic_json(folder / "config.json", {"plan": plan, "n_originals": len(originals), "crop_weight": crop_weight,
                                         "calibration_seed": 242, "full_data_heldout_score": None})
    return model


def probability(model, texts):
    values = np.asarray(model.predict_proba(texts)[:, list(model.classes_).index(1)])
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid AI probabilities")
    return values


def calibrated_development(study, prefix, train, validation, plan, reference):
    from .crop_evaluation import make_conditions
    all_text = [r["text"] for r in validation]
    labels = np.array([int(r["label"]) for r in validation])
    declared_conditions = make_conditions(all_text, include_sealed=True)
    counts = {name: {"n_human": int(np.sum(labels[indices] == 0)),
                     "n_ai": int(np.sum(labels[indices] == 1))}
              for name, (indices, _) in declared_conditions.items()}
    if any(min(row.values()) < 50 for row in counts.values()):
        gate = {"passed": False, "reason": "insufficient_condition_class_counts", "class_counts": counts}
        atomic_json(study / "calibrated_gate.json", gate)
        return gate
    models = {}
    for arm, weight in (("baseline", 0.0), ("weak_crop", .1)):
        emit(study, f"calibrated_development_{arm}")
        models[arm] = fit_calibrated(study, study.parent / f"{prefix}_{arm}_8000", train, weight, reference, plan, 3)
    values, rows, predictions = {}, [], []
    for condition, (indices, texts) in declared_conditions.items():
        before, after = (probability(models[arm], texts) for arm in ("baseline", "weak_crop"))
        y = labels[indices]
        values[condition] = (indices, before, after)
        row = {"condition": condition, "n": len(indices), "n_human": int(np.sum(y == 0)), "n_ai": int(np.sum(y == 1)),
               "baseline_auroc": float(roc_auc_score(y, before)), "weak_crop_auroc": float(roc_auc_score(y, after)),
               "delta_auroc": float(roc_auc_score(y, after) - roc_auc_score(y, before))}
        rows.append(row)
        predictions.extend({"id": validation[i]["id"], "label": int(y[j]), "condition": condition,
                            "baseline_p_ai": format(float(before[j]), ".17g"),
                            "weak_crop_p_ai": format(float(after[j]), ".17g")} for j, i in enumerate(indices))
    source = read_csv((ROOT / plan["reference_source_model"]).parent / "validation_predictions.csv")
    source_by_id = {r["id"]: float(r["p_ai"]) for r in source}
    error = float(np.max(np.abs(values["clean"][1] - [source_by_id[r["id"]] for r in validation])))
    if error > 1e-10:
        raise ValueError(f"Baseline reproduction failed: {error}")
    rng, samples = np.random.default_rng(550), {name: [] for name in ("clean", "prefix_100")}
    for _ in range(1000):
        draw = np.concatenate([rng.choice(np.flatnonzero(labels == c), np.sum(labels == c), replace=True) for c in (0, 1)])
        for name in samples:
            indices, before, after = values[name]
            positions = np.full(len(labels), -1, dtype=int)
            positions[indices] = np.arange(len(indices))
            local = positions[draw]
            local = local[local >= 0]
            y = labels[indices][local]
            if len(np.unique(y)) == 2:
                samples[name].append(float(roc_auc_score(y, after[local]) - roc_auc_score(y, before[local])))
    intervals = {name: {"valid_replicates": len(delta), "lower": float(np.quantile(delta, .025)) if len(delta) >= 900 else None,
                         "upper": float(np.quantile(delta, .975)) if len(delta) >= 900 else None}
                 for name, delta in samples.items()}
    by_condition = {r["condition"]: r for r in rows}
    checks = {
        "baseline_reproduction": error <= 1e-10,
        "clean_point": by_condition["clean"]["delta_auroc"] >= -.003,
        "clean_interval": intervals["clean"]["lower"] is not None and intervals["clean"]["lower"] >= -.003,
        "prefix_point": by_condition["prefix_100"]["delta_auroc"] >= .005,
        "prefix_interval": intervals["prefix_100"]["lower"] is not None and intervals["prefix_100"]["lower"] > 0,
        "other_conditions": all(r["delta_auroc"] >= -.003 for r in rows if r["condition"] not in ("clean", "prefix_100")),
        "condition_class_counts": all(min(r["n_human"], r["n_ai"]) >= 50 for r in rows),
    }
    boundaries = np.quantile([len(r["text"].split()) for r in train], [1 / 3, 2 / 3])
    bins = np.digitize([len(t.split()) for t in all_text], boundaries, right=True)
    length_rows = []
    for bin_number in range(3):
        mask = bins == bin_number
        y = labels[mask]
        enough = min(int(np.sum(y == 0)), int(np.sum(y == 1))) >= 30
        delta = float(roc_auc_score(y, values["clean"][2][mask]) - roc_auc_score(y, values["clean"][1][mask])) if enough else None
        length_rows.append({"bin": bin_number, "n": int(mask.sum()), "n_human": int(np.sum(y == 0)),
                            "n_ai": int(np.sum(y == 1)), "delta_auroc": delta})
    checks["length_bins"] = all(r["delta_auroc"] is not None and r["delta_auroc"] >= -.01 for r in length_rows)
    for arm, column in (("baseline", 1), ("weak_crop", 2)):
        folder = study.parent / f"{prefix}_{arm}_8000"
        p = values["clean"][column]
        atomic_csv(folder / "validation_predictions.csv", ["id", "label", "p_ai"],
                   ({"id": r["id"], "label": r["label"], "p_ai": format(float(p[i]), ".17g")} for i, r in enumerate(validation)))
        check(folder / "validation_predictions.csv", "validation")
        atomic_json(folder / "metrics.json", {"validation": {"n": len(labels), "auroc": float(roc_auc_score(labels, p)),
                                                               "log_loss": float(log_loss(labels, p)),
                                                               "brier": float(brier_score_loss(labels, p))}, "conditions": rows})
    atomic_csv(study / "calibrated_condition_predictions.csv", list(predictions[0]), predictions)
    gate = {"passed": all(checks.values()), "checks": checks, "conditions": rows, "intervals": intervals,
            "length_boundaries": boundaries.tolist(), "length_bins": length_rows,
            "baseline_reproduction_max_error": error,
            "probability_spearman": float(spearmanr(values["clean"][1], values["clean"][2]).statistic)}
    atomic_json(study / "calibrated_gate.json", gate)
    return gate


def export_candidate(study, prefix, plan, reference):
    emit(study, "eligible_full_10000_refit")
    all_originals = read_csv(raw_path("train.csv"))
    folder = study.parent / f"{prefix}_weak_crop_full10000"
    model = fit_calibrated(study, folder, all_originals, .1, reference, plan, 4)
    test_path = raw_path("test.csv")
    test = read_csv(test_path)
    probabilities = probability(model, [r["text"] for r in test])
    restored = probability(joblib.load(folder / "model.joblib"), [r["text"] for r in test])
    error = float(np.max(np.abs(probabilities - restored)))
    if error > 1e-12:
        raise ValueError("Reload disagreement")
    path = ROOT / "submissions" / f"{prefix}_weak_crop_full10000.csv"
    if path.exists():
        raise FileExistsError("Candidate already exists; do not overwrite")
    atomic_csv(path, ["id", "label"], ({"id": r["id"], "label": format(float(p), ".17g")} for r, p in zip(test, probabilities)))
    check(path, "submission")
    metadata = {"run_id": folder.name, "category": "exploratory_robustness_tradeoff", "n_originals": 10000,
                "heldout_score": None, "n_test": len(test), "csv_sha256": sha256(path),
                "model_sha256": sha256(folder / "model.joblib"), "test_sha256": sha256(test_path),
                "plan_sha256": sha256(study / "plan.json"), "reload_max_error": error, "uploaded": False}
    atomic_json(path.with_suffix(".metadata.json"), metadata)
    atomic_json(study / "completion.json", {"outcome": "eligible_candidate_exported", "new_candidates": 1,
                "test_text_accessed": True, "uploads": 0, "incumbent_retained": True,
                "competition_svc_fits": len(json.loads((study / "fit_receipts.json").read_text())),
                "submission": str(path.relative_to(ROOT)), "completed_at_utc": now()})
    emit(study, "complete_candidate_exported", submission=str(path.relative_to(ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="a_q5_crop_20261109_v5")
    parser.add_argument("--action", choices=("prepare", "run"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--review-a", type=Path)
    parser.add_argument("--review-b", type=Path)
    args = parser.parse_args()
    study, plan, reference = initialize(args.prefix, args.resume)
    from .crop_topics import prepare_topic_proxy, finalize_topic_proxy
    from .crop_evaluation import sealed_screen
    train, validation = load_partitions()
    with threadpool_limits(limits=1):
        if args.action == "prepare":
            emit(study, "preparing_topic_proxy")
            prepare_topic_proxy(study, train, plan, resume=args.resume)
            emit(study, "awaiting_blind_topic_reviews")
            return
        if not args.review_a or not args.review_b:
            raise ValueError("Two independent blinded review files are required")
        result = finalize_topic_proxy(study, train, plan, json.loads(args.review_a.read_text()), json.loads(args.review_b.read_text()))
        random_records, random_rows, random_gate = raw_stage(study, train, result["random_folds"], "random", plan, reference)
        if not result["topic_valid"]:
            terminal(study, "topic_proxy_invalid_diagnostic_random_only", random_gate=random_gate,
                     topic_quality=result["quality"])
            return
        if not random_gate["passed"]:
            terminal(study, "random_screen_failed", random_gate=random_gate)
            return
        topics = np.array([int(r["topic"]) for r in read_csv(study / "topic_assignments.csv")])
        topic_records, topic_rows, topic_gate = raw_stage(study, train, result["topic_folds"], "topic", plan, reference, topics)
        if not topic_gate["passed"]:
            terminal(study, "topic_screen_failed", random_gate=random_gate, topic_gate=topic_gate)
            return
        emit(study, "revealing_new_conditions")
        sealed_rows = []
        for scheme, assignment, records in (("random", result["random_folds"], random_records),
                                            ("topic", result["topic_folds"], topic_records)):
            for fold in range(3):
                held = np.flatnonzero(assignment == fold)
                record, rows = condition_record(study, study / "folds" / f"{scheme}_{fold}", train, held, scheme, fold, True)
                records[fold]["conditions"].update(record["conditions"])
                sealed_rows.extend(rows)
        atomic_json(study / "sealed_metrics.json", sealed_rows)
        sealed_gate = sealed_screen(sealed_rows, plan)
        atomic_json(study / "sealed_gate.json", sealed_gate)
        if not sealed_gate["passed"]:
            terminal(study, "sealed_conditions_failed", sealed_gate=sealed_gate)
            return
        calibrated_gate = calibrated_development(study, args.prefix, train, validation, plan, reference)
        if not calibrated_gate["passed"]:
            terminal(study, "calibrated_development_failed", calibrated_gate=calibrated_gate)
            return
        export_candidate(study, args.prefix, plan, reference)


if __name__ == "__main__":
    main()
