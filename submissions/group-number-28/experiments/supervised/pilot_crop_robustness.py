"""One post-diagnostic, fixed-feature crop augmentation pilot; no export.

Saved WC120 vocabulary/IDF are frozen in both conditions. Only classifier
training inputs and per-document distribution of sample weights change.
Validation is reused development data; crop weakness motivated this pilot.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from human_ai.data import ROOT, load_partitions, sha256, write_csv
from experiments.supervised.q5_sensitivity import apply_condition
import re


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/supervised/a_public_gap_20261109_v1_crop_pilot")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    train, validation = load_partitions()
    train_text = [row["text"] for row in train]
    text = [row["text"] for row in validation]
    train_y = np.array([int(row["label"]) for row in train])
    y_full = np.array([int(row["label"]) for row in validation])
    crop_train = np.array([i for i, passage in enumerate(train_text) if len(passage.split()) >= 150])
    crop_validation = np.array([i for i, passage in enumerate(text) if len(passage.split()) >= 150])
    model_path = ROOT / "results/supervised/a_q5_20261108_v1_wc120/model.joblib"
    # Plan is persisted before the model is loaded and before either fit.
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Exploratory pilot selected after observing prefix weakness on the reused canonical development validation set; no clean holdout claim.",
        "source_model": str(model_path.relative_to(ROOT)), "source_model_sha256": sha256(model_path),
        "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(__file__),
                          "experiments/supervised/q5_sensitivity.py": sha256(ROOT / "experiments/supervised/q5_sensitivity.py"),
                          "data/splits/manifest.json": sha256(ROOT / "data/splits/manifest.json")},
        "frozen_representation": "Saved 8000-training-row WC120 full-training feature transformer; same learned word/character vocabularies, IDF and L2 geometry for baseline and augmentation. Never refit feature extraction.",
        "baseline": "Refit saved underlying LinearSVC parameters on frozen-feature original 8000 rows, all weights 1.",
        "augmentation": "Original >=150-whitespace-word training docs receive weight 0.5 and an additional first-100-whitespace-word crop joined by one ASCII space receives weight 0.5 with same label; shorter originals retain weight 1. Total sample weight remains 8000.",
        "single_changed_factor": "Weighted inclusion of train-document crops with fixed representation and classifier hyperparameters.",
        "fit_parameters": {"C": 1, "class_weight": None, "dual": "auto", "fit_intercept": True, "intercept_scaling": 1,
                           "loss": "squared_hinge", "max_iter": 10000, "multi_class": "ovr", "penalty": "l2",
                           "random_state": 42, "tol": 1e-5, "verbose": 0},
        "conditions": {"clean": "All 2000 canonical validation rows unchanged.",
                       "quote_and_dash": "Existing frozen quote_typography then dash_typography mappings; same full validation IDs.",
                       "digit_runs_to_zero": "Python Unicode re.sub(r'\\d+', '0', text); same full validation IDs.",
                       "prefix_100_on_min150": "First 100 whitespace words joined by one ASCII space, only canonical validation docs with >=150 whitespace words; equal IDs across models."},
        "n_training_original": len(train), "n_training_crops": len(crop_train), "total_training_sample_weight": len(train),
        "n_validation": len(validation), "n_validation_prefix_subset": len(crop_validation),
        "bootstrap": {"seed": 20261109, "repetitions": 500, "conditions": ["clean", "prefix_100_on_min150"],
                      "method": "Class-stratified paired percentile resampling of validation document IDs; fixed models, no training or selection correction."},
        "score_schema": "Raw LinearSVC decision margins; NOT probabilities; never use this CSV as a submission.",
        "stopping_rule": "Exactly one baseline and one augmented classifier fit; no tuning, calibration, model saving, candidate selection, test prediction or export.",
        "limitations": "Fixed features use all original 8000 training documents in both arms; this isolates an input-augmentation factor for one fixed representation, not a full production training recipe. Cropping changes length, content and whitespace; label-preservation is an authorship hypothesis. Constructed stress gains do not establish hidden-test transfer.",
    }
    save_json(output / "plan.json", plan)
    write_csv(output / "training_crop_ids.csv", ["id"], ({"id": train[i]["id"]} for i in crop_train))
    original_model = joblib.load(model_path)
    base = original_model.calibrated_classifiers_[0].estimator
    features, saved_classifier = base.named_steps["features"], base.named_steps["classifier"]
    if saved_classifier.get_params() != plan["fit_parameters"]:
        raise ValueError("Classifier params differ from frozen plan")
    conditions = {
        "clean": (text, np.arange(len(text))),
        "quote_and_dash": ([apply_condition(apply_condition(t, "quote_typography"), "dash_typography") for t in text], np.arange(len(text))),
        "digit_runs_to_zero": ([re.sub(r"\d+", "0", t) for t in text], np.arange(len(text))),
        "prefix_100_on_min150": ([" ".join(text[i].split()[:100]) for i in crop_validation], crop_validation),
    }
    fits, predictions, rows, prediction_rows = {}, {}, [], []
    with threadpool_limits(limits=1):
        print("Transforming frozen features", flush=True)
        original_x = features.transform(train_text)
        crop_x = features.transform([" ".join(train_text[i].split()[:100]) for i in crop_train])
        matrices = {condition: features.transform(changed) for condition, (changed, _) in conditions.items()}
        original_weights = np.ones(len(train))
        original_weights[crop_train] = 0.5
        augmented_x = vstack([original_x, crop_x], format="csr")
        augmented_y = np.concatenate([train_y, train_y[crop_train]])
        augmented_weights = np.concatenate([original_weights, np.full(len(crop_train), 0.5)])
        assert augmented_weights.sum() == len(train)
        for name, matrix, labels, weights in (("baseline", original_x, train_y, np.ones(len(train))),
                                             ("crop_augmented", augmented_x, augmented_y, augmented_weights)):
            print(f"Fitting {name}: {matrix.shape}", flush=True)
            classifier = clone(saved_classifier)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                classifier.fit(matrix, labels, sample_weight=weights)
            fits[name] = {"fit_seconds": time.perf_counter() - started,
                          "n_training_rows": matrix.shape[0], "n_features": matrix.shape[1],
                          "sample_weight_sum": float(weights.sum()), "n_iter": int(classifier.n_iter_),
                          "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]}
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                save_json(output / "fit_failure.json", fits)
                raise RuntimeError("Convergence warning")
            predictions[name] = {condition: classifier.decision_function(matrix) for condition, matrix in matrices.items()}
        saved_margins = saved_classifier.decision_function(matrices["clean"])
        reproduction_error = float(np.max(np.abs(saved_margins - predictions["baseline"]["clean"])))
        saved_probability_auc = roc_auc_score(y_full, original_model.predict_proba(text)[:, list(original_model.classes_).index(1)])
        baseline_auc = roc_auc_score(y_full, predictions["baseline"]["clean"])
        if reproduction_error > 1e-10 or abs(saved_probability_auc - baseline_auc) > 1e-12:
            raise ValueError(f"Baseline reproduction gate failed: error={reproduction_error}")
        rng = np.random.default_rng(plan["bootstrap"]["seed"])
        for condition, (_, indices) in conditions.items():
            y = y_full[indices]
            before, after = predictions["baseline"][condition], predictions["crop_augmented"][condition]
            row = {"condition": condition, "n": len(indices), "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
                   "baseline_raw_margin_auroc": float(roc_auc_score(y, before)),
                   "augmented_raw_margin_auroc": float(roc_auc_score(y, after)),
                   "augmented_minus_baseline_auroc": float(roc_auc_score(y, after) - roc_auc_score(y, before)),
                   "paired_delta_ci_2_5": None, "paired_delta_ci_97_5": None}
            if condition in plan["bootstrap"]["conditions"]:
                zero, one = np.where(y == 0)[0], np.where(y == 1)[0]
                delta = []
                for _ in range(500):
                    draw = np.concatenate([rng.choice(zero, len(zero), replace=True), rng.choice(one, len(one), replace=True)])
                    delta.append(roc_auc_score(y[draw], after[draw]) - roc_auc_score(y[draw], before[draw]))
                row["paired_delta_ci_2_5"], row["paired_delta_ci_97_5"] = np.quantile(delta, [0.025, 0.975]).tolist()
            rows.append(row)
            prediction_rows.extend({"id": validation[i]["id"], "label": int(y[j]), "condition": condition,
                                    "baseline_margin_not_probability": float(before[j]),
                                    "augmented_margin_not_probability": float(after[j])} for j, i in enumerate(indices))
            print(f"{condition}: {row['baseline_raw_margin_auroc']:.6f} -> {row['augmented_raw_margin_auroc']:.6f}", flush=True)
    write_csv(output / "summary.csv", list(rows[0]), rows)
    write_csv(output / "validation_margins_not_probabilities.csv", list(prediction_rows[0]), prediction_rows)
    save_json(output / "results.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "fit_records": fits,
                                       "baseline_reproduction": {"max_absolute_margin_error": reproduction_error,
                                                                 "saved_probability_auroc": float(saved_probability_auc),
                                                                 "baseline_margin_auroc": float(baseline_auc)},
                                       "rows": rows, "plan_sha256": sha256(output / "plan.json"),
                                       "artifact_sha256": {p.name: sha256(p) for p in output.glob("*.csv")}})


if __name__ == "__main__":
    main()
