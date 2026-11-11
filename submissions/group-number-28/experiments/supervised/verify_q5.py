"""Audit saved Q5 artifacts without refitting the full-data classifiers."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, SPLIT_DIR, load_partitions, read_csv, sha256
from . import run_q5 as q5


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def id_hash(ids):
    import hashlib
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def vectorizers(transformer):
    if isinstance(transformer, TfidfVectorizer):
        return [transformer]
    children = getattr(transformer, "steps", None)
    if children is None:
        children = getattr(transformer, "transformer_list", [])
    return [leaf for _, child in children for leaf in vectorizers(child)]


def verify(prefix, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    base = ROOT / "results/supervised"
    study = base / f"{prefix}_study"
    plan = read_json(study / "q5_plan.json")
    context = read_json(study / "provenance.json")
    for relative, expected in context["source_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"Changed executing source: {relative}")
        require(sha256(study / "source" / relative) == expected, f"Changed complete source snapshot: {relative}")
        frozen = study / Path(relative).name
        if frozen.exists():
            require(sha256(frozen) == expected, f"Changed source snapshot: {relative}")
    for name in ("run_q5.py", "q5_plan.json", "run_q1.py", "q1_plan.json", "requirements.txt"):
        require((study / name).is_file(), f"Missing frozen source: {name}")

    train, validation = load_partitions()
    train_ids = [r["id"] for r in train]
    validation_ids = [r["id"] for r in validation]
    require(not set(train_ids) & set(validation_ids), "Training and validation IDs overlap")
    for name, expected in (("train", train_ids), ("validation", validation_ids)):
        rows = read_csv(study / f"{name}_ids.csv")
        require([r["id"] for r in rows] == expected, f"Noncanonical {name} IDs")
    manifest = read_json(SPLIT_DIR / "manifest.json")
    frozen_manifest = study / "split_manifest.json"
    if frozen_manifest.exists():
        require(read_json(frozen_manifest) == manifest, "Changed split manifest")

    y = np.asarray([int(r["label"]) for r in train])
    vy = np.asarray([int(r["label"]) for r in validation])
    val_text = [r["text"] for r in validation]
    assignments = read_csv(study / "inner_folds.csv")
    require([r["id"] for r in assignments] == train_ids, "Noncanonical CV IDs")
    folds = list(StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["inner_cv"]).split(train_ids, y))
    for fold, (_, held) in enumerate(folds):
        require(all(int(assignments[i]["inner_validation_fold"]) == fold for i in held), "CV fold assignment mismatch")
    progress = read_json(study / "cv_progress.json")
    require(len(progress) == len(q5.MODEL_KEYS) * 3, "Incomplete or duplicate CV records")
    cv = read_csv(study / "cv_results.csv")
    summary = read_csv(study / "comparison.csv")
    require(len(cv) == len(summary) == len(q5.MODEL_KEYS), "Incomplete model summaries")
    require({r["model"] for r in cv} == set(q5.MODEL_KEYS), "Unexpected CV models")
    require({r["model"] for r in summary} == set(q5.MODEL_KEYS), "Unexpected comparison models")
    audit = {
        "source_hashes_match": True,
        "canonical_split_and_fold_ids_match": True,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_cv_fits": len(progress),
        "full_model_refits_performed_by_verifier": False,
        "models": {},
    }
    probabilities = {}
    with threadpool_limits(limits=1):
        for key in q5.MODEL_KEYS:
            observations = [r for r in progress if r["model"] == key]
            require(sorted(r["fold"] for r in observations) == [0, 1, 2], f"Incorrect folds for {key}")
            fold_scores = []
            for fold, (fit, held) in enumerate(folds):
                row = next(r for r in observations if r["fold"] == fold)
                require(row["fit_ids_sha256"] == id_hash([train_ids[i] for i in fit]), f"Incorrect CV fit IDs for {key}")
                require(row["held_ids_sha256"] == id_hash([train_ids[i] for i in held]), f"Incorrect CV held IDs for {key}")
                fold_scores.append(row["auroc"])
            cv_row = next(r for r in cv if r["model"] == key)
            comparison = next(r for r in summary if r["model"] == key)
            np.testing.assert_allclose(float(cv_row["mean_cv_auroc"]), np.mean(fold_scores), atol=1e-14, rtol=0)
            np.testing.assert_allclose(float(cv_row["std_cv_auroc"]), np.std(fold_scores, ddof=1), atol=1e-14, rtol=0)
            for fold, value in enumerate(fold_scores):
                np.testing.assert_allclose(float(cv_row[f"fold_{fold}_auroc"]), value, atol=1e-14, rtol=0)
            np.testing.assert_allclose(float(comparison["cv_mean_auroc"]), np.mean(fold_scores), atol=1e-14, rtol=0)

            folder = base / f"{prefix}_{key}"
            completed = read_json(folder / "completed.json")
            for name, expected_hash in completed["artifact_sha256"].items():
                require(sha256(folder / name) == expected_hash, f"Changed completed artifact: {key}/{name}")
            require(read_json(folder / "config.json")["plan"] == plan, f"Different model plan: {key}")
            count = check(folder / "validation_predictions.csv", "validation")
            rows = read_csv(folder / "validation_predictions.csv")
            require([r["id"] for r in rows] == validation_ids, f"Prediction order mismatch: {key}")
            p = np.asarray([float(r["p_ai"]) for r in rows])
            probabilities[key] = p
            model = joblib.load(folder / "model.joblib")
            require(isinstance(model, CalibratedClassifierCV), f"Missing calibration: {key}")
            require(model.method == "sigmoid" and model.ensemble is False, f"Changed calibration settings: {key}")
            require(model.cv.n_splits == 3 and model.cv.random_state == plan["seeds"]["calibration"], f"Changed calibration folds: {key}")
            require(isinstance(model.estimator, Pipeline) and "features" in model.estimator.named_steps, f"Features outside calibration boundary: {key}")
            require(model.estimator.named_steps["classifier"].C == 1, f"Changed SVM C: {key}")
            classifier = model.estimator.named_steps["classifier"]
            for parameter, expected_value in plan["classifier"].items():
                require(getattr(classifier, parameter) == expected_value, f"Changed classifier parameter {parameter}: {key}")
            require(classifier.random_state == plan["seeds"]["model"], f"Changed model seed: {key}")
            require(set(model.classes_) == {0, 1}, f"Incorrect class direction: {key}")
            restored = model.predict_proba(val_text)[:, list(model.classes_).index(1)]
            np.testing.assert_allclose(restored, p, atol=1e-14, rtol=0)
            auc = float(roc_auc_score(vy, p))
            metrics = read_json(folder / "metrics.json")
            np.testing.assert_allclose(auc, metrics["validation"]["auroc"], atol=1e-14, rtol=0)
            np.testing.assert_allclose(auc, float(comparison["validation_auroc"]), atol=1e-14, rtol=0)
            require(metrics["validation"]["n"] == len(validation), f"Wrong evaluation size: {key}")
            require(metrics["train"]["n"] == len(train), f"Wrong training size: {key}")

            fitted = model.calibrated_classifiers_[0].estimator
            features = fitted.named_steps["features"]
            if key == "wc120":
                require(isinstance(features, Pipeline), "Missing combined-feature normalization pipeline")
                require(features.named_steps["normalize"].norm == "l2", "Changed combined-feature norm")
                require(features.named_steps["union"].transformer_weights is None, "Changed equal feature-block weights")
            sample = features.transform(val_text[:32])
            norms = np.sqrt(np.asarray(sample.multiply(sample).sum(axis=1)).ravel())
            nonempty = norms > 0
            np.testing.assert_allclose(norms[nonempty], np.ones(nonempty.sum()), atol=1e-12, rtol=0)
            require(sample.shape[1] == int(comparison["n_features"]), f"Feature count mismatch: {key}")
            leaves = vectorizers(features)
            expected = plan["models"][key]
            actual_counts = {}
            for leaf in leaves:
                require(leaf.min_df == 2 and leaf.lowercase and leaf.norm == "l2" and leaf.sublinear_tf, f"Changed TF-IDF settings: {key}")
                require(leaf.use_idf and leaf.smooth_idf, f"Changed IDF settings: {key}")
                kind = "word" if leaf.analyzer == "word" else "char"
                require(leaf.analyzer == kind, f"Unexpected analyzer: {key}")
                require(leaf.ngram_range == ((1, 2) if kind == "word" else (3, 5)), f"Changed n-grams: {key}")
                require(leaf.max_features == expected[f"{kind}_max_features"], f"Changed feature budget: {key}")
                actual_counts[kind] = len(leaf.vocabulary_)
            require(sum(actual_counts.values()) == sample.shape[1], f"Incorrect combined feature dimensions: {key}")
            fit_record = folder / "model_fit.json"
            if fit_record.exists():
                require(read_json(fit_record)["model_sha256"] == sha256(folder / "model.joblib"), f"Saved model checksum mismatch: {key}")
            audit["models"][key] = {
                "n_predictions": count,
                "validation_auroc": auc,
                "saved_model_prediction_max_error": float(np.max(np.abs(restored - p))),
                "n_features": sample.shape[1],
                "feature_block_dimensions": actual_counts,
                "unit_norm_nonempty_rows_checked": int(nonempty.sum()),
                "cv_mean_reconstructed": float(np.mean(fold_scores)),
                "complete_pipeline_inside_calibration": True,
                "prediction_sha256": sha256(folder / "validation_predictions.csv"),
            }
            print(f"Verified {key}: AUROC={auc:.6f}, features={sample.shape[1]}", flush=True)

    baseline_gate = read_json(study / "baseline_gate.json")
    reference_path = Path(baseline_gate["reference_predictions"])
    if not reference_path.is_absolute():
        reference_path = ROOT / reference_path
    require(sha256(reference_path) == baseline_gate["reference_sha256"], "Changed Q1 reference predictions")
    reference_rows = read_csv(reference_path)
    require([r["id"] for r in reference_rows] == validation_ids, "Reference baseline ID order differs")
    baseline_p = np.asarray([float(r["p_ai"]) for r in reference_rows])
    np.testing.assert_allclose(probabilities["w60"], baseline_p, atol=1e-10, rtol=0)
    require(baseline_gate["passed"], "Baseline reproduction gate did not pass")
    audit["baseline_reproduction_max_probability_error"] = float(np.max(np.abs(probabilities["w60"] - baseline_p)))
    np.testing.assert_allclose(audit["baseline_reproduction_max_probability_error"], baseline_gate["max_probability_error"], atol=1e-14, rtol=0)

    bootstrap = read_json(study / "bootstrap.json")
    require(bootstrap["replicates"] == plan["evaluation"]["bootstrap_replicates"], "Changed bootstrap replicate count")
    require(bootstrap["seed"] == plan["seeds"]["bootstrap"], "Changed bootstrap seed")
    require(bootstrap["primary_comparison"] == plan["evaluation"]["primary_comparison"], "Changed primary comparison")
    primary_delta = float(roc_auc_score(vy, probabilities["wc120"]) - roc_auc_score(vy, probabilities["w60"]))
    np.testing.assert_allclose(primary_delta, bootstrap["primary_delta"], atol=1e-14, rtol=0)
    rng = np.random.default_rng(plan["seeds"]["bootstrap"])
    groups = [np.flatnonzero(vy == label) for label in (0, 1)]
    deltas = []
    for _ in range(bootstrap.get("replicates", 1000)):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        deltas.append(float(roc_auc_score(vy[indices], probabilities["wc120"][indices]) -
                            roc_auc_score(vy[indices], probabilities["w60"][indices])))
    interval = np.quantile(deltas, [0.025, 0.975])
    np.testing.assert_allclose(interval, bootstrap["primary_delta_95_interval"], atol=1e-14, rtol=0)
    audit["primary_delta"] = primary_delta
    audit["primary_bootstrap_interval_reconstructed"] = interval.tolist()
    diagnostics = read_json(study / "error_change_summary.json")
    threshold = plan["evaluation"]["error_threshold"]
    require(diagnostics["threshold"] == threshold, "Changed error threshold")
    old_correct = (probabilities["w60"] >= threshold) == vy
    new_correct = (probabilities["wc120"] >= threshold) == vy
    expected_counts = {
        "w60_errors": int((~old_correct).sum()),
        "wc120_errors": int((~new_correct).sum()),
        "corrected": int((~old_correct & new_correct).sum()),
        "new_errors": int((old_correct & ~new_correct).sum()),
    }
    require(all(diagnostics[name] == value for name, value in expected_counts.items()), "Error-change diagnostic counts differ")
    changed_errors = read_csv(study / "changed_errors.csv")
    expected_changed_ids = {validation_ids[i] for i in np.flatnonzero(old_correct != new_correct)}
    require({row["id"] for row in changed_errors} == expected_changed_ids, "Incorrect changed-error IDs")
    require(len(changed_errors) == len(expected_changed_ids), "Duplicate changed-error IDs")
    train_lengths = np.asarray([len(row["text"].split()) for row in train])
    np.testing.assert_allclose(diagnostics["length_cutpoints_from_training_only"], np.quantile(train_lengths, [1/3, 2/3]), atol=1e-14, rtol=0)
    audit["error_change_counts_reconstructed"] = expected_counts
    audit["uncertainty_limitation"] = "Conditional on fitted models and this development split; no training variation or model-selection correction."
    audit["all_checks_passed"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Saved verification: {output}", flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True, help="NEW verification JSON; never overwrite")
    args = parser.parse_args()
    verify(args.prefix, args.output)


if __name__ == "__main__":
    main()
