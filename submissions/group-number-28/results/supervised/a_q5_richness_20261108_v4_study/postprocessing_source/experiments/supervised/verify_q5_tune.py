"""Audit the recorded six-configuration Q5 tuning study without full refits."""
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
from human_ai.data import ROOT, SPLIT_DIR, load_partitions, raw_path, read_csv, sha256
from .q5_sensitivity import apply_condition, CONDITIONS
from .run_q1 import id_hash, predict_ai
from .run_q5 import atomic_json


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vectorizers(transformer):
    if isinstance(transformer, TfidfVectorizer):
        return [transformer]
    children = getattr(transformer, "steps", None)
    if children is None:
        children = getattr(transformer, "transformer_list", [])
    return [leaf for _, child in children for leaf in vectorizers(child)]


def verify_receipt(folder):
    receipt = read_json(folder / "completed.json")
    for name, expected in receipt["artifact_sha256"].items():
        path = ROOT / name if str(name).startswith(("results/", "submissions/")) else folder / name
        require(sha256(path) == expected, f"Changed completed artifact: {name}")


def verify_model_structure(model, representation, selected_c, q5_plan, sample_text):
    require(isinstance(model, CalibratedClassifierCV), "Missing probability calibration")
    require(model.method == "sigmoid" and model.ensemble is False, "Changed calibration method")
    require(model.cv.n_splits == 3 and model.cv.shuffle, "Changed calibration fold structure")
    require(model.cv.random_state == q5_plan["seeds"]["calibration"], "Changed calibration seed")
    require(isinstance(model.estimator, Pipeline), "Classifier is not a complete feature pipeline")
    require(set(model.estimator.named_steps) == {"features", "classifier"}, "Incomplete calibration boundary")
    classifier = model.estimator.named_steps["classifier"]
    for parameter, value in q5_plan["classifier"].items():
        require(getattr(classifier, parameter) == (selected_c if parameter == "C" else value),
                f"Changed classifier parameter: {parameter}")
    require(classifier.random_state == q5_plan["seeds"]["model"], "Changed model seed")
    require(set(model.classes_) == {0, 1}, "Incorrect probability class direction")
    fitted = model.calibrated_classifiers_[0].estimator
    features = fitted.named_steps["features"]
    if representation == "wc120":
        require(isinstance(features, Pipeline), "Missing combination pipeline")
        require(features.named_steps["normalize"].norm == "l2", "Changed combined-feature norm")
        require(features.named_steps["union"].transformer_weights is None, "Changed feature-block weights")
    leaves = vectorizers(features)
    kinds = [leaf.analyzer for leaf in leaves]
    require(kinds == (["char"] if representation == "c60" else ["word", "char"]),
            "Unexpected feature representation")
    counts = {}
    for leaf in leaves:
        original = q5_plan[f"{leaf.analyzer}_representation"]
        for parameter in ("min_df", "max_df", "max_features", "lowercase", "strip_accents",
                          "sublinear_tf", "smooth_idf", "use_idf", "norm"):
            require(getattr(leaf, parameter) == original[parameter], f"Changed TF-IDF parameter: {parameter}")
        require(leaf.ngram_range == tuple(original["ngram_range"]), "Changed n-gram range")
        counts[leaf.analyzer] = len(leaf.vocabulary_)
    matrix = features.transform(sample_text)
    require(matrix.shape[1] == sum(counts.values()), "Incorrect feature dimensions")
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    nonempty = norms > 0
    np.testing.assert_allclose(norms[nonempty], 1, atol=1e-12, rtol=0)
    return {"feature_block_dimensions": counts, "n_features": matrix.shape[1],
            "unit_norm_nonempty_rows_checked": int(nonempty.sum()),
            "complete_pipeline_inside_calibration": True, "C": selected_c}


def verify(prefix, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    base = ROOT / "results/supervised"
    study = base / f"{prefix}_study"
    plan = read_json(study / "q5_tuning_plan.json")
    feature_plan = read_json(study / "q5_feature_plan.json")
    context = read_json(study / "provenance.json")
    require(read_json(study / "status.json")["stage"] == "complete", "Tuning study is incomplete")
    require(plan["representations"] == ["c60", "wc120"] and plan["C_candidates"] == [0.1, 1, 10],
            "Unexpected search grid")
    for relative, expected in context["source_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"Changed execution source: {relative}")
        require(sha256(study / "source" / relative) == expected, f"Changed source snapshot: {relative}")
    require(plan == read_json(study / "source/experiments/supervised/q5_tuning_plan.json"),
            "Frozen tuning plan differs from executed source")
    require(feature_plan == read_json(ROOT / plan["feature_plan"]),
            "Frozen feature plan differs from the original representation")
    for relative, expected in read_json(study / "source_guards.json").items():
        require(sha256(ROOT / relative) == expected, f"Changed original evidence: {relative}")

    train, validation = load_partitions()
    train_ids = [r["id"] for r in train]
    validation_ids = [r["id"] for r in validation]
    y = np.asarray([int(r["label"]) for r in train])
    vy = np.asarray([int(r["label"]) for r in validation])
    val_text = [r["text"] for r in validation]
    require(read_json(study / "split_manifest.json") == read_json(SPLIT_DIR / "manifest.json"),
            "Changed split manifest")
    for name, ids in (("train", train_ids), ("validation", validation_ids)):
        require([r["id"] for r in read_csv(study / f"{name}_ids.csv")] == ids, f"Noncanonical {name} IDs")
    folds = list(StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["inner_cv"]).split(train_ids, y))
    assignments = read_csv(study / "inner_folds.csv")
    require([r["id"] for r in assignments] == train_ids, "Noncanonical fold IDs")
    for fold, (_, held) in enumerate(folds):
        require(all(int(assignments[i]["inner_validation_fold"]) == fold for i in held), "Changed fold assignment")

    progress = read_json(study / "cv_progress.json")
    cv = read_csv(study / "cv_results.csv")
    require(len(progress) == 18 and len(cv) == 6, "Incomplete six-configuration search")
    expected_ids = {f"{rep}_c{format(float(c), 'g').replace('.', 'p')}"
                    for rep in plan["representations"] for c in plan["C_candidates"]}
    require({r["candidate_id"] for r in cv} == expected_ids, "Unexpected candidate IDs")
    numeric_cv = []
    for row in cv:
        cid, rep, c = row["candidate_id"], row["representation"], float(row["C"])
        require(cid == f"{rep}_c{format(c, 'g').replace('.', 'p')}", "Candidate name/settings mismatch")
        records = [r for r in progress if r["candidate_id"] == cid]
        require(sorted(r["fold"] for r in records) == [0, 1, 2], f"Wrong folds: {cid}")
        values = []
        for fold, (fit, held) in enumerate(folds):
            record = next(r for r in records if r["fold"] == fold)
            require(record["representation"] == rep and record["C"] == c, f"Inconsistent candidate: {cid}")
            require(record["fit_ids_sha256"] == id_hash([train_ids[i] for i in fit]), f"Wrong fitting rows: {cid}")
            require(record["held_ids_sha256"] == id_hash([train_ids[i] for i in held]), f"Wrong held rows: {cid}")
            require(np.isfinite(record["auroc"]) and 0 <= record["auroc"] <= 1, f"Invalid CV AUROC: {cid}")
            values.append(record["auroc"])
            np.testing.assert_allclose(float(row[f"fold_{fold}_auroc"]), record["auroc"], atol=1e-14, rtol=0)
        np.testing.assert_allclose(float(row["mean_cv_auroc"]), np.mean(values), atol=1e-14, rtol=0)
        np.testing.assert_allclose(float(row["std_cv_auroc"]), np.std(values, ddof=1), atol=1e-14, rtol=0)
        numeric_cv.append({**row, "C": c, "mean_cv_auroc": float(row["mean_cv_auroc"]),
                           "std_cv_auroc": float(row["std_cv_auroc"])})
    audits = read_json(study / "feature_fit_audit.json")
    require(len(audits) == 6 and {(r["representation"], r["fold"]) for r in audits} ==
            {(rep, fold) for rep in plan["representations"] for fold in range(3)}, "Wrong feature-fitting boundaries")
    for audit in audits:
        fit, held = folds[audit["fold"]]
        require(audit["fit_ids_sha256"] == id_hash([train_ids[i] for i in fit]), "Feature fit ID mismatch")
        require(audit["held_ids_sha256"] == id_hash([train_ids[i] for i in held]), "Feature held ID mismatch")
    preference = {1.0: 2, 0.1: 1, 10.0: 0}
    expected_finalists = {rep: max((r for r in numeric_cv if r["representation"] == rep),
                          key=lambda r: (r["mean_cv_auroc"], -r["std_cv_auroc"], preference[r["C"]]))
                          for rep in plan["representations"]}
    finalists = read_json(study / "finalists.json")
    chosen = read_json(study / "chosen_parameters.json")
    for rep, expected in expected_finalists.items():
        require(finalists[rep]["candidate_id"] == expected["candidate_id"] and chosen[rep] == expected["C"],
                f"Finalist not selected by training-only CV: {rep}")

    baseline_key = "baseline_wc120"
    baseline_folder = base / plan["baseline_run_id"]
    verify_receipt(baseline_folder)
    reference_rows = read_csv(baseline_folder / "validation_predictions.csv")
    require([(r["id"], r["label"]) for r in reference_rows] ==
            [(r["id"], r["label"]) for r in validation], "Baseline reference IDs/labels differ")
    baseline_p = np.asarray([float(r["p_ai"]) for r in reference_rows])
    require(sha256(study / "baseline_validation_predictions.csv") == sha256(baseline_folder / "validation_predictions.csv"),
            "Baseline prediction copy differs")
    original_cv = read_json(base / plan["reference_study"] / "cv_progress.json")
    cv_gate = read_json(study / "baseline_cv_gate.json")
    for rep in plan["representations"]:
        row = next(r for r in numeric_cv if r["representation"] == rep and r["C"] == 1)
        errors = [abs(float(row[f"fold_{fold}_auroc"]) -
                      next(r["auroc"] for r in original_cv if r["model"] == rep and r["fold"] == fold))
                  for fold in range(3)]
        require(max(errors) <= 1e-12, f"Original C=1 CV no longer reproduces: {rep}")
        np.testing.assert_allclose(cv_gate[rep]["max_cv_auroc_error"], max(errors), atol=1e-14, rtol=0)

    comparison = read_csv(study / "comparison.csv")
    expected_models = {baseline_key} | {r["candidate_id"] for r in expected_finalists.values()}
    require(len(comparison) == 3 and {r["candidate_id"] for r in comparison} == expected_models,
            "Expected the baseline and exactly two finalists")
    probabilities = {baseline_key: baseline_p}
    models = {baseline_key: joblib.load(baseline_folder / "model.joblib")}
    audit = {"source_hashes_match": True, "original_evidence_unchanged": True,
             "canonical_split_and_fold_ids_match": True, "n_cv_fits": 18,
             "n_feature_fit_boundaries": 6, "finalists_selected_from_training_cv": chosen,
             "n_validation": len(validation), "full_model_refits_performed_by_verifier": False,
             "models": {}}
    with threadpool_limits(limits=1):
        np.testing.assert_array_equal(predict_ai(models[baseline_key], val_text), baseline_p)
        for rep, expected in expected_finalists.items():
            cid, c = expected["candidate_id"], expected["C"]
            folder = base / f"{prefix}_{cid}"
            verify_receipt(folder)
            config = read_json(folder / "config.json")
            require(config["representation"] == rep and config["C"] == c and config["feature_plan"] == feature_plan,
                    f"Changed model configuration: {cid}")
            require(config["n_train"] == len(train), "Finalist used noncanonical training size")
            require([r["id"] for r in read_csv(folder / "train_ids.csv")] == train_ids, "Finalist training order changed")
            count = check(folder / "validation_predictions.csv", "validation")
            rows = read_csv(folder / "validation_predictions.csv")
            require([r["id"] for r in rows] == validation_ids, "Finalist prediction order changed")
            p = np.asarray([float(r["p_ai"]) for r in rows])
            model = joblib.load(folder / "model.joblib")
            values = verify_model_structure(model, rep, c, feature_plan, val_text[:32])
            restored = predict_ai(model, val_text)
            np.testing.assert_array_equal(restored, p)
            metrics = read_json(folder / "metrics.json")
            auc = float(roc_auc_score(vy, p))
            row = next(r for r in comparison if r["candidate_id"] == cid)
            for observed in (metrics["validation"]["auroc"], float(row["validation_auroc"])):
                np.testing.assert_allclose(auc, observed, atol=1e-14, rtol=0)
            require(metrics["validation"]["n"] == len(validation) and metrics["train"]["n"] == len(train),
                    "Incorrect finalist evaluation size")
            require(metrics["n_features"] == values["n_features"], "Incorrect finalist feature count")
            np.testing.assert_allclose(float(row["cv_mean_auroc"]), expected["mean_cv_auroc"], atol=1e-14, rtol=0)
            np.testing.assert_allclose(float(row["cv_std_auroc"]), expected["std_cv_auroc"], atol=1e-14, rtol=0)
            require(read_json(folder / "model_fit.json")["model_sha256"] == sha256(folder / "model.joblib"),
                    "Changed model checkpoint")
            audit["models"][cid] = {**values, "n_predictions": count, "validation_auroc": auc,
                                    "saved_model_prediction_max_error": float(np.max(abs(restored - p)))}
            models[cid], probabilities[cid] = model, p
            print(f"Verified tuned finalist {cid}: AUROC={auc:.6f}", flush=True)
        baseline_row = next(r for r in comparison if r["candidate_id"] == baseline_key)
        np.testing.assert_allclose(float(baseline_row["validation_auroc"]), roc_auc_score(vy, baseline_p), atol=1e-14, rtol=0)

        uncertainty = read_json(study / "bootstrap.json")
        require(uncertainty["seed"] == plan["seeds"]["bootstrap"] and
                uncertainty["replicates"] == plan["evaluation"]["bootstrap_replicates"], "Changed bootstrap protocol")
        rng = np.random.default_rng(plan["seeds"]["bootstrap"])
        groups = [np.flatnonzero(vy == value) for value in (0, 1)]
        sampled = {key: [] for key in probabilities}
        for _ in range(uncertainty["replicates"]):
            indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
            for key, p in probabilities.items():
                sampled[key].append(float(roc_auc_score(vy[indices], p[indices])))
        for key, p in probabilities.items():
            interval = np.quantile(sampled[key], [.025, .975])
            delta_interval = np.quantile(np.asarray(sampled[key]) - sampled[baseline_key], [.025, .975])
            delta = float(roc_auc_score(vy, p) - roc_auc_score(vy, baseline_p))
            np.testing.assert_allclose(interval, uncertainty["auroc_95_intervals"][key], atol=1e-14, rtol=0)
            np.testing.assert_allclose(delta_interval, uncertainty["delta_vs_baseline_95_intervals"][key], atol=1e-14, rtol=0)
            np.testing.assert_allclose(delta, uncertainty["point_deltas_vs_baseline"][key], atol=1e-14, rtol=0)
        audit["all_bootstrap_intervals_reconstructed"] = True

        sensitivity = read_json(study / "format_sensitivity.json")
        sensitivity_csv = read_csv(study / "format_sensitivity.csv")
        require(sensitivity["conditions"] == list(CONDITIONS) and len(sensitivity_csv) == 9, "Unexpected format conditions")
        require(sensitivity["paired_by_original_id"] and not sensitivity["new_model_fits_performed"],
                "Format diagnostics should use paired IDs and no new model fits")
        for relative, expected in sensitivity["prediction_sha256"].items():
            require(sha256(study / relative) == expected, f"Changed sensitivity predictions: {relative}")
        threshold = plan["evaluation"]["error_threshold"]
        for key, model in models.items():
            original = probabilities[key]
            for condition in CONDITIONS:
                records = read_csv(study / "format_predictions" / f"{key}_{condition}.csv")
                require([(r["id"], r["label"]) for r in records] ==
                        [(r["id"], r["label"]) for r in validation], "Changed sensitivity IDs/labels")
                transformed = [apply_condition(text, condition) for text in val_text]
                p = np.asarray([float(r["p_ai"]) for r in records])
                expected_p = original if condition == "original" else predict_ai(model, transformed)
                np.testing.assert_array_equal(p, expected_p)
                changed = np.asarray([before != after for before, after in zip(val_text, transformed, strict=True)])
                row = sensitivity["models"][key][condition]
                csv_row = next(r for r in sensitivity_csv if r["model"] == key and r["condition"] == condition)
                require(row["n"] == len(validation), "Sensitivity sample size differs")
                require(row["n_changed"] == int(changed.sum()) == int(csv_row["n_changed"]), "Changed document count differs")
                np.testing.assert_allclose(row["changed_fraction"], np.mean(changed), atol=1e-14, rtol=0)
                require(row["n_changed_human"] == int(np.sum(changed & (vy == 0))) and
                        row["n_changed_ai"] == int(np.sum(changed & (vy == 1))), "Changed subgroup counts differ")
                auc = float(roc_auc_score(vy, p))
                for observed in (row["auroc"], float(csv_row["auroc"])):
                    np.testing.assert_allclose(auc, observed, atol=1e-14, rtol=0)
                np.testing.assert_allclose(row["delta_auroc"], auc - roc_auc_score(vy, original), atol=1e-14, rtol=0)
                np.testing.assert_allclose(row["mean_abs_probability_change"], np.mean(abs(p - original)), atol=1e-14, rtol=0)
                np.testing.assert_allclose(row["flip_rate_at_0_5"], np.mean((p >= threshold) != (original >= threshold)), atol=1e-14, rtol=0)
                if changed.any() and len(set(vy[changed])) == 2:
                    subset_auc = roc_auc_score(vy[changed], p[changed])
                    subset_delta = subset_auc - roc_auc_score(vy[changed], original[changed])
                    np.testing.assert_allclose(row["changed_subset_auroc"], subset_auc, atol=1e-14, rtol=0)
                    np.testing.assert_allclose(row["changed_subset_delta_auroc"], subset_delta, atol=1e-14, rtol=0)
                else:
                    require(row["changed_subset_auroc"] is None and row["changed_subset_delta_auroc"] is None,
                            "Undefined changed-subset AUROC must be null")
                for field in ("auroc", "delta_auroc", "changed_fraction", "mean_abs_probability_change",
                              "flip_rate_at_0_5", "changed_subset_auroc", "changed_subset_delta_auroc"):
                    if row[field] is None:
                        require(csv_row[field] == "", "Undefined CSV metric must be empty")
                    else:
                        np.testing.assert_allclose(float(csv_row[field]), row[field], atol=1e-14, rtol=0)
        audit["all_sensitivity_predictions_and_metrics_reconstructed"] = True

    cuts = np.quantile([len(r["text"].split()) for r in train], [1/3, 2/3])
    boundaries = read_json(study / "length_boundaries.json")
    np.testing.assert_allclose(boundaries["training_word_count_tertiles"], cuts, atol=1e-14, rtol=0)
    bins = np.digitize([len(r["text"].split()) for r in validation], cuts, right=True)
    length_rows = read_csv(study / "length_subgroups.csv")
    require(len(length_rows) == 9, "Unexpected length subgroup rows")
    for row in length_rows:
        key, b = row["candidate_id"], int(row["length_bin"])
        require(key in probabilities and b in (0, 1, 2), "Invalid length subgroup")
        mask = bins == b
        require(int(row["n"]) == int(mask.sum()) and int(row["n_human"]) == int(np.sum(vy[mask] == 0)) and
                int(row["n_ai"]) == int(np.sum(vy[mask] == 1)), "Incorrect subgroup class counts")
        np.testing.assert_allclose(float(row["validation_auroc"]), roc_auc_score(vy[mask], probabilities[key][mask]), atol=1e-14, rtol=0)

    selection = read_json(study / "selection.json")
    baseline_row = next(r for r in comparison if r["candidate_id"] == baseline_key)
    base_lengths = {int(r["length_bin"]): float(r["validation_auroc"]) for r in length_rows if r["candidate_id"] == baseline_key}
    eligible = []
    for row in comparison:
        key = row["candidate_id"]
        if key == baseline_key:
            continue
        maximum_length_regression = max(base_lengths[int(r["length_bin"])] - float(r["validation_auroc"])
                                        for r in length_rows if r["candidate_id"] == key)
        maximum_extra_format_loss = max(sensitivity["models"][baseline_key][condition]["delta_auroc"] -
                                        sensitivity["models"][key][condition]["delta_auroc"] for condition in CONDITIONS[1:])
        checks = {
            "different_from_current_settings": not (row["representation"] == "wc120" and float(row["C"]) == 1),
            "cv_mean_improves": float(row["cv_mean_auroc"]) > float(baseline_row["cv_mean_auroc"]),
            "paired_interval_lower_positive": uncertainty["delta_vs_baseline_95_intervals"][key][0] > 0,
            "length_guard_passes": maximum_length_regression <= plan["export_rule"]["maximum_length_bin_auroc_regression"],
            "format_guard_passes": maximum_extra_format_loss <= plan["export_rule"]["maximum_extra_format_auroc_loss_vs_baseline"],
        }
        actual = selection["eligibility"][key]
        require(actual["checks"] == checks and actual["eligible"] == all(checks.values()), "Incorrect export eligibility")
        np.testing.assert_allclose(actual["maximum_length_bin_regression"], maximum_length_regression, atol=1e-14, rtol=0)
        np.testing.assert_allclose(actual["maximum_extra_format_loss"], maximum_extra_format_loss, atol=1e-14, rtol=0)
        if all(checks.values()):
            eligible.append(row)
    finalists_rows = [r for r in comparison if r["candidate_id"] != baseline_key]
    selected = max(eligible or finalists_rows, key=lambda r: (float(r["validation_auroc"]), float(r["cv_mean_auroc"]),
                   -float(r["cv_std_auroc"]), r["representation"] == "c60"))
    require(selection["selected_candidate_id"] == selected["candidate_id"] and selection["export_eligible"] == bool(eligible),
            "Incorrect between-finalist decision")
    require(selection["selected_representation"] == selected["representation"] and
            float(selection["selected_C"]) == float(selected["C"]), "Selected settings differ from the winner")
    require(selection["retain_current_public_baseline"] == (not bool(eligible)), "Incorrect baseline retention flag")
    audit["export_eligibility_and_selection_reconstructed"] = True
    audit["export_eligible"] = bool(eligible)

    if eligible:
        csv_path = ROOT / selection["submission_csv"]
        meta = read_json(csv_path.with_suffix(".metadata.json"))
        require(meta["representation"] == selected["representation"] and float(meta["C"]) == float(selected["C"]),
                "Export metadata identifies different selected settings")
        folder = base / meta["model_run_id"]
        verify_receipt(folder)
        require(check(csv_path, "submission") == 3000, "Incorrect submission row count")
        require(meta["n_train"] == 10000 and meta["n_test"] == 3000 and meta["full_refit"], "Incorrect export sizes")
        require(meta["full_refit_held_out_evaluation"] is None and not meta["uploaded_to_kaggle"], "Invalid full-refit evaluation/upload claim")
        require(read_json(folder / "metrics.json")["validation"] is None, "Full-label model has a held-out score")
        require([r["id"] for r in read_csv(folder / "train_ids.csv")] == train_ids + validation_ids, "Incorrect full-label training IDs")
        require(meta["csv_sha256"] == sha256(csv_path) and meta["test_sha256"] == sha256(raw_path("test.csv")), "Export checksum mismatch")
        test = read_csv(raw_path("test.csv"))
        exported = read_csv(csv_path)
        require([r["id"] for r in exported] == [r["id"] for r in test], "Submission test-ID order changed")
        with threadpool_limits(limits=1):
            model = joblib.load(folder / "model.joblib")
            verify_model_structure(model, selected["representation"], float(selected["C"]), feature_plan, [r["text"] for r in test[:32]])
            restored = predict_ai(model, [r["text"] for r in test])
        np.testing.assert_array_equal(restored, np.asarray([float(r["label"]) for r in exported]))
        audit["full_label_export"] = {"csv": str(csv_path.relative_to(ROOT)), "n_test": 3000,
                                      "saved_model_prediction_max_error": 0.0, "uploaded_to_kaggle": False}
    else:
        require("submission_csv" not in selection, "An ineligible candidate was exported")
    exported_paths = list((ROOT / "submissions").glob(f"{prefix}_*_full10000.csv"))
    require(len(exported_paths) == int(bool(eligible)), "The study exported an unexpected number of candidates")
    audit["verifier_source_sha256"] = sha256(Path(__file__))
    audit["uncertainty_limitations"] = "Conditional development intervals omit training variation and selection adjustment; no hidden-test gain is established."
    audit["all_checks_passed"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, audit)
    print(f"Saved tuned-Q5 verification: {output}", flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.prefix, args.output)


if __name__ == "__main__":
    main()
