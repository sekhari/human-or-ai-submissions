"""Audit the quote-canonicalization ablation without fresh full-size refits."""
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
from .verify_q5_tune import read_json, require, vectorizers, verify_receipt


def verify_structure(model, representation, normalized, feature_plan, sample_text):
    require(isinstance(model, CalibratedClassifierCV), "Missing calibrated classifier")
    require(model.method == "sigmoid" and model.ensemble is False, "Changed calibration method")
    require(model.cv.n_splits == 3 and model.cv.shuffle, "Changed calibration folds")
    require(model.cv.random_state == feature_plan["seeds"]["calibration"], "Changed calibration seed")
    expected_steps = ["quote_normalize", "features", "classifier"] if normalized else ["features", "classifier"]
    require(isinstance(model.estimator, Pipeline), "Features outside calibration pipeline")
    require(list(model.estimator.named_steps) == expected_steps, "Incorrect normalization/calibration boundary")
    fitted = model.calibrated_classifiers_[0].estimator
    require(list(fitted.named_steps) == expected_steps, "Fitted pipeline differs from declaration")
    classifier = model.estimator.named_steps["classifier"]
    for parameter, expected in feature_plan["classifier"].items():
        require(getattr(classifier, parameter) == expected, f"Changed classifier parameter: {parameter}")
    require(classifier.C == 1 and classifier.random_state == feature_plan["seeds"]["model"], "Changed SVM settings")
    require(set(model.classes_) == {0, 1}, "Incorrect probability class direction")
    features = fitted.named_steps["features"]
    if normalized:
        transformer = fitted.named_steps["quote_normalize"]
        transformed = list(transformer.transform(sample_text))
        expected = [apply_condition(text, "quote_typography") for text in sample_text]
        require(transformed == expected, "Quote normalizer differs from frozen mapping")
        require(list(transformer.transform(transformed)) == transformed, "Quote normalization is not idempotent")
        feature_text = transformed
    else:
        feature_text = sample_text
    if representation == "wc120":
        require(isinstance(features, Pipeline), "Missing combined-feature pipeline")
        require(features.named_steps["normalize"].norm == "l2", "Changed global normalization")
        require(features.named_steps["union"].transformer_weights is None, "Changed equal block weights")
    leaves = vectorizers(features)
    require([leaf.analyzer for leaf in leaves] == (["char"] if representation == "c60" else ["word", "char"]),
            "Changed representation")
    counts = {}
    for leaf in leaves:
        original = feature_plan[f"{leaf.analyzer}_representation"]
        for parameter in ("min_df", "max_df", "max_features", "lowercase", "strip_accents", "sublinear_tf",
                          "smooth_idf", "use_idf", "norm"):
            require(getattr(leaf, parameter) == original[parameter], f"Changed TF-IDF setting: {parameter}")
        require(leaf.ngram_range == tuple(original["ngram_range"]), "Changed n-gram range")
        counts[leaf.analyzer] = len(leaf.vocabulary_)
    matrix = features.transform(feature_text)
    require(matrix.shape[1] == sum(counts.values()), "Feature dimensions differ")
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    nonempty = norms > 0
    np.testing.assert_allclose(norms[nonempty], 1, atol=1e-12, rtol=0)
    canonical_counts = {"total": int(matrix.shape[1]), "word": counts.get("word", 0), "char": counts.get("char", 0)}
    return {"feature_counts": canonical_counts, "n_features": int(matrix.shape[1]), "C": 1,
            "complete_pipeline_inside_calibration": True, "normalization_inside_calibration": bool(normalized)}


def verify_cv(study, plan, train):
    ids = [row["id"] for row in train]
    y = np.asarray([int(row["label"]) for row in train])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["inner_cv"]).split(ids, y))
    assignments = read_csv(study / "inner_folds.csv")
    require([r["id"] for r in assignments] == ids, "Noncanonical CV ID order")
    for fold, (_, held) in enumerate(folds):
        require(all(int(assignments[i]["inner_validation_fold"]) == fold for i in held), "Changed fold assignments")
    progress = read_json(study / "cv_progress.json")
    rows = read_csv(study / "cv_results.csv")
    expected_ids = {f"{rep}_{mode}" for rep in plan["representations"] for mode in ("raw", "norm")}
    require(len(progress) == 12 and len(rows) == 4, "Incomplete four-pipeline CV")
    require({r["model_id"] for r in rows} == expected_ids, "Unexpected model IDs")
    numeric = {}
    for row in rows:
        key = row["model_id"]
        representation = key.removesuffix("_norm").removesuffix("_raw")
        normalized = key.endswith("_norm")
        observed = [r for r in progress if r["model_id"] == key]
        require(sorted(r["fold"] for r in observed) == [0, 1, 2], f"Incorrect fold records: {key}")
        scores = {condition: [] for condition in CONDITIONS}
        for fold, (fit, held) in enumerate(folds):
            record = next(r for r in observed if r["fold"] == fold)
            require(record["representation"] == representation and record["normalize_quotes"] == normalized,
                    "CV settings/model ID mismatch")
            require(record["fit_ids_sha256"] == id_hash([ids[i] for i in fit]), "Incorrect fitting IDs")
            require(record["held_ids_sha256"] == id_hash([ids[i] for i in held]), "Incorrect held IDs")
            for condition, field, suffix in (("original", "original_auroc", ""),
                                             ("quote_typography", "quote_auroc", "_quote"),
                                             ("dash_typography", "dash_auroc", "_dash")):
                value = record[field]
                require(np.isfinite(value) and 0 <= value <= 1, "Invalid CV AUROC")
                scores[condition].append(value)
                # Dash fold scores are checkpointed in progress; only their
                # summary mean is persisted by the frozen runner.
                column = f"fold_{fold}{suffix}_auroc"
                if column in row:
                    np.testing.assert_allclose(float(row[column]), value, atol=1e-14, rtol=0)
            if normalized:
                require(record["quote_invariance_max_margin_error"] <= 1e-12,
                        "Normalized model changed under repeated quote mapping")
                np.testing.assert_allclose(record["original_auroc"], record["quote_auroc"], atol=1e-14, rtol=0)
        means = {condition: float(np.mean(values)) for condition, values in scores.items()}
        for condition, field in (("original", "mean_cv_auroc"), ("quote_typography", "mean_quote_cv_auroc"),
                                 ("dash_typography", "mean_dash_cv_auroc")):
            np.testing.assert_allclose(float(row[field]), means[condition], atol=1e-14, rtol=0)
        np.testing.assert_allclose(float(row["std_cv_auroc"]), np.std(scores["original"], ddof=1), atol=1e-14, rtol=0)
        numeric[key] = {"representation": representation, "normalize_quotes": normalized,
                        "means": means, "scores": scores}
    reference = read_json(study.parent / plan["reference_study"] / "cv_progress.json")
    for representation in plan["representations"]:
        for fold in range(3):
            original = next(r["auroc"] for r in reference if r["model"] == representation and r["fold"] == fold)
            np.testing.assert_allclose(numeric[f"{representation}_raw"]["scores"]["original"][fold], original,
                                       atol=1e-12, rtol=0)
    return numeric


def verify(prefix, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    base = ROOT / "results/supervised"
    study = base / f"{prefix}_study"
    plan = read_json(study / "q5_quote_plan.json")
    features = read_json(study / "q5_feature_plan.json")
    context = read_json(study / "provenance.json")
    require(read_json(study / "status.json")["stage"] == "complete", "Quote study is incomplete")
    require(plan["representations"] == ["c60", "wc120"] and plan["C"] == 1,
            "Changed frozen representation/classifier protocol")
    require(plan["conditions"] == list(CONDITIONS), "Changed input conditions")
    require(read_json(ROOT / "experiments/supervised/q5_quote_plan.json") == plan,
            "Current quote protocol differs from executed protocol")
    require(read_json(ROOT / plan["feature_plan"]) == features,
            "Feature plan differs from original Q5")
    for relative, expected in context["source_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"Changed executing source: {relative}")
        require(sha256(study / "source" / relative) == expected, f"Changed source snapshot: {relative}")
    for relative, expected in read_json(study / "source_guards.json").items():
        require(sha256(ROOT / relative) == expected, f"Changed prior evidence: {relative}")

    train, validation = load_partitions()
    train_ids = [row["id"] for row in train]
    validation_ids = [row["id"] for row in validation]
    labels = np.asarray([int(row["label"]) for row in validation])
    text = [row["text"] for row in validation]
    require(len(train_ids) == 8000 and len(validation_ids) == 2000, "Unexpected partition sizes")
    require(not set(train_ids) & set(validation_ids), "Canonical partitions overlap")
    require(read_json(study / "split_manifest.json") == read_json(SPLIT_DIR / "manifest.json"),
            "Changed split manifest")
    for name, expected in (("train", train_ids), ("validation", validation_ids)):
        require([row["id"] for row in read_csv(study / f"{name}_ids.csv")] == expected,
                f"Changed {name} IDs or order")
    cv = verify_cv(study, plan, train)

    comparison = read_csv(study / "comparison.csv")
    model_ids = ("c60_raw", "c60_norm", "wc120_raw", "wc120_norm")
    expected_cells = {(mid, condition) for mid in model_ids for condition in CONDITIONS}
    require(len(comparison) == 12 and {(r["model_id"], r["condition"]) for r in comparison} == expected_cells,
            "Incomplete or duplicate training/input matrix")
    probabilities, aucs = {}, {}
    audit = {"source_hashes_match": True, "prior_evidence_unchanged": True,
             "canonical_split_and_fold_ids_match": True, "n_cv_fits": 12,
             "n_matrix_conditions": 12, "n_validation": len(validation),
             "full_size_refits_performed_by_verifier": False, "models": {}}
    with threadpool_limits(limits=1):
        for mid in model_ids:
            rep = mid.removesuffix("_norm").removesuffix("_raw")
            normalized = mid.endswith("_norm")
            folder = base / (f"{prefix}_{mid}" if normalized else f"a_q5_20261108_v1_{rep}")
            verify_receipt(folder)
            model = joblib.load(folder / "model.joblib")
            structure = verify_structure(model, rep, normalized, features, text[:32])
            metrics = read_json(folder / "metrics.json")
            if normalized:
                cfg = read_json(folder / "config.json")
                require(cfg["normalize_quotes"] and cfg["C"] == 1 and cfg["n_train"] == 8000,
                        "Incorrect normalized-model training configuration")
                require(cfg["feature_plan"] == features and cfg["representation"] == rep,
                        "Incorrect normalized-model features")
                require([r["id"] for r in read_csv(folder / "train_ids.csv")] == train_ids,
                        "Normalized model used different training rows")
                require(read_json(folder / "model_fit.json")["model_sha256"] == sha256(folder / "model.joblib"),
                        "Changed model checkpoint")
            model_audit = {**structure, "saved_prediction_max_error": 0.0, "conditions": {}}
            for condition in CONDITIONS:
                row = next(r for r in comparison if r["model_id"] == mid and r["condition"] == condition)
                require(row["representation"] == rep and row["train_preprocessing"] ==
                        ("quote_normalized" if normalized else "raw"), "Matrix configuration mismatch")
                inputs = [apply_condition(document, condition) for document in text]
                predicted = predict_ai(model, inputs)
                csv_path = study / "predictions" / f"{mid}_{condition}.csv"
                require(check(csv_path, "validation") == len(validation), "Wrong prediction row count")
                records = read_csv(csv_path)
                require([(r["id"], r["label"]) for r in records] ==
                        [(r["id"], r["label"]) for r in validation], "Changed matrix IDs or labels")
                saved = np.asarray([float(r["p_ai"]) for r in records])
                np.testing.assert_array_equal(predicted, saved)
                key = f"{mid}__{condition}"
                probabilities[key] = saved
                aucs[(mid, condition)] = float(roc_auc_score(labels, saved))
                changed = sum(before != after for before, after in zip(text, inputs, strict=True))
                require(int(row["n"]) == len(validation) and int(row["n_changed"]) == changed,
                        "Incorrect modified-document count")
                np.testing.assert_allclose(float(row["validation_auroc"]), aucs[(mid, condition)], atol=1e-14, rtol=0)
                model_audit["conditions"][condition] = {"auroc": aucs[(mid, condition)],
                                                        "n_changed": changed,
                                                        "prediction_sha256": sha256(csv_path)}
            original = probabilities[f"{mid}__original"]
            source_p = np.asarray([float(r["p_ai"]) for r in read_csv(folder / "validation_predictions.csv")])
            np.testing.assert_array_equal(original, source_p)
            require(metrics["validation"]["n"] == len(validation) and metrics["train"]["n"] == len(train),
                    "Incorrect model evaluation sizes")
            np.testing.assert_allclose(metrics["validation"]["auroc"], aucs[(mid, "original")], atol=1e-14, rtol=0)
            if normalized:
                np.testing.assert_array_equal(original, probabilities[f"{mid}__quote_typography"])
                require(metrics["quote_invariance_is_by_construction"], "Missing invariance limitation")
                np.testing.assert_allclose(metrics["quote_validation_auroc"], aucs[(mid, "quote_typography")], atol=1e-14, rtol=0)
                np.testing.assert_allclose(metrics["dash_validation_auroc"], aucs[(mid, "dash_typography")], atol=1e-14, rtol=0)
                require(metrics["feature_counts"] == structure["feature_counts"], "Incorrect feature count")
            audit["models"][mid] = model_audit
            print(f"Verified quote matrix {mid}: clean={aucs[(mid, 'original')]:.6f}, quote={aucs[(mid, 'quote_typography')]:.6f}", flush=True)
            del model

    uncertainty = read_json(study / "bootstrap.json")
    require(uncertainty["seed"] == plan["seeds"]["bootstrap"] and uncertainty["replicates"] ==
            plan["evaluation"]["bootstrap_replicates"], "Changed paired bootstrap protocol")
    require(uncertainty["primary_comparison_key"] == "wc120_quote", "Changed primary comparison")
    rng = np.random.default_rng(plan["seeds"]["bootstrap"])
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    sampled = {key: [] for key in probabilities}
    for _ in range(uncertainty["replicates"]):
        ix = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for key, p in probabilities.items():
            sampled[key].append(float(roc_auc_score(labels[ix], p[ix])))
    for key, samples in sampled.items():
        np.testing.assert_allclose(np.quantile(samples, [.025, .975]), uncertainty["auroc_95_intervals"][key], atol=1e-14, rtol=0)
    expected_contrasts = {}
    for rep in plan["representations"]:
        for suffix, condition in (("clean", "original"), ("quote", "quote_typography")):
            expected_contrasts[f"{rep}_{suffix}"] = (f"{rep}_norm__{condition}", f"{rep}_raw__{condition}")
            expected_contrasts[f"{rep}_{suffix}_vs_incumbent"] = (f"{rep}_norm__{condition}", f"wc120_raw__{condition}")
    require(set(uncertainty["comparisons"]) == set(expected_contrasts), "Unexpected contrast set")
    for name, (left, right) in expected_contrasts.items():
        row = uncertainty["comparisons"][name]
        require((row["left"], row["right"]) == (left, right), "Incorrect matched contrast")
        difference = float(roc_auc_score(labels, probabilities[left]) - roc_auc_score(labels, probabilities[right]))
        np.testing.assert_allclose(row["point_delta"], difference, atol=1e-14, rtol=0)
        np.testing.assert_allclose(row["interval95"], np.quantile(np.asarray(sampled[left]) - sampled[right], [.025, .975]), atol=1e-14, rtol=0)
    audit["all_bootstrap_intervals_reconstructed"] = True

    boundaries = read_json(study / "length_boundaries.json")
    cuts = np.quantile([len(r["text"].split()) for r in train], [1/3, 2/3])
    np.testing.assert_allclose(boundaries["training_word_count_tertiles"], cuts, atol=1e-14, rtol=0)
    bins = np.digitize([len(r["text"].split()) for r in validation], cuts, right=True)
    lengths = read_csv(study / "length_subgroups.csv")
    require(len(lengths) == 12 and {(r["model_id"], int(r["length_bin"])) for r in lengths} ==
            {(mid, b) for mid in model_ids for b in range(3)}, "Incorrect length subgroup matrix")
    length_aucs = {}
    for row in lengths:
        mid, b = row["model_id"], int(row["length_bin"])
        mask = bins == b
        require(int(row["n"]) == int(mask.sum()) and int(row["n_human"]) == int(np.sum(labels[mask] == 0)) and
                int(row["n_ai"]) == int(np.sum(labels[mask] == 1)), "Incorrect length-group class counts")
        value = float(roc_auc_score(labels[mask], probabilities[f"{mid}__original"][mask]))
        np.testing.assert_allclose(float(row["validation_auroc"]), value, atol=1e-14, rtol=0)
        length_aucs[(mid, b)] = value

    selection = read_json(study / "selection.json")
    rules = plan["export_rule"]
    require(rules["candidate_priority"] == ["wc120_norm", "c60_norm"] and rules["maximum_new_candidates"] == 1,
            "Changed candidate priority or count")
    decisions = {}
    base_dash_delta = aucs[("wc120_raw", "dash_typography")] - aucs[("wc120_raw", "original")]
    for rep in plan["representations"]:
        mid = f"{rep}_norm"
        gains = [a - b for a, b in zip(cv[mid]["scores"]["quote_typography"], cv[f"{rep}_raw"]["scores"]["quote_typography"], strict=True)]
        clean_delta = aucs[(mid, "original")] - aucs[("wc120_raw", "original")]
        cv_delta = cv[mid]["means"]["original"] - cv["wc120_raw"]["means"]["original"]
        length_loss = max(length_aucs[("wc120_raw", b)] - length_aucs[(mid, b)] for b in range(3))
        dash_loss = base_dash_delta - (aucs[(mid, "dash_typography")] - aucs[(mid, "original")])
        checks = {
            "quote_cv_mean_improves_own_raw": float(np.mean(gains)) > 0,
            "quote_cv_positive_fold_count_passes": sum(g > 0 for g in gains) >= rules["minimum_positive_quote_cv_folds"],
            "quote_validation_interval_lower_vs_incumbent_positive": uncertainty["comparisons"][f"{rep}_quote_vs_incumbent"]["interval95"][0] > 0,
            "clean_point_within_budget_vs_incumbent": clean_delta >= -rules["maximum_clean_point_regression_vs_incumbent"],
            "clean_cv_within_budget_vs_incumbent": cv_delta >= -rules["maximum_clean_cv_regression_vs_incumbent"],
            "length_guard_passes_vs_incumbent": length_loss <= rules["maximum_length_bin_regression_vs_incumbent"],
            "dash_guard_passes_vs_incumbent": dash_loss <= rules["maximum_extra_dash_loss_vs_incumbent"],
        }
        actual = selection["eligibility"][mid]
        require(actual["checks"] == checks and actual["eligible"] == all(checks.values()), "Incorrect export eligibility")
        for name, expected in (("quote_cv_fold_gains", gains), ("clean_delta_vs_incumbent", clean_delta),
                               ("clean_cv_delta_vs_incumbent", cv_delta),
                               ("maximum_length_bin_regression_vs_incumbent", length_loss),
                               ("extra_dash_loss_vs_incumbent", dash_loss)):
            np.testing.assert_allclose(actual[name], expected, atol=1e-14, rtol=0)
        decisions[mid] = all(checks.values())
    selected = next((mid for mid in rules["candidate_priority"] if decisions[mid]), None)
    category = None
    if selected:
        rep = selected.removesuffix("_norm")
        category = ("clean_and_quote_improvement" if uncertainty["comparisons"][f"{rep}_clean_vs_incumbent"]["interval95"][0] > 0
                    else "exploratory_robustness_tradeoff")
    require(selection["selected_model_id"] == selected and selection["export_eligible"] == (selected is not None),
            "Incorrect candidate selection")
    require(selection["candidate_category"] == category and selection["retain_incumbent_until_external_check"] and
            not selection["new_kaggle_upload"], "Invalid candidate interpretation/upload claim")
    audit["export_eligibility_and_selection_reconstructed"] = True
    audit["selected_model_id"] = selected
    audit["export_eligible"] = selected is not None
    audit["candidate_category"] = category
    candidate_files = {p.resolve() for p in (ROOT / "submissions").glob(f"{prefix}_*.csv")}
    expected_files = {(ROOT / selection["submission_csv"]).resolve()} if selected else set()
    require(candidate_files == expected_files and len(candidate_files) <= rules["maximum_new_candidates"],
            "Unexpected additional candidate CSVs")

    if selected:
        csv_path = ROOT / selection["submission_csv"]
        meta = read_json(csv_path.with_suffix(".metadata.json"))
        folder = base / meta["model_run_id"]
        verify_receipt(folder)
        cfg = read_json(folder / "config.json")
        require(cfg["n_train"] == 10000 and cfg["held_out_evaluation"] is None and cfg["normalize_quotes"],
                "Incorrect full-label training configuration")
        require([r["id"] for r in read_csv(folder / "train_ids.csv")] == train_ids + validation_ids,
                "Full-label candidate used incorrect training IDs")
        require(read_json(folder / "metrics.json")["validation"] is None, "Full-label refit has a held-out score")
        require(meta["n_train"] == 10000 and meta["n_test"] == 3000 and meta["full_refit"] and
                meta["full_refit_held_out_evaluation"] is None and not meta["uploaded_to_kaggle"],
                "Invalid export sizes/evaluation/upload claim")
        require(meta["quote_normalization"] and meta["C"] == 1 and meta["candidate_category"] == category,
                "Incorrect exported preprocessing/category")
        require(meta["csv_sha256"] == sha256(csv_path) and meta["test_sha256"] == sha256(raw_path("test.csv")) and
                meta["model_sha256"] == sha256(folder / "model.joblib"), "Incorrect export checksums")
        require(check(csv_path, "submission") == 3000, "Incorrect submission contract")
        test, exported = read_csv(raw_path("test.csv")), read_csv(csv_path)
        require([r["id"] for r in exported] == [r["id"] for r in test], "Incorrect test ID order")
        with threadpool_limits(limits=1):
            model = joblib.load(folder / "model.joblib")
            verify_structure(model, selected.removesuffix("_norm"), True, features, [r["text"] for r in test[:32]])
            restored = predict_ai(model, [r["text"] for r in test])
        np.testing.assert_array_equal(restored, np.asarray([float(r["label"]) for r in exported]))
        require(meta["model_reload_prediction_max_error"] == 0, "Nonzero recorded export reload error")
        audit["full_label_export"] = {"csv": str(csv_path.relative_to(ROOT)), "n_test": 3000,
                                      "model_reload_prediction_max_error": 0.0, "uploaded_to_kaggle": False}
    else:
        require("submission_csv" not in selection, "Ineligible candidate was exported")
    audit["all_checks_passed"] = True
    audit["verifier_source_sha256"] = sha256(Path(__file__))
    audit["limitations"] = "Conditional development verification; constructed invariance is not independent OOD evidence. No new model was fitted by this verifier."
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, audit)
    print(f"Saved quote-study verification: {output}", flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.prefix, args.output)


if __name__ == "__main__":
    main()
