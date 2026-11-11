"""Independently audit the four-group richness ablation without refitting models."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, SPLIT_DIR, load_partitions, raw_path, read_csv, sha256
from .q5_sensitivity import CONDITIONS, apply_condition
from .run_q1 import id_hash, predict_ai
from .run_q5 import atomic_json
from .verify_q5_tune import read_json, require, verify_receipt


MODEL_IDS = ("c60", "c60_length", "c60_richness", "c60_length_richness")
ALL_MODEL_IDS = (*MODEL_IDS, "incumbent_wc120")


def numeric_blocks(model_id):
    return [name for name in ("length", "richness") if name in model_id]


def independent_document_features(texts, block):
    """Recompute the frozen formulas independently of the training transformer."""
    rows = []
    for text in texts:
        tokens = re.findall(r"(?u)\b[^\W\d_]+\b", text.lower())
        n = len(tokens)
        if block == "length":
            values = [math.log1p(n), math.log1p(len(text))]
        elif block == "richness":
            width = min(50, n)
            diversity = (sum(len(set(tokens[i:i + width])) / width
                             for i in range(n - width + 1)) / (n - width + 1)
                         if width else 0.0)
            counts = Counter(tokens)
            entropy = (-sum((v / n) * math.log(v / n) for v in counts.values()) / math.log(n)
                       if n >= 2 else 0.0)
            bigrams = list(zip(tokens, tokens[1:]))
            repetition = ((len(bigrams) - len(set(bigrams))) / len(bigrams)
                          if bigrams else 0.0)
            values = [diversity, entropy, repetition]
        else:
            raise ValueError(f"Unknown numeric block: {block}")
        rows.append(values)
    return np.asarray(rows, dtype=np.float64).reshape(len(texts), 2 if block == "length" else 3)


def verify_vectorizer(vectorizer, feature_plan):
    require(isinstance(vectorizer, TfidfVectorizer), "Character block is not TF-IDF")
    original = feature_plan["char_representation"]
    for parameter in ("analyzer", "min_df", "max_df", "max_features", "lowercase",
                      "strip_accents", "sublinear_tf", "smooth_idf", "use_idf", "norm"):
        require(getattr(vectorizer, parameter) == original[parameter],
                f"Changed character TF-IDF setting: {parameter}")
    require(vectorizer.ngram_range == tuple(original["ngram_range"]), "Changed character range")
    require(vectorizer.dtype == np.float64, "Changed character matrix dtype")
    return len(vectorizer.vocabulary_)


def assert_scaler(scaler, values, context):
    require(isinstance(scaler, StandardScaler) and scaler.with_mean and scaler.with_std,
            f"Incorrect learned scaling: {context}")
    require(np.all(np.asarray(scaler.n_samples_seen_) == len(values)),
            f"Scaler saw incorrect row count: {context}")
    np.testing.assert_allclose(scaler.mean_, np.mean(values, axis=0), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(scaler.var_, np.var(values, axis=0), atol=1e-12, rtol=1e-12)
    expected_scale = np.sqrt(np.var(values, axis=0))
    expected_scale[expected_scale == 0] = 1
    np.testing.assert_allclose(scaler.scale_, expected_scale, atol=1e-12, rtol=1e-12)


def verify_structure(model, model_id, feature_plan, train_numeric, sample_text):
    require(isinstance(model, CalibratedClassifierCV), "Missing sigmoid calibration")
    require(model.method == "sigmoid" and model.ensemble is False and model.n_jobs == 1,
            "Changed calibration settings")
    require(model.cv.n_splits == 3 and model.cv.shuffle and
            model.cv.random_state == feature_plan["seeds"]["calibration"],
            "Changed calibration fold protocol")
    require(isinstance(model.estimator, Pipeline) and
            list(model.estimator.named_steps) == ["features", "classifier"],
            "Complete features/classifier pipeline must stay inside calibration")
    classifier = model.estimator.named_steps["classifier"]
    for parameter, value in feature_plan["classifier"].items():
        require(getattr(classifier, parameter) == value, f"Changed classifier setting: {parameter}")
    require(classifier.C == 1 and classifier.random_state == feature_plan["seeds"]["model"],
            "Changed classifier seed or C")
    require(set(model.classes_) == {0, 1}, "Incorrect class-1 probability direction")
    fitted = model.calibrated_classifiers_[0].estimator
    features = fitted.named_steps["features"]
    if model_id == "incumbent_wc120":
        from .verify_q5_tune import verify_model_structure
        structure = verify_model_structure(model, "wc120", 1, feature_plan, sample_text)
        return {"feature_counts": {"char": 60000, "word": 60000, "length": 0,
                                    "richness": 0, "total": structure["n_features"]},
                "complete_pipeline_inside_calibration": True}
    expected_blocks = numeric_blocks(model_id)
    if not expected_blocks:
        n_char = verify_vectorizer(features, feature_plan)
        vectorizer = features
    else:
        require(isinstance(features, FeatureUnion), "Expected direct block concatenation")
        children = dict(features.transformer_list)
        require(list(children) == ["char", *expected_blocks], "Changed feature-block membership/order")
        expected_weights = {"char": 1.0, **{b: 1 / math.sqrt(2 if b == "length" else 3)
                                            for b in expected_blocks}}
        require(set(features.transformer_weights) == set(expected_weights), "Changed block weights")
        for block, value in expected_weights.items():
            np.testing.assert_allclose(features.transformer_weights[block], value, atol=1e-15, rtol=0)
        vectorizer = children["char"]
        n_char = verify_vectorizer(vectorizer, feature_plan)
        for block in expected_blocks:
            numeric = children[block]
            require(isinstance(numeric, Pipeline) and list(numeric.named_steps) == ["extract", "scale"],
                    "Numeric extraction and learned scaling must stay together")
            extracted = numeric.named_steps["extract"].transform(sample_text)
            expected = independent_document_features(sample_text, block)
            np.testing.assert_allclose(extracted, expected, atol=1e-12, rtol=1e-12)
            if block == "richness":
                require(numeric.named_steps["extract"].window_size == 50, "Changed richness window")
            assert_scaler(numeric.named_steps["scale"], train_numeric[block], f"{model_id}/{block}")
    transformed = features.transform(sample_text)
    char_matrix = vectorizer.transform(sample_text)
    np.testing.assert_array_equal(transformed[:, :n_char].toarray(), char_matrix.toarray())
    n_numeric = sum(2 if b == "length" else 3 for b in expected_blocks)
    require(transformed.shape[1] == n_char + n_numeric, "Wrong concatenated feature dimension")
    counts = {"char": n_char, "word": 0, "length": 2 if "length" in expected_blocks else 0,
              "richness": 3 if "richness" in expected_blocks else 0,
              "total": n_char + n_numeric}
    vocabulary = sorted((token, int(index)) for token, index in vectorizer.vocabulary_.items())
    digest = hashlib.sha256(json.dumps(vocabulary, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    return {"feature_counts": counts, "complete_pipeline_inside_calibration": True,
            "char_vocab_sha256": digest,
            "numeric_scaling_training_only_checked": expected_blocks,
            "character_coordinates_preserved_without_final_row_normalization": True}


def verify_cv(study, plan, train, train_numeric):
    ids = [r["id"] for r in train]
    labels = np.asarray([int(r["label"]) for r in train])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["inner_cv"]).split(ids, labels))
    assignments = read_csv(study / "inner_folds.csv")
    require([r["id"] for r in assignments] == ids, "Changed CV assignment ID order")
    for fold, (_, held) in enumerate(folds):
        require(all(int(assignments[i]["inner_validation_fold"]) == fold for i in held), "Changed CV folds")
    progress = read_json(study / "cv_progress.json")
    summaries = read_csv(study / "cv_results.csv")
    require(len(progress) == 12 and len(summaries) == 4 and
            {(r["model_id"], r["fold"]) for r in progress} ==
            {(mid, fold) for mid in MODEL_IDS for fold in range(3)}, "Incomplete/duplicate CV matrix")
    require({r["model_id"] for r in summaries} == set(MODEL_IDS), "Changed CV configurations")
    numeric = {}
    for mid in MODEL_IDS:
        scores = {condition: [] for condition in CONDITIONS}
        summary = next(r for r in summaries if r["model_id"] == mid)
        for fold, (fit, held) in enumerate(folds):
            record = next(r for r in progress if r["model_id"] == mid and r["fold"] == fold)
            require(record["fit_ids_sha256"] == id_hash([ids[i] for i in fit]) and
                    record["held_ids_sha256"] == id_hash([ids[i] for i in held]),
                    "Changed CV fitting or held-out IDs")
            predicted = read_csv(study / "cv_predictions" / f"{mid}_fold_{fold}.csv")
            require(sha256(study / "cv_predictions" / f"{mid}_fold_{fold}.csv") == record["predictions_sha256"],
                    "Changed held-out margin checkpoint")
            require([(r["id"], int(r["label"])) for r in predicted] ==
                    [(ids[i], int(labels[i])) for i in held], "Wrong held-out margin IDs/labels/order")
            for condition, field, margin_field, suffix in (
                    ("original", "original_auroc", "original_margin", ""),
                    ("quote_typography", "quote_auroc", "quote_margin", "_quote"),
                    ("dash_typography", "dash_auroc", "dash_margin", "_dash")):
                margins = np.asarray([float(r[margin_field]) for r in predicted])
                require(np.all(np.isfinite(margins)), "Nonfinite saved CV margins")
                auc = float(roc_auc_score(labels[held], margins))
                np.testing.assert_allclose(record[field], auc, atol=1e-14, rtol=0)
                np.testing.assert_allclose(float(summary[f"fold_{fold}{suffix}_auroc"]), auc, atol=1e-14, rtol=0)
                scores[condition].append(auc)
            feature_audit = record["feature_audit"]
            require(set(feature_audit["scalers"]) == set(numeric_blocks(mid)), "Wrong CV numeric blocks")
            require(re.fullmatch(r"[0-9a-f]{64}", feature_audit["char_vocab_sha256"]) is not None,
                    "Invalid vocabulary receipt hash")
            for block in numeric_blocks(mid):
                actual = feature_audit["scalers"][block]
                values = train_numeric[block][fit]
                np.testing.assert_allclose(actual["mean"], values.mean(axis=0), atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(actual["var"], values.var(axis=0), atol=1e-12, rtol=1e-12)
                require(np.all(np.asarray(actual["n_samples_seen"]) == len(fit)), "CV scaler saw held-out rows")
                expected_scale = np.sqrt(values.var(axis=0))
                expected_scale[expected_scale == 0] = 1
                np.testing.assert_allclose(actual["scale"], expected_scale, atol=1e-12, rtol=1e-12)
            require(int(record["n_features"]) == feature_audit["feature_counts"]["total"], "Wrong CV feature dimension")
            counts = feature_audit["feature_counts"]
            require(0 < counts["char"] <= 60000 and counts["length"] == (2 if "length" in numeric_blocks(mid) else 0)
                    and counts["richness"] == (3 if "richness" in numeric_blocks(mid) else 0) and
                    counts["total"] == counts["char"] + counts["length"] + counts["richness"],
                    "Incorrect CV numeric block dimensions")
        means = {condition: float(np.mean(values)) for condition, values in scores.items()}
        for condition, column in (("original", "mean_cv_auroc"), ("quote_typography", "mean_quote_cv_auroc"),
                                  ("dash_typography", "mean_dash_cv_auroc")):
            np.testing.assert_allclose(float(summary[column]), means[condition], atol=1e-14, rtol=0)
        np.testing.assert_allclose(float(summary["std_cv_auroc"]), np.std(scores["original"], ddof=1), atol=1e-14, rtol=0)
        numeric[mid] = {"scores": scores, "means": means}
    historical = read_json(study.parent / plan["reference_study"] / "cv_progress.json")
    for fold in range(3):
        old = next(r["auroc"] for r in historical if r["model"] == "c60" and r["fold"] == fold)
        np.testing.assert_allclose(numeric["c60"]["scores"]["original"][fold], old, atol=1e-12, rtol=0)
        hashes = {r["feature_audit"]["char_vocab_sha256"] for r in progress if r["fold"] == fold}
        require(len(hashes) == 1, "Character vocabulary differs between matched configurations")
    return numeric


def verify(prefix, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    base = ROOT / "results/supervised"
    study = base / f"{prefix}_study"
    plan = read_json(study / "q5_richness_plan.json")
    feature_plan = read_json(study / "q5_feature_plan.json")
    context = read_json(study / "provenance.json")
    require(read_json(study / "status.json")["stage"] == "complete", "Richness study incomplete")
    require(plan["C"] == 1 and plan["conditions"] == list(CONDITIONS), "Changed fixed study settings")
    require(read_json(ROOT / "experiments/supervised/q5_richness_plan.json") == plan,
            "Executed richness protocol differs from current source")
    require(read_json(ROOT / plan["feature_plan"]) == feature_plan, "Changed historical character feature plan")
    for relative, expected in context["source_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"Changed executing source: {relative}")
        require(sha256(study / "source" / relative) == expected, f"Changed source snapshot: {relative}")
    for relative, expected in read_json(study / "source_guards.json").items():
        require(sha256(ROOT / relative) == expected, f"Changed prior evidence: {relative}")
    train, validation = load_partitions()
    train_ids, validation_ids = [r["id"] for r in train], [r["id"] for r in validation]
    labels = np.asarray([int(r["label"]) for r in validation])
    text = [r["text"] for r in validation]
    require(len(train) == 8000 and len(validation) == 2000, "Unexpected canonical partition sizes")
    require(not set(train_ids) & set(validation_ids), "Canonical split overlap")
    require(read_json(study / "split_manifest.json") == read_json(SPLIT_DIR / "manifest.json"), "Changed split manifest")
    for name, ids in (("train", train_ids), ("validation", validation_ids)):
        require([r["id"] for r in read_csv(study / f"{name}_ids.csv")] == ids, f"Wrong {name} IDs/order")
    train_numeric = {block: independent_document_features([r["text"] for r in train], block)
                     for block in ("length", "richness")}
    cv = verify_cv(study, plan, train, train_numeric)
    comparison = read_csv(study / "comparison.csv")
    require(len(comparison) == 15 and {(r["model_id"], r["condition"]) for r in comparison} ==
            {(mid, condition) for mid in ALL_MODEL_IDS for condition in CONDITIONS},
            "Incorrect or incomplete four-group/reference condition matrix")
    audit = {"source_hashes_match": True, "prior_evidence_unchanged": True,
             "canonical_split_and_fold_ids_match": True, "n_cv_fits": 12,
             "cv_aurocs_recomputed_from_held_out_margins": True,
             "fold_scalers_recomputed_from_training_only_rows": True,
             "n_matrix_conditions": 15, "n_validation": len(validation),
             "full_size_refits_performed_by_verifier": False, "models": {}}
    probabilities, aucs = {}, {}
    with threadpool_limits(limits=1):
        for mid in ALL_MODEL_IDS:
            folder = base / ("a_q5_20261108_v1_c60" if mid == "c60" else
                             "a_q5_20261108_v1_wc120" if mid == "incumbent_wc120" else f"{prefix}_{mid}")
            verify_receipt(folder)
            model = joblib.load(folder / "model.joblib")
            structure = verify_structure(model, mid, feature_plan, train_numeric, text[:32])
            metrics = read_json(folder / "metrics.json")
            if mid not in ("c60", "incumbent_wc120"):
                cfg = read_json(folder / "config.json")
                require(cfg["model_id"] == mid and cfg["C"] == 1 and cfg["n_train"] == 8000,
                        "Incorrect four-group model configuration")
                require([r["id"] for r in read_csv(folder / "train_ids.csv")] == train_ids,
                        "Model saw different training IDs")
                require(read_json(folder / "model_fit.json")["model_sha256"] == sha256(folder / "model.joblib"),
                        "Changed model checkpoint")
            model_audit = {**structure, "saved_prediction_max_error": 0.0, "conditions": {}}
            for condition in CONDITIONS:
                row = next(r for r in comparison if r["model_id"] == mid and r["condition"] == condition)
                inputs = [apply_condition(document, condition) for document in text]
                predicted = predict_ai(model, inputs)
                path = study / "predictions" / f"{mid}_{condition}.csv"
                require(check(path, "validation") == len(validation), "Invalid validation prediction contract")
                records = read_csv(path)
                require([(r["id"], int(r["label"])) for r in records] ==
                        [(r["id"], int(r["label"])) for r in validation], "Wrong prediction IDs/labels/order")
                saved = np.asarray([float(r["p_ai"]) for r in records])
                np.testing.assert_array_equal(predicted, saved)
                key = f"{mid}__{condition}"
                probabilities[key] = saved
                aucs[(mid, condition)] = float(roc_auc_score(labels, saved))
                changed = sum(before != after for before, after in zip(text, inputs, strict=True))
                require(int(row["n"]) == len(validation) and int(row["n_changed"]) == changed,
                        "Incorrect condition population or change count")
                np.testing.assert_allclose(float(row["validation_auroc"]), aucs[(mid, condition)], atol=1e-14, rtol=0)
                model_audit["conditions"][condition] = {"auroc": aucs[(mid, condition)],
                                                       "n_changed": changed, "prediction_sha256": sha256(path)}
            source_p = np.asarray([float(r["p_ai"]) for r in read_csv(folder / "validation_predictions.csv")])
            np.testing.assert_array_equal(probabilities[f"{mid}__original"], source_p)
            require(metrics["validation"]["n"] == len(validation) and metrics["train"]["n"] == len(train),
                    "Wrong recorded fitting/evaluation sizes")
            np.testing.assert_allclose(metrics["validation"]["auroc"], aucs[(mid, "original")], atol=1e-14, rtol=0)
            if mid not in ("c60", "incumbent_wc120"):
                feature_audit = metrics["feature_audit"]
                for key, value in feature_audit["feature_counts"].items():
                    require(structure["feature_counts"][key] == value, "Incorrect recorded feature dimensions")
                require(feature_audit["char_vocab_sha256"] == structure["char_vocab_sha256"],
                        "Incorrect final character-vocabulary hash")
                require(set(feature_audit["scalers"]) == set(numeric_blocks(mid)), "Incorrect final scaler blocks")
                for block in numeric_blocks(mid):
                    scaler = feature_audit["scalers"][block]
                    values = train_numeric[block]
                    np.testing.assert_allclose(scaler["mean"], values.mean(axis=0), atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(scaler["var"], values.var(axis=0), atol=1e-12, rtol=1e-12)
                    require(scaler["n_samples_seen"] == len(train), "Recorded final scaler sample count differs")
            audit["models"][mid] = model_audit
            print(f"Verified richness matrix {mid}: clean={aucs[(mid, 'original')]:.6f}", flush=True)
            del model
    uncertainty = read_json(study / "bootstrap.json")
    require(uncertainty["seed"] == plan["seeds"]["bootstrap"] and uncertainty["replicates"] ==
            plan["evaluation"]["bootstrap_replicates"], "Changed bootstrap protocol")
    require(uncertainty["primary_comparison_key"] == "richness_given_length", "Changed primary contrast")
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    rng = np.random.default_rng(plan["seeds"]["bootstrap"])
    sampled = {key: [] for key in probabilities}
    for _ in range(uncertainty["replicates"]):
        ix = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for key, p in probabilities.items():
            sampled[key].append(float(roc_auc_score(labels[ix], p[ix])))
    for key, values in sampled.items():
        np.testing.assert_allclose(uncertainty["auroc_95_intervals"][key], np.quantile(values, [.025, .975]), atol=1e-14, rtol=0)
    contrasts = {"richness_given_length": ("c60_length_richness", "c60_length"),
                 "length_only": ("c60_length", "c60"),
                 "richness_only": ("c60_richness", "c60"),
                 "length_given_richness": ("c60_length_richness", "c60_richness")}
    for mid in MODEL_IDS[1:]:
        contrasts[f"{mid}_vs_incumbent"] = (mid, "incumbent_wc120")
        contrasts[f"{mid}_vs_c60"] = (mid, "c60")
    require(set(uncertainty["comparisons"]) == {*contrasts, "interaction"}, "Changed pairwise contrast set")
    for name, (left_mid, right_mid) in contrasts.items():
        left, right = f"{left_mid}__original", f"{right_mid}__original"
        actual = uncertainty["comparisons"][name]
        require((actual["left"], actual["right"]) == (left, right), "Incorrect matched contrast definition")
        require(actual["terms"] == [[left, 1], [right, -1]], "Incorrect saved contrast coefficients")
        np.testing.assert_allclose(actual["point_delta"], aucs[(left_mid, "original")] - aucs[(right_mid, "original")], atol=1e-14, rtol=0)
        np.testing.assert_allclose(actual["interval95"], np.quantile(np.asarray(sampled[left]) - sampled[right], [.025, .975]), atol=1e-14, rtol=0)
    interaction_terms = [("c60_length_richness", 1), ("c60_length", -1), ("c60_richness", -1), ("c60", 1)]
    interaction = uncertainty["comparisons"]["interaction"]
    require(interaction["terms"] == [[f"{mid}__original", weight] for mid, weight in interaction_terms],
            "Incorrect four-term interaction definition")
    delta = sum(weight * aucs[(mid, "original")] for mid, weight in interaction_terms)
    distribution = sum(weight * np.asarray(sampled[f"{mid}__original"]) for mid, weight in interaction_terms)
    np.testing.assert_allclose(interaction["point_delta"], delta, atol=1e-14, rtol=0)
    np.testing.assert_allclose(interaction["interval95"], np.quantile(distribution, [.025, .975]), atol=1e-14, rtol=0)
    audit["all_bootstrap_intervals_reconstructed"] = True
    boundaries = read_json(study / "length_boundaries.json")
    cuts = np.quantile([len(r["text"].split()) for r in train], [1 / 3, 2 / 3])
    np.testing.assert_allclose(boundaries["training_word_count_tertiles"], cuts, atol=1e-14, rtol=0)
    bins = np.digitize([len(r["text"].split()) for r in validation], cuts, right=True)
    lengths = read_csv(study / "length_subgroups.csv")
    require(len(lengths) == 15 and {(r["model_id"], int(r["length_bin"])) for r in lengths} ==
            {(mid, b) for mid in ALL_MODEL_IDS for b in range(3)}, "Wrong length subgroup matrix")
    length_aucs = {}
    for row in lengths:
        mid, b = row["model_id"], int(row["length_bin"])
        mask = bins == b
        require(int(row["n"]) == int(mask.sum()) and int(row["n_human"]) == int(np.sum(labels[mask] == 0)) and
                int(row["n_ai"]) == int(np.sum(labels[mask] == 1)), "Incorrect length subgroup counts")
        auc = float(roc_auc_score(labels[mask], probabilities[f"{mid}__original"][mask]))
        np.testing.assert_allclose(float(row["validation_auroc"]), auc, atol=1e-14, rtol=0)
        length_aucs[(mid, b)] = auc
    audit["length_subgroups_reconstructed"] = True
    numeric_diagnostics = read_json(study / "numeric_feature_diagnostics.json")
    for name, records in (("train", train), ("validation", validation)):
        values = np.column_stack([independent_document_features([r["text"] for r in records], block)
                                  for block in ("length", "richness")])
        actual = numeric_diagnostics[name]
        require(actual["n"] == len(records), "Wrong numeric-diagnostic row count")
        for field, expected in (("mean", values.mean(axis=0)), ("std", values.std(axis=0)),
                                ("min", values.min(axis=0)), ("max", values.max(axis=0)),
                                ("pearson_correlations", np.corrcoef(values, rowvar=False))):
            np.testing.assert_allclose(actual[field], expected, atol=1e-12, rtol=1e-12)
        require(actual["n_shorter_than_50_tokens"] == int(np.sum(np.expm1(values[:, 0]) < 49.5)),
                "Wrong short-document count")
    audit["document_local_feature_diagnostics_reconstructed"] = True

    selection = read_json(study / "selection.json")
    chosen = max(MODEL_IDS[1:], key=lambda mid: (cv[mid]["means"]["original"],
                                               -sum(2 if b == "length" else 3 for b in numeric_blocks(mid)),
                                               -MODEL_IDS.index(mid)))
    prevalidation = read_json(study / "cv_candidate.json")
    require(prevalidation["model_id"] == chosen and not prevalidation["selection_uses_validation"],
            "Incorrect training-only candidate choice")
    np.testing.assert_allclose(prevalidation["mean_cv_auroc"], cv[chosen]["means"]["original"], atol=1e-14, rtol=0)
    require(prevalidation["rule"] == plan["export_rule"]["selection"], "Changed CV selection rule")
    historical = read_json(study.parent / plan["reference_study"] / "cv_progress.json")
    incumbent_cv = [next(r["auroc"] for r in historical if r["model"] == "wc120" and r["fold"] == fold)
                    for fold in range(3)]
    gains = [value - reference for value, reference in zip(cv[chosen]["scores"]["original"], incumbent_cv, strict=True)]
    worst_length = max(length_aucs[("incumbent_wc120", b)] - length_aucs[(chosen, b)] for b in range(3))
    extra_loss = {condition: (aucs[("incumbent_wc120", condition)] - aucs[("incumbent_wc120", "original")]) -
                  (aucs[(chosen, condition)] - aucs[(chosen, "original")]) for condition in CONDITIONS[1:]}
    rules = plan["export_rule"]
    checks = {
        "positive_validation_interval_lower_vs_incumbent": uncertainty["comparisons"][f"{chosen}_vs_incumbent"]["interval95"][0] > 0,
        "positive_validation_interval_lower_vs_c60": uncertainty["comparisons"][f"{chosen}_vs_c60"]["interval95"][0] > 0,
        "positive_mean_cv_delta_vs_incumbent": float(np.mean(gains)) > 0,
        "positive_cv_fold_count_vs_incumbent": sum(g > 0 for g in gains) >= rules["minimum_positive_cv_folds_vs_incumbent"],
        "length_guard": worst_length <= rules["maximum_length_bin_regression_vs_incumbent"],
        "typography_guard": max(extra_loss.values()) <= rules["maximum_extra_typography_loss_vs_incumbent"],
    }
    require(set(selection["eligibility"]) == {chosen}, "Runner-up was considered after candidate selection")
    recorded_eligibility = selection["eligibility"][chosen]
    eligible = all(checks.values())
    require(recorded_eligibility["checks"] == checks and recorded_eligibility["eligible"] == eligible,
            "Incorrect export-eligibility decision")
    for field, expected in (("cv_fold_gains_vs_incumbent", gains),
                            ("maximum_length_bin_regression_vs_incumbent", worst_length)):
        np.testing.assert_allclose(recorded_eligibility[field], expected, atol=1e-14, rtol=0)
    require(set(recorded_eligibility["extra_typography_loss_vs_incumbent"]) == set(extra_loss),
            "Changed typography guard axes")
    for condition, value in extra_loss.items():
        np.testing.assert_allclose(recorded_eligibility["extra_typography_loss_vs_incumbent"][condition], value, atol=1e-14, rtol=0)
    primary_gains = [a - b for a, b in zip(cv["c60_length_richness"]["scores"]["original"],
                                         cv["c60_length"]["scores"]["original"], strict=True)]
    primary_checks = {
        "positive_validation_interval_lower": uncertainty["comparisons"]["richness_given_length"]["interval95"][0] > 0,
        "positive_mean_cv_delta": float(np.mean(primary_gains)) > 0,
        "minimum_positive_cv_folds": sum(g > 0 for g in primary_gains) >= plan["primary_evidence_rule"]["minimum_positive_cv_folds"],
    }
    require(selection["primary_comparison_key"] == "richness_given_length" and
            selection["primary_evidence"]["checks"] == primary_checks and
            selection["primary_evidence"]["all_passed"] == all(primary_checks.values()),
            "Incorrect primary-evidence interpretation")
    np.testing.assert_allclose(selection["primary_evidence"]["cv_fold_gains"], primary_gains, atol=1e-14, rtol=0)
    selected = chosen if eligible else None
    category = "development_improvement_candidate" if eligible else None
    require(selection["cv_selected_model_id"] == chosen and selection["selected_model_id"] == selected and
            selection["export_eligible"] == eligible and selection["candidate_category"] == category,
            "Incorrect exported candidate or category")
    require(selection["retain_incumbent_until_external_check"] and not selection["new_kaggle_upload"],
            "Incorrect incumbent/upload claim")
    candidate_files = {p.resolve() for p in (ROOT / "submissions").glob(f"{prefix}_*.csv")}
    expected_files = {(ROOT / selection["submission_csv"]).resolve()} if eligible else set()
    require(candidate_files == expected_files and len(candidate_files) <= rules["maximum_new_candidates"],
            "Unexpected extra submission CSV")
    audit.update(export_eligibility_and_selection_reconstructed=True, cv_selected_model_id=chosen,
                 selected_model_id=selected, export_eligible=eligible, candidate_category=category,
                 primary_evidence_all_passed=all(primary_checks.values()))
    if eligible:
        csv_path = ROOT / selection["submission_csv"]
        meta = read_json(csv_path.with_suffix(".metadata.json"))
        folder = base / meta["model_run_id"]
        verify_receipt(folder)
        cfg = read_json(folder / "config.json")
        require(cfg["model_id"] == selected and cfg["n_train"] == 10000 and cfg["held_out_evaluation"] is None,
                "Incorrect full-label model configuration")
        require([r["id"] for r in read_csv(folder / "train_ids.csv")] == train_ids + validation_ids,
                "Full-label refit used incorrect IDs/order")
        require(read_json(folder / "metrics.json")["validation"] is None, "Full-label model has a held-out score")
        require(meta["n_train"] == 10000 and meta["n_test"] == 3000 and meta["full_refit"] and
                meta["full_refit_held_out_evaluation"] is None and not meta["uploaded_to_kaggle"],
                "Incorrect export sizes or evaluation/upload claim")
        require(meta["C"] == 1 and meta["model_id"] == selected and meta["candidate_category"] == category,
                "Incorrect exported settings/category")
        require(meta["csv_sha256"] == sha256(csv_path) and meta["model_sha256"] == sha256(folder / "model.joblib") and
                meta["test_sha256"] == sha256(raw_path("test.csv")), "Incorrect export checksum")
        require(check(csv_path, "submission") == 3000, "Invalid submission contract")
        test, exported = read_csv(raw_path("test.csv")), read_csv(csv_path)
        require([r["id"] for r in exported] == [r["id"] for r in test], "Changed test ID order")
        all_training_numeric = {block: independent_document_features([r["text"] for r in train + validation], block)
                                for block in ("length", "richness")}
        with threadpool_limits(limits=1):
            model = joblib.load(folder / "model.joblib")
            verify_structure(model, selected, feature_plan, all_training_numeric, [r["text"] for r in test[:32]])
            restored = predict_ai(model, [r["text"] for r in test])
        np.testing.assert_array_equal(restored, np.asarray([float(r["label"]) for r in exported]))
        require(meta["model_reload_prediction_max_error"] == 0, "Nonzero recorded reload error")
        audit["full_label_export"] = {"csv": str(csv_path.relative_to(ROOT)), "n_test": 3000,
                                      "model_reload_prediction_max_error": 0.0, "uploaded_to_kaggle": False}
    else:
        require("submission_csv" not in selection, "Ineligible candidate was exported")
    audit["all_checks_passed"] = True
    audit["verifier_source_sha256"] = sha256(Path(__file__))
    audit["limitations"] = "Conditional development verification; no new model fitted. Length control is explicit and incomplete; no causal or hidden-test inference."
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, audit)
    print(f"Saved richness-study verification: {output}", flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.prefix, args.output)


if __name__ == "__main__":
    main()
