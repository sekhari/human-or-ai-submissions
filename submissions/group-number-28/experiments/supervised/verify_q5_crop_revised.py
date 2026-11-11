"""Independent, zero-fit verification of the explicitly revised crop protocol.

The prior run remains immutable. Only the minimum-content-topic count and
maximum-held-fold size become diagnostics. All predictive gates are recomputed
from saved ID-aligned predictions; no classifier or feature learner is fitted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss, normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from human_ai.data import ROOT, load_partitions, raw_path, read_csv, sha256
from .verify_q5_crop import (
    PRIMARY, SEALED, assignments, close, demand, expected_texts, read_json,
    verify_hashes, verify_idf, verify_raw,
)


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parent_path(plan):
    value = plan["parent_study"]
    if isinstance(value, dict):
        value = value["path"]
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def match_random(labels, topic_folds, seed):
    rng = np.random.default_rng(seed)
    expected = np.full(len(labels), -1, dtype=int)
    for label in (0, 1):
        permutation = rng.permutation(np.flatnonzero(labels == label))
        cursor = 0
        for fold in range(3):
            count = int(np.sum((labels == label) & (topic_folds == fold)))
            expected[permutation[cursor:cursor + count]] = fold
            cursor += count
        demand(cursor == len(permutation), "Matched random allocation missed originals")
    return expected


def verify_reused_topics(study, train, plan):
    parent = parent_path(plan)
    for name, digest in plan["parent_artifact_sha256"].items():
        source = Path(name)
        source = source if source.is_absolute() else parent / source
        demand(sha256(source) == digest, f"Changed parent artifact: {name}")
    reuse = read_json(study / "topic_reuse_receipt.json")
    demand(reuse["topic_grouping_fits_performed"] == 0 and reuse["new_blind_reviews_performed"] is False,
           "Revision unexpectedly refitted groups or claimed a new blind review")
    verify_hashes(study, read_json(study / "topic_finalization.json")["artifact_sha256"])
    parent_provenance = read_json(parent / "provenance.json")
    verify_hashes(parent / "source", parent_provenance["source_sha256"])
    demand(read_json(parent / "completion.json")["reason"] == "topic_proxy_invalid_diagnostic_random_only",
           "Revision parent does not match the recorded original stop")
    for name in ("topic_assignments.csv", "topic_folds.csv", "topic_review.json", "topic_review_packet.json",
                 "topic_proxy.joblib", "topic_preparation.json", "topic_preparation_context.json"):
        demand(sha256(study / name) == sha256(parent / name), f"Inherited topic artifact changed: {name}")
    for original, inherited in (("topic_finalization.json", "parent_topic_finalization.json"),
                                ("topic_quality.json", "parent_topic_quality.json"),
                                ("plan.json", "parent_plan.json"),
                                ("completion.json", "parent_completion.json")):
        demand(sha256(study / inherited) == sha256(parent / original), f"Parent archive differs: {inherited}")
    ids = [row["id"] for row in train]
    labels = np.array([int(row["label"]) for row in train])
    rows = read_csv(study / "topic_assignments.csv")
    demand([row["id"] for row in rows] == ids, "Reused assignments are not canonical")
    topics = np.array([int(row["topic"]) for row in rows])
    topic_folds = assignments(study / "topic_folds.csv", ids)
    random_folds = assignments(study / "random_folds.csv", ids)
    parent_folds = assignments(parent / "topic_folds.csv", ids)
    demand(np.array_equal(topic_folds, parent_folds), "Revision changed topic fold IDs")
    for topic in set(topics.tolist()):
        demand(len(set(topic_folds[topics == topic].tolist())) == 1, "A topic crosses folds")
    demand(np.array_equal(random_folds, match_random(labels, topic_folds, plan["seeds"]["matched_random_folds"])),
           "Random folds differ from exact matched permutation")
    for fold in range(3):
        for label in (0, 1):
            demand(np.sum((random_folds == fold) & (labels == label)) ==
                   np.sum((topic_folds == fold) & (labels == label)), "Matched class counts differ")
    frozen = read_json(study / "topic_review.json")
    packet = read_json(study / "topic_review_packet.json")
    demand(frozen["labels_read_before_review_freeze"] is False, "Historical review did not attest blindness")
    demand(frozen["topic_review_packet_sha256"] == sha256(study / "topic_review_packet.json"), "Review packet hash mismatch")
    a, b = frozen["review_a"], frozen["review_b"]
    demand(a["reviewer"] != b["reviewer"], "Reviewers are not distinct")
    for review in (a, b):
        demand(review["reviewed_without_labels_or_scores"] is True, "Missing historical blind-review attestation")
        by_topic = {int(row["topic"]): row for row in review["clusters"]}
        demand(len(by_topic) == 12, "Incomplete historical review")
        for cluster in packet["clusters"]:
            demand(set(by_topic[cluster["topic"]]["inspected_ids"]) == set(cluster["sample_ids"]), "Historical review IDs differ")
    combined = {int(row["topic"]): row["decision"] for row in frozen["combined"]}
    n_content = sum(value == "content" for value in combined.values())
    n_style = sum(value == "style-format" for value in combined.values())
    q = plan["topic_proxy"]["quality_gates"]
    bounds = plan["topic_proxy"]["fold_quality_gates"]
    counts = np.bincount(topics, minlength=12)
    dominant = [max(np.sum(labels[topics == topic] == label) for label in (0, 1)) / counts[topic]
                if counts[topic] else None for topic in range(12)]
    nmi = float(normalized_mutual_info_score(topics, labels, average_method=q["nmi_average_method"]))
    # The changed maximum-held-n and minimum-content-count are deliberately
    # absent from the promotion checks, and remain visible diagnostics below.
    hard_checks = {
        "all_clusters_nonempty_and_large_enough": bool(np.all(counts >= q["minimum_cluster_n"])),
        "maximum_cluster_size": bool(np.all(counts <= q["maximum_cluster_n"])),
        "maximum_style_format_topics": n_style <= q["maximum_style_or_format_clusters"],
        "topic_label_nmi": nmi <= q["maximum_topic_label_nmi_inclusive"],
        "maximum_cluster_dominant_class_fraction": all(value is not None and value < q["maximum_cluster_dominant_class_fraction_exclusive"] for value in dominant),
    }
    valid_folds = []
    held_sizes = []
    for fold in range(3):
        y = labels[topic_folds == fold]
        held_sizes.append(len(y))
        valid_folds.append(len(y) >= bounds["minimum_held_n"]
                           and min(np.sum(y == 0), np.sum(y == 1)) >= bounds["minimum_held_n_per_class"]
                           and bounds["minimum_held_ai_fraction"] <= np.mean(y) <= bounds["maximum_held_ai_fraction"]
                           and len(y) < len(train))
    hard_checks["topic_groups_are_disjoint_across_folds"] = True
    hard_checks["topic_fold_structural_balance"] = all(valid_folds)
    hard_checks["matched_random_class_and_total_counts"] = True
    quality = read_json(study / "topic_quality.json")
    demand(quality["random_scheme"] == "matched_random", "Revision did not use matched random controls")
    demand(quality["topic_valid"] == all(hard_checks.values()), "Revised topic-valid flag differs")
    recorded_checks = quality.get("hard_checks", quality["checks"])
    demand(recorded_checks == hard_checks, "Revised structural/semantic hard checks differ")
    diagnostics = quality["diagnostics"]
    demand(set(diagnostics) == {"minimum_interpretable_content_topics", "maximum_held_n"},
           "Revision softened an unexpected group of constraints")
    demand(all(row["policy"] == "diagnostic_only" for row in diagnostics.values()), "Softened criteria still gate promotion")
    demand(diagnostics["minimum_interpretable_content_topics"]["observed"] == n_content,
           "Historical content diagnostic differs")
    demand(diagnostics["maximum_held_n"]["held_fold_sizes"] == held_sizes, "Historical size diagnostic differs")
    close(quality["topic_label_nmi"], nmi, "Topic-label NMI differs")
    demand(quality["n_content_topics"] == n_content and quality["n_style_format_topics"] == n_style,
           "Content/style diagnostic counts differ")
    demand("minimum_interpretable_content_topics" not in recorded_checks,
           "Historical content cutoff remains a hard gate")
    for row in read_csv(study / "fold_counts.csv"):
        assignment = topic_folds if row["scheme"] == "topic" else random_folds
        held = assignment == int(row["fold"])
        for key, expected in (("held_n", held.sum()), ("held_human", np.sum(held & (labels == 0))),
                              ("held_ai", np.sum(held & (labels == 1))), ("fitting_n", (~held).sum()),
                              ("fitting_human", np.sum(~held & (labels == 0))), ("fitting_ai", np.sum(~held & (labels == 1)))):
            demand(int(row[key]) == expected, f"Fold counts differ: {key}")
    return topics, random_folds, topic_folds, quality, {
        "parent_study": str(parent.relative_to(ROOT)), "no_topic_refit": True,
        "historical_content_topics": n_content, "held_topic_sizes": held_sizes,
        "retained_hard_checks": hard_checks,
        "two_thresholds_are_diagnostic": True,
        "adaptive_revision_not_preregistered": True,
    }


def verify_sealed(study, raw_metrics, plan):
    checks, means = {}, {}
    limits = plan["screening_gates"]["sealed_conditions_after_primary_pass"]
    for scheme in ("random", "topic"):
        means[scheme] = {}
        for condition in SEALED:
            rows = [r for r in raw_metrics[scheme] if r["condition"] == condition]
            complete = len(rows) == 3 and sorted(r["fold"] for r in rows) == [0, 1, 2]
            enough = complete and all(r["evidence_sufficient"] for r in rows)
            delta = [r["delta_auroc"] for r in rows]
            mean = float(np.mean(delta)) if enough else None
            minimum = min(delta) if enough else None
            checks[f"{scheme}_{condition}"] = bool(enough and mean >= limits["suffix_middle_minimum_each_mean_delta_in_each_scheme"]
                                                    and minimum >= limits["suffix_middle_minimum_any_fold_delta"])
            means[scheme][condition] = {"mean_delta": mean, "minimum_fold_delta": minimum}
    expected = all(checks.values())
    recorded = read_json(study / "sealed_gate.json")
    demand(recorded["passed"] == expected, "New-condition gate differs")
    return {"passed": expected, "checks": checks, "conditions": means}


def verify_sensitivity(study, train, topics):
    path = study / "topic_sensitivity.json"
    if not path.exists():
        return {"evaluated": False}
    saved = read_json(path)
    rows = saved if isinstance(saved, list) else saved["rows"]
    reviewed = {int(r["topic"]): r["decision"] for r in read_json(study / "topic_review.json")["combined"]}
    topic_by_id = {row["id"]: int(topic) for row, topic in zip(train, topics)}
    predictions = []
    for fold in range(3):
        folder = study / "folds" / f"topic_{fold}"
        for stem in ("primary", "sealed"):
            if (folder / f"{stem}_predictions.csv").exists():
                predictions.extend(read_csv(folder / f"{stem}_predictions.csv"))
    verified = []
    for row in rows:
        group = row.get("group", row.get("topic_group"))
        demand(group in ("content", "mixed_and_style"), "Unexpected sensitivity group")
        selected = [r for r in predictions if int(r["fold"]) == int(row["fold"])
                    and r["condition"] == row["condition"]
                    and ((reviewed[topic_by_id[r["id"]]] == "content") == (group == "content"))]
        y = np.array([int(r["label"]) for r in selected])
        demand(row["n"] == len(y), "Sensitivity population differs")
        for key, expected in (("n_human", np.sum(y == 0)), ("n_ai", np.sum(y == 1))):
            demand(row[key] == expected, "Sensitivity class counts differ")
        before = np.array([float(r["baseline_margin_not_probability"]) for r in selected])
        after = np.array([float(r["weak_crop_margin_not_probability"]) for r in selected])
        a = float(roc_auc_score(y, before)) if len(set(y)) == 2 else None
        b = float(roc_auc_score(y, after)) if len(set(y)) == 2 else None
        for key, expected in (("baseline_auroc", a), ("weak_crop_auroc", b), ("delta_auroc", b - a if a is not None else None)):
            close(row[key], expected, f"Sensitivity AUC differs: {key}")
        if "evidence_sufficient" in row:
            demand(row["evidence_sufficient"] == (min(np.sum(y == 0), np.sum(y == 1)) >= 50), "Sensitivity sufficiency differs")
        verified.append({"group": group, "fold": int(row["fold"]), "condition": row["condition"], "delta_auroc": None if a is None else b - a})
    if isinstance(saved, dict) and "summaries" in saved:
        for summary in saved["summaries"]:
            selected = [r for r in rows if r["group"] == summary["group"] and r["condition"] == summary["condition"]]
            defined = [r for r in selected if r["delta_auroc"] is not None]
            for key in ("n", "n_human", "n_ai"):
                demand(summary[key] == sum(r[key] for r in selected), "Sensitivity aggregate counts differ")
            demand(summary["defined_folds"] == [r["fold"] for r in defined], "Sensitivity defined-fold set differs")
            demand(summary["sufficient_folds"] == [r["fold"] for r in selected if r["evidence_sufficient"]],
                   "Sensitivity sufficient-fold set differs")
            close(summary["defined_fold_mean_delta"], np.mean([r["delta_auroc"] for r in defined]) if defined else None,
                  "Sensitivity aggregate delta differs")
    return {"evaluated": True, "verified_rows": len(rows), "rows": verified,
            "method": "Same saved topic-fold models and predictions, no refitting"}


def probabilities(model, texts):
    demand(np.array_equal(model.classes_, [0, 1]), "Class probability orientation differs")
    values = model.predict_proba(texts)[:, 1]
    demand(np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 1)), "Invalid probability")
    return values


def verify_calibration_fits(study, originals, arm, stage):
    rows = [r for r in read_json(study / "fit_receipts.json") if r["stage"] == stage and r["arm"] == arm]
    texts = [r["text"] for r in originals]
    labels = np.array([int(r["label"]) for r in originals])
    cv = StratifiedKFold(3, shuffle=True, random_state=242)
    expected = [[originals[i]["id"] for i in fit] for fit, _ in cv.split(texts, labels)] + [[r["id"] for r in originals]]
    demand(len(rows) == 4 and [r["fitting_ids"] for r in rows] == expected, "Calibration original-ID boundaries differ")
    for row, identifiers in zip(rows, expected):
        demand(row["n_originals"] == len(identifiers), "Calibration fitting size differs")
        weight = row["weight_receipt"]
        demand(weight["n_originals"] == len(identifiers), "Calibration weight receipt differs")
        close(weight["sample_weight_sum"], len(identifiers), "Calibration total weight changed", 1e-9)
        close(weight["minimum_parent_weight"], 1, "Calibration minimum original weight changed")
        close(weight["maximum_parent_weight"], 1, "Calibration maximum original weight changed")
    return {"base_fits": 4, "calibration_original_ids_verified": True}


def verify_oof_calibration(folder, originals, model):
    receipt = read_json(folder / "fit_complete.json")
    hashes = receipt["oof_artifact_sha256"]
    demand(set(hashes) == {f"calibration_oof_{i}.csv" for i in (1, 2, 3)}, "Missing calibration OOF artifacts")
    verify_hashes(folder, hashes)
    texts = [r["text"] for r in originals]
    y = np.array([int(r["label"]) for r in originals])
    margins = np.full(len(originals), np.nan)
    cv = StratifiedKFold(3, shuffle=True, random_state=242)
    for i, (_, held) in enumerate(cv.split(texts, y), 1):
        rows = read_csv(folder / f"calibration_oof_{i}.csv")
        demand([r["id"] for r in rows] == [originals[j]["id"] for j in held], "Calibration OOF original IDs differ")
        demand([int(r["label"]) for r in rows] == y[held].tolist(), "Calibration OOF labels differ")
        margins[held] = [float(r["margin_not_probability"]) for r in rows]
    demand(np.all(np.isfinite(margins)), "Nonfinite/missing calibration OOF margins")
    calibrator = model.calibrated_classifiers_[0].calibrators[0]
    p = expit(-(calibrator.a_ * margins + calibrator.b_))
    n0, n1 = np.sum(y == 0), np.sum(y == 1)
    target = np.where(y == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
    # A no-optimization numeric diagnostic at the saved sigmoid parameters.
    # The gradient is reported, not an additional scientific promotion gate.
    scale = np.max(np.abs(margins)) if np.max(np.abs(margins)) >= 30 else 1.0
    gradient = np.array([-np.dot(p - target, margins / scale), -np.sum(p - target)])
    return {"original_oof_n": len(originals), "three_original_id_folds_verified": True,
            "a": float(calibrator.a_), "b": float(calibrator.b_),
            "saved_sigmoid_scaled_gradient_max_abs": float(np.max(np.abs(gradient))),
            "numeric_calibrator_refit_performed": False}


def verify_calibrated_development(study, train, validation, plan):
    gate_path = study / "calibrated_gate.json"
    if not gate_path.exists():
        return {"evaluated": False}
    gate = read_json(gate_path)
    if gate.get("reason") == "insufficient_condition_class_counts":
        demand(not gate["passed"], "Insufficient classes recorded as passed")
        return {"evaluated": True, "passed": False, "reason": gate["reason"]}
    prefix = study.name.removesuffix("_study")
    models, model_checks = {}, {}
    for arm in ("baseline", "weak_crop"):
        folder = study.parent / f"{prefix}_{arm}_8000"
        receipt = read_json(folder / "fit_complete.json")
        demand(sha256(folder / "model.joblib") == receipt["model_sha256"], "Calibrated model hash differs")
        model = joblib.load(folder / "model.joblib")
        demand(model.method == "sigmoid" and model.ensemble is False and model.n_jobs == 1, "Calibration recipe differs")
        demand(model.cv.n_splits == 3 and model.cv.shuffle is True and model.cv.random_state == 242, "Calibration split recipe differs")
        base = model.calibrated_classifiers_[0].estimator
        demand(tuple(base.fitting_text_hashes_) == tuple(text_hash(r["text"]) for r in train), "Final calibrated fitting originals differ")
        model_checks[arm] = verify_calibration_fits(study, train, arm, 3)
        model_checks[arm]["idf"] = verify_idf(base.transformer_, [r["text"] for r in train])
        model_checks[arm]["oof_calibration"] = verify_oof_calibration(folder, train, model)
        models[arm] = model
    saved = read_csv(study / "calibrated_condition_predictions.csv")
    labels = np.array([int(r["label"]) for r in validation])
    values, rows = {}, []
    for condition in PRIMARY + SEALED:
        indices, texts = expected_texts(validation, condition)
        selected = [r for r in saved if r["condition"] == condition]
        demand([r["id"] for r in selected] == [validation[i]["id"] for i in indices], "Calibrated condition IDs differ")
        demand([int(r["label"]) for r in selected] == labels[indices].tolist(), "Calibrated labels differ")
        before, after = probabilities(models["baseline"], texts), probabilities(models["weak_crop"], texts)
        close(float(np.max(np.abs(before - [float(r["baseline_p_ai"]) for r in selected]))), 0, "Baseline calibrated probabilities differ")
        close(float(np.max(np.abs(after - [float(r["weak_crop_p_ai"]) for r in selected]))), 0, "Crop calibrated probabilities differ")
        y = labels[indices]
        a, b = float(roc_auc_score(y, before)), float(roc_auc_score(y, after))
        rows.append({"condition": condition, "n": len(y), "n_human": int(np.sum(y == 0)), "n_ai": int(np.sum(y == 1)),
                     "baseline_auroc": a, "weak_crop_auroc": b, "delta_auroc": b - a})
        values[condition] = (indices, before, after)
    recorded = {r["condition"]: r for r in gate["conditions"]}
    for row in rows:
        for key, expected in row.items():
            if key.endswith("auroc"):
                close(recorded[row["condition"]][key], expected, "Calibrated metric differs")
            else:
                demand(recorded[row["condition"]][key] == expected, "Calibrated metric population differs")
    reference = read_csv((ROOT / plan["reference_source_model"]).parent / "validation_predictions.csv")
    by_id = {r["id"]: float(r["p_ai"]) for r in reference}
    error = float(np.max(np.abs(values["clean"][1] - [by_id[r["id"]] for r in validation])))
    close(error, gate["baseline_reproduction_max_error"], "Calibrated baseline reproduction error differs")
    rng = np.random.default_rng(plan["seeds"]["validation_bootstrap"])
    samples = {name: [] for name in ("clean", "prefix_100")}
    for _ in range(plan["statistics"]["bootstrap_repetitions"]):
        draw = np.concatenate([rng.choice(np.flatnonzero(labels == c), int(np.sum(labels == c)), replace=True) for c in (0, 1)])
        for condition in samples:
            indices, before, after = values[condition]
            lookup = np.full(len(labels), -1, dtype=int)
            lookup[indices] = np.arange(len(indices))
            local = lookup[draw]
            local = local[local >= 0]
            y = labels[indices][local]
            if len(set(y)) == 2:
                samples[condition].append(float(roc_auc_score(y, after[local]) - roc_auc_score(y, before[local])))
    for name, draws in samples.items():
        recorded_interval = gate["intervals"][name]
        demand(recorded_interval["valid_replicates"] == len(draws), "Calibrated bootstrap valid count differs")
        low, high = np.quantile(draws, [.025, .975]) if len(draws) >= 900 else (None, None)
        close(low, recorded_interval["lower"], "Calibrated CI lower differs")
        close(high, recorded_interval["upper"], "Calibrated CI upper differs")
    rows_by_condition = {r["condition"]: r for r in rows}
    limits = plan["calibrated_development_gate"]
    bounds = np.quantile([len(r["text"].split()) for r in train], [1 / 3, 2 / 3])
    bins = np.digitize([len(r["text"].split()) for r in validation], bounds, right=True)
    length_rows = []
    for b in range(3):
        mask = bins == b
        y = labels[mask]
        enough = min(np.sum(y == 0), np.sum(y == 1)) >= 30
        difference = float(roc_auc_score(y, values["clean"][2][mask]) - roc_auc_score(y, values["clean"][1][mask])) if enough else None
        length_rows.append({"bin": b, "n": int(mask.sum()), "n_human": int(np.sum(y == 0)), "n_ai": int(np.sum(y == 1)), "delta_auroc": difference})
    for expected, actual in zip(length_rows, gate["length_bins"]):
        for key in expected:
            if key == "delta_auroc":
                close(expected[key], actual[key], "Length-bin delta differs")
            else:
                demand(expected[key] == actual[key], "Length-bin counts differ")
    checks = {
        "baseline_reproduction": error <= limits["baseline_max_probability_reproduction_error"],
        "clean_point": rows_by_condition["clean"]["delta_auroc"] >= limits["clean_minimum_point_delta"],
        "clean_interval": gate["intervals"]["clean"]["lower"] is not None and gate["intervals"]["clean"]["lower"] >= limits["clean_minimum_paired_95_interval_lower"],
        "prefix_point": rows_by_condition["prefix_100"]["delta_auroc"] >= limits["prefix_minimum_point_delta"],
        "prefix_interval": gate["intervals"]["prefix_100"]["lower"] is not None and gate["intervals"]["prefix_100"]["lower"] > 0,
        "other_conditions": all(r["delta_auroc"] >= limits["other_conditions_minimum_each_point_delta"] for r in rows if r["condition"] not in ("clean", "prefix_100")),
        "condition_class_counts": all(min(r["n_human"], r["n_ai"]) >= 50 for r in rows),
        "length_bins": all(r["delta_auroc"] is not None and r["delta_auroc"] >= -.01 for r in length_rows),
    }
    demand(gate["checks"] == checks and gate["passed"] == all(checks.values()), "Calibrated gate differs")
    for arm, column in (("baseline", 1), ("weak_crop", 2)):
        folder = study.parent / f"{prefix}_{arm}_8000"
        held = read_csv(folder / "validation_predictions.csv")
        demand([r["id"] for r in held] == [r["id"] for r in validation], "Canonical probability IDs differ")
        p = values["clean"][column]
        close(float(np.max(np.abs(p - [float(r["p_ai"]) for r in held]))), 0, "Canonical probabilities differ")
        metrics = read_json(folder / "metrics.json")["validation"]
        close(metrics["auroc"], roc_auc_score(labels, p), "Canonical AUC differs")
        close(metrics["log_loss"], log_loss(labels, p), "Canonical log loss differs")
        close(metrics["brier"], brier_score_loss(labels, p), "Canonical Brier differs")
    close(gate["probability_spearman"], spearmanr(values["clean"][1], values["clean"][2]).statistic, "Calibrated rank correlation differs")
    return {"evaluated": True, "passed": all(checks.values()), "checks": checks, "models": model_checks,
            "conditions": rows, "intervals": gate["intervals"], "baseline_reproduction_error": error}


def verify(study):
    study = Path(study).resolve()
    completion = read_json(study / "completion.json")
    plan, provenance = read_json(study / "plan.json"), read_json(study / "provenance.json")
    demand(plan["protocol_version"] >= 2, "Use this verifier only for the explicitly revised protocol")
    verify_hashes(ROOT, plan["expected_reference_sha256"])
    verify_hashes(study / "source", provenance["source_sha256"])
    source_plan = study / "source/experiments/supervised/q5_crop_revised_plan.json"
    demand(sha256(source_plan) == provenance["plan_sha256"], "Revised source plan hash differs")
    demand(read_json(source_plan) == plan, "Revised source/runtime plans differ")
    original_plan = read_json(parent_path(plan) / "plan.json")
    for key in ("arms", "predictive_representation", "seeds", "screening_gates"):
        demand(plan[key] == original_plan[key], f"An unrelated scientific setting changed: {key}")
    old_calibration = dict(original_plan["calibrated_development_gate"])
    new_calibration = dict(plan["calibrated_development_gate"])
    old_calibration.pop("run_only_after")
    new_calibration.pop("run_only_after")
    demand(old_calibration == new_calibration, "Calibrated predictive thresholds changed")
    train, validation = load_partitions()
    topics, random_folds, topic_folds, quality, topic_receipt = verify_reused_topics(study, train, plan)
    reference = read_json(study / "source" / plan["reference_plan"])
    raw_summary, raw_metrics = verify_raw(study, train, plan, reference,
                                         {"random": random_folds, "topic": topic_folds}, topics)
    sealed = verify_sealed(study, raw_metrics, plan) if (study / "sealed_gate.json").exists() else {"evaluated": False}
    sensitivity = verify_sensitivity(study, train, topics)
    calibrated = verify_calibrated_development(study, train, validation, plan)
    fits = read_json(study / "fit_receipts.json")
    demand([r["fit_number"] for r in fits] == list(range(1, len(fits) + 1)), "Incomplete fit numbering")
    demand(len(fits) <= plan["stages_and_svc_fit_budget"]["maximum_svc_fits"], "Real-fit budget exceeded")
    demand(all(r["state"] == "completed" for r in fits), "A failed fit was counted as completed evidence")
    demand(all(not any(w["category"] == "ConvergenceWarning" for w in r["warnings"]) for r in fits), "Convergence warning invalidates evidence")
    demand(sum(r["stage"] == 1 for r in fits) == 6, "Random-stage actual fit count differs")
    demand(sum(r["stage"] == 2 for r in fits) == (6 if "topic" in raw_summary else 0), "Topic-stage actual fit count differs")
    demand(completion["competition_svc_fits"] == len(fits), "Completion fit count differs")
    demand(completion["uploads"] == 0 and completion["incumbent_retained"] is True, "Unexpected upload/incumbent change")
    prefix = study.name.removesuffix("_study")
    candidate = ROOT / "submissions" / f"{prefix}_weak_crop_full10000.csv"
    reason = completion.get("reason")
    counts = {"random_screen_failed": 6, "topic_screen_failed": 12, "sealed_conditions_failed": 12, "calibrated_development_failed": 20}
    if completion["outcome"] == "stopped_no_export":
        demand(reason in counts and len(fits) == counts[reason], "Stop reason/fit count differs")
        demand(completion["new_candidates"] == 0 and completion["test_text_accessed"] is False and not candidate.exists(),
               "Stopped study accessed test/export")
        if reason == "random_screen_failed":
            demand(not raw_summary["random"]["gate_passed"] and "topic" not in raw_summary, "Random stop inconsistent")
        elif reason == "topic_screen_failed":
            demand(raw_summary["random"]["gate_passed"] and not raw_summary["topic"]["gate_passed"], "Topic stop inconsistent")
        else:
            demand(raw_summary["random"]["gate_passed"] and raw_summary["topic"]["gate_passed"], "Later stage skipped primary gates")
            if reason == "sealed_conditions_failed":
                demand(sealed["passed"] is False and not calibrated["evaluated"], "Sealed stop inconsistent")
            else:
                demand(sealed["passed"] and calibrated["passed"] is False, "Calibrated stop inconsistent")
    else:
        demand(completion["outcome"] == "eligible_candidate_exported" and len(fits) == 24, "Unexpected eligible outcome")
        demand(calibrated["passed"] and sealed["passed"], "Ineligible export")
        demand(completion["new_candidates"] == 1 and candidate.exists(), "Missing single candidate")
        from human_ai.check_predictions import check
        check(candidate, "submission")
        metadata = read_json(candidate.with_suffix(".metadata.json"))
        demand(metadata["csv_sha256"] == sha256(candidate) and metadata["heldout_score"] is None, "Candidate metadata differs")
        all_originals = read_csv(raw_path("train.csv"))
        verify_calibration_fits(study, all_originals, "weak_crop", 4)
        model_path = study.parent / f"{prefix}_weak_crop_full10000/model.joblib"
        demand(metadata["model_sha256"] == sha256(model_path), "Final model hash differs")
        model = joblib.load(model_path)
        verify_oof_calibration(model_path.parent, all_originals, model)
        test = read_csv(raw_path("test.csv"))
        values = probabilities(model, [r["text"] for r in test])
        submitted = {r["id"]: float(r["label"]) for r in read_csv(candidate)}
        demand(np.max(np.abs(values - [submitted[r["id"]] for r in test])) <= 1e-12, "Final model/CSV reload differs")
    return {"passed": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "study": str(study.relative_to(ROOT)), "outcome": completion["outcome"], "reason": reason,
            "competition_svc_fits": len(fits), "models_fitted_by_verifier": 0,
            "test_text_read_by_verifier": completion["outcome"] == "eligible_candidate_exported",
            "verifier_source_sha256": sha256(__file__),
            "helper_source_sha256": sha256(ROOT / "experiments/supervised/verify_q5_crop.py"),
            "source_snapshot_hashes_checked": len(provenance["source_sha256"]),
            "topic_reuse": topic_receipt, "raw_stages": raw_summary, "sealed_conditions": sealed,
            "topic_sensitivity": sensitivity, "calibrated_development": calibrated,
            "limitations": ["This protocol was revised after the first run's grouping diagnostics and random-screen results.",
                            "The unchanged clusters are lexical topic proxies, not true source/generator labels.",
                            "Historical review blindness is attested; no new blind review is claimed.",
                            "Conditional bootstrap omits training and adaptive-selection variation; no model is refitted by this verifier."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    demand(not args.output.exists(), "Use a new independent verification output path")
    with threadpool_limits(limits=1):
        result = verify(args.study)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "outcome", "reason", "competition_svc_fits", "models_fitted_by_verifier")}))


if __name__ == "__main__":
    main()
