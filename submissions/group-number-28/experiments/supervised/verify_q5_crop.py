"""Independently verify a completed weak-crop study without fitting a model.

This verifier reconstructs AUCs with sklearn, grouping quality and fold counts
from canonical IDs, weight conservation from original documents, source/model
hashes and the stage-stop decision. Topic/condition/gate runner helpers are not
imported. Bootstrap reproduction uses sklearn's weighted AUC, independently of
the runner's cached-AUC implementation. No test text is read unless the recorded
study is eligible and actually exported its single allowed candidate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from human_ai.data import ROOT, load_partitions, raw_path, read_csv, sha256


PRIMARY = ("clean", "quote_and_dash", "digit_runs_to_zero", "whitespace_only", "prefix_100")
SEALED = ("suffix_100", "middle_100")
QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'",
                          "“": '"', "”": '"', "„": '"', "‟": '"'})
DASH_MAP = str.maketrans({"–": "-", "—": "-"})


def read_json(path):
    return json.loads(Path(path).read_text())


def demand(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, message, tolerance=1e-12):
    if a is None or b is None:
        demand(a is None and b is None, message)
    else:
        demand(abs(float(a) - float(b)) <= tolerance, message)


def verify_hashes(base, expected):
    for name, digest in expected.items():
        demand((base / name).is_file(), f"Missing hashed file: {base / name}")
        demand(sha256(base / name) == digest, f"Hash mismatch: {base / name}")


def assignments(path, canonical_ids):
    rows = read_csv(path)
    demand([row["id"] for row in rows] == canonical_ids, f"Noncanonical ID order: {path}")
    values = np.array([int(row["held_fold"]) for row in rows])
    demand(set(values.tolist()) == {0, 1, 2}, f"Missing or unexpected folds: {path}")
    return values


def expected_texts(held, condition):
    indices = np.arange(len(held), dtype=int)
    original = [row["text"] for row in held]
    words = [text.split() for text in original]
    if condition in ("prefix_100", "suffix_100", "middle_100"):
        indices = np.array([i for i, tokens in enumerate(words) if len(tokens) >= 150], dtype=int)
        if condition == "prefix_100":
            texts = [" ".join(words[i][:100]) for i in indices]
        elif condition == "suffix_100":
            texts = [" ".join(words[i][-100:]) for i in indices]
        else:
            texts = [" ".join(words[i][(len(words[i]) - 100) // 2:(len(words[i]) - 100) // 2 + 100]) for i in indices]
    elif condition == "clean":
        texts = original
    elif condition == "quote_and_dash":
        texts = [text.translate(QUOTE_MAP).translate(DASH_MAP) for text in original]
    elif condition == "digit_runs_to_zero":
        texts = [re.sub(r"\d+", "0", text) for text in original]
    elif condition == "whitespace_only":
        texts = [" ".join(tokens) for tokens in words]
    else:
        raise AssertionError(f"Unknown condition: {condition}")
    return indices, texts


def verify_topics(study, train, plan):
    ids = [row["id"] for row in train]
    labels = np.array([int(row["label"]) for row in train])
    rows = read_csv(study / "topic_assignments.csv")
    demand([row["id"] for row in rows] == ids, "Topic IDs are not the canonical fitting originals")
    topics = np.array([int(row["topic"]) for row in rows])
    n_topics = plan["topic_proxy"]["kmeans"]["n_clusters"]
    demand(np.all((topics >= 0) & (topics < n_topics)), "Topic index out of range")
    preparation, finalization = read_json(study / "topic_preparation.json"), read_json(study / "topic_finalization.json")
    verify_hashes(study, preparation["artifact_sha256"])
    verify_hashes(study, finalization["artifact_sha256"])
    demand(preparation["record_labels_read"] is False and preparation["model_score_computed"] is False,
           "Preparation does not attest label/score blindness")
    packet, frozen = read_json(study / "topic_review_packet.json"), read_json(study / "topic_review.json")
    demand(frozen["labels_read_before_review_freeze"] is False, "Labels were not held until review freeze")
    demand(frozen["topic_review_packet_sha256"] == sha256(study / "topic_review_packet.json"), "Changed blind packet")
    a, b = frozen["review_a"], frozen["review_b"]
    demand(a["reviewer"] != b["reviewer"], "Reviewers are not distinct")
    combined = []
    for review in (a, b):
        demand(review["reviewed_without_labels_or_scores"] is True, "Unblinded topic review")
        demand(len(review["clusters"]) == n_topics, "Incomplete topic review")
        reviewed = {int(row["topic"]): row for row in review["clusters"]}
        demand(set(reviewed) == set(range(n_topics)), "Duplicate/missing topic review")
        for item in packet["clusters"]:
            demand(set(reviewed[item["topic"]]["inspected_ids"]) == set(item["sample_ids"]), "Wrong blind sample IDs")
    by_a = {int(row["topic"]): row for row in a["clusters"]}
    by_b = {int(row["topic"]): row for row in b["clusters"]}
    for topic in range(n_topics):
        decision = by_a[topic]["decision"] if by_a[topic]["decision"] == by_b[topic]["decision"] else "mixed"
        combined.append({"topic": topic, "decision": decision})
    demand(combined == frozen["combined"], "Combined review violates disagreement rule")
    q = plan["topic_proxy"]["quality_gates"]
    counts = np.bincount(topics, minlength=n_topics)
    dominant = [max(np.sum(labels[topics == topic] == c) for c in (0, 1)) / counts[topic]
                if counts[topic] else None for topic in range(n_topics)]
    n_content = sum(row["decision"] == "content" for row in combined)
    n_style = sum(row["decision"] == "style-format" for row in combined)
    nmi = float(normalized_mutual_info_score(topics, labels, average_method=q["nmi_average_method"]))
    checks = {
        "all_clusters_nonempty_and_large_enough": bool(np.all(counts >= q["minimum_cluster_n"])),
        "maximum_cluster_size": bool(np.all(counts <= q["maximum_cluster_n"])),
        "minimum_interpretable_content_topics": n_content >= q["minimum_content_topic_clusters"],
        "maximum_style_format_topics": n_style <= q["maximum_style_or_format_clusters"],
        "topic_label_nmi": nmi <= q["maximum_topic_label_nmi_inclusive"],
        "maximum_cluster_dominant_class_fraction": all(value is not None and value < q["maximum_cluster_dominant_class_fraction_exclusive"] for value in dominant),
    }
    topic_folds = None
    if (study / "topic_folds.csv").exists():
        topic_folds = assignments(study / "topic_folds.csv", ids)
        splitter_settings = {k: v for k, v in plan["topic_proxy"]["folds"].items() if k != "type"}
        expected = np.full(len(train), -1)
        for fold, (_, held) in enumerate(StratifiedGroupKFold(**splitter_settings).split(np.zeros(len(train)), labels, topics)):
            expected[held] = fold
        demand(np.array_equal(expected, topic_folds), "Topic folds differ from prescribed fixed splitter")
        for topic in set(topics.tolist()):
            demand(len(set(topic_folds[topics == topic].tolist())) == 1, "A topic crosses held folds")
        bounds = plan["topic_proxy"]["fold_quality_gates"]
        valid_folds = []
        for fold in range(3):
            y = labels[topic_folds == fold]
            valid_folds.append(bounds["minimum_held_n"] <= len(y) <= bounds["maximum_held_n"]
                               and min(np.sum(y == 0), np.sum(y == 1)) >= bounds["minimum_held_n_per_class"]
                               and bounds["minimum_held_ai_fraction"] <= np.mean(y) <= bounds["maximum_held_ai_fraction"])
        checks["topic_fold_balance_and_size"] = all(valid_folds)
    else:
        checks["topic_fold_balance_and_size"] = False
    quality = read_json(study / "topic_quality.json")
    demand(quality["checks"] == checks, "Recorded topic quality checks differ from independent calculations")
    demand(quality["topic_valid"] == all(checks.values()), "Topic-valid flag is inconsistent")
    close(quality["topic_label_nmi"], nmi, "Topic/label NMI mismatch")
    demand(quality["n_content_topics"] == n_content and quality["n_style_format_topics"] == n_style, "Review count mismatch")
    random_folds = assignments(study / "random_folds.csv", ids)
    expected = np.full(len(train), -1)
    if quality["topic_valid"]:
        rng = np.random.default_rng(plan["seeds"]["matched_random_folds"])
        for value in (0, 1):
            permutation = rng.permutation(np.flatnonzero(labels == value))
            offset = 0
            for fold in range(3):
                n = int(np.sum((labels == value) & (topic_folds == fold)))
                expected[permutation[offset:offset + n]] = fold
                offset += n
                demand(np.sum((random_folds == fold) & (labels == value)) == n, "Random/topic class counts differ")
    else:
        splitter = StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["fallback_random_folds"])
        for fold, (_, held) in enumerate(splitter.split(np.zeros(len(train)), labels)):
            expected[held] = fold
    demand(np.array_equal(random_folds, expected), "Random folds differ from matched/fallback specification")
    for row in read_csv(study / "fold_counts.csv"):
        assignment = topic_folds if row["scheme"] == "topic" else random_folds
        held = assignment == int(row["fold"])
        for key, expected_count in (("held_n", held.sum()), ("held_human", np.sum(held & (labels == 0))),
                                    ("held_ai", np.sum(held & (labels == 1))), ("fitting_n", (~held).sum()),
                                    ("fitting_human", np.sum(~held & (labels == 0))), ("fitting_ai", np.sum(~held & (labels == 1)))):
            demand(int(row[key]) == expected_count, f"Fold count mismatch: {key}")
    return topics, random_folds, topic_folds, quality


def vectorizers(features):
    if isinstance(features, TfidfVectorizer):
        return [features]
    children = getattr(features, "steps", getattr(features, "transformer_list", []))
    return [leaf for _, child in children for leaf in vectorizers(child)]


def verify_idf(features, originals):
    """Recompute fitted-vocabulary document frequencies without a fit call."""
    receipts = []
    for vectorizer in vectorizers(features):
        vocabulary = vectorizer.vocabulary_
        dfs = np.zeros(len(vocabulary), dtype=np.int64)
        analyzer = vectorizer.build_analyzer()
        for text in originals:
            for token in set(analyzer(text)):
                position = vocabulary.get(token)
                if position is not None:
                    dfs[position] += 1
        smooth = int(vectorizer.smooth_idf)
        expected = np.log((len(originals) + smooth) / (dfs + smooth)) + 1
        error = float(np.max(np.abs(expected - vectorizer.idf_)))
        demand(np.all(dfs >= vectorizer.min_df), "A fitted feature lacks the training-only minimum document frequency")
        demand(error <= 1e-12, "IDF does not match fitting originals only")
        receipts.append({"analyzer": vectorizer.analyzer, "n_features": len(vocabulary),
                         "n_fitting_originals": len(originals), "idf_max_error": error})
    demand(len(receipts) == 2 and {row["analyzer"] for row in receipts} == {"word", "char"}, "Unexpected WC120 blocks")
    return receipts


def independently_gate(rows, scheme, plan):
    limits = plan["screening_gates"][scheme]
    summaries = {}
    complete = True
    for condition in PRIMARY:
        group = [row for row in rows if row["condition"] == condition]
        sufficient = len(group) == 3 and sorted(row["fold"] for row in group) == [0, 1, 2] and all(row["evidence_sufficient"] for row in group)
        complete = complete and sufficient
        deltas = [row["delta_auroc"] for row in group]
        summaries[condition] = {"mean": float(np.mean(deltas)) if sufficient else None,
                                "minimum": float(min(deltas)) if sufficient else None,
                                "positive": sum(value > 0 for value in deltas) if sufficient else None}
    if not complete:
        return False, summaries
    good = summaries["clean"]["mean"] >= limits["clean_minimum_mean_delta"] and summaries["clean"]["minimum"] >= limits["clean_minimum_any_fold_delta"]
    good = good and summaries["prefix_100"]["mean"] >= limits["prefix_minimum_mean_delta"] and summaries["prefix_100"]["positive"] >= limits["prefix_minimum_positive_folds"]
    if "prefix_minimum_any_fold_delta" in limits:
        good = good and summaries["prefix_100"]["minimum"] >= limits["prefix_minimum_any_fold_delta"]
    for condition in ("quote_and_dash", "digit_runs_to_zero", "whitespace_only"):
        good = good and summaries[condition]["mean"] >= limits["quote_digit_whitespace_minimum_each_mean_delta"]
        good = good and summaries[condition]["minimum"] >= limits["quote_digit_whitespace_minimum_any_fold_delta"]
    return bool(good), summaries


def verify_predictions(folder, held, features, baseline, weak, scheme, fold, sealed=False):
    stem, names = ("sealed", SEALED) if sealed else ("primary", PRIMARY)
    receipt = read_json(folder / f"{stem}_receipt.json")
    demand(sha256(folder / f"{stem}_predictions.csv") == receipt["prediction_sha256"], "Prediction checksum mismatch")
    demand(sha256(folder / f"{stem}_metrics.json") == receipt["metrics_sha256"], "Metric checksum mismatch")
    saved = read_csv(folder / f"{stem}_predictions.csv")
    demand(set(row["condition"] for row in saved) <= set(names), "Unexpected revealed condition")
    labels = np.array([int(row["label"]) for row in held])
    metrics, record = [], {"fold": fold, "labels": labels, "conditions": {}}
    for name in names:
        indices, texts = expected_texts(held, name)
        rows = [row for row in saved if row["condition"] == name]
        demand([row["id"] for row in rows] == [held[i]["id"] for i in indices], "Condition prediction ID/order mismatch")
        demand(all(int(row["label"]) == labels[i] for row, i in zip(rows, indices)), "Condition labels changed")
        demand(all(row["scheme"] == scheme and int(row["fold"]) == fold for row in rows), "Condition fold metadata mismatch")
        before = np.array([float(row["baseline_margin_not_probability"]) for row in rows])
        after = np.array([float(row["weak_crop_margin_not_probability"]) for row in rows])
        demand(np.all(np.isfinite(before)) and np.all(np.isfinite(after)), "Nonfinite margin")
        if len(indices):
            matrix = features.transform(texts)
            demand(np.max(np.abs(baseline.decision_function(matrix) - before)) <= 1e-12, "Baseline saved margin/model mismatch")
            demand(np.max(np.abs(weak.decision_function(matrix) - after)) <= 1e-12, "Crop saved margin/model mismatch")
        y = labels[indices]
        n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
        a = float(roc_auc_score(y, before)) if n0 and n1 else None
        b = float(roc_auc_score(y, after)) if n0 and n1 else None
        metrics.append({"scheme": scheme, "fold": fold, "condition": name, "n": len(indices),
                        "n_human": n0, "n_ai": n1, "baseline_auroc": a, "weak_crop_auroc": b,
                        "delta_auroc": b - a if a is not None else None, "evidence_sufficient": min(n0, n1) >= 50})
        record["conditions"][name] = (indices, before, after)
    recorded = read_json(folder / f"{stem}_metrics.json")
    demand(len(recorded) == len(metrics), "Wrong metric row count")
    for expected, actual in zip(metrics, recorded):
        for key in expected:
            if key.endswith("auroc"):
                close(expected[key], actual[key], f"Reconstructed AUC mismatch: {key}")
            else:
                demand(expected[key] == actual[key], f"Reconstructed metric mismatch: {key}")
    return metrics, record


def verify_bootstrap(records, scheme, recorded, plan):
    """Reproduce bootstrap using sklearn weighted AUC, not cached runner math."""
    seed = plan["seeds"]["topic_bootstrap" if scheme == "topic" else "random_bootstrap"]
    rng = np.random.default_rng(seed)
    repetitions = plan["statistics"]["bootstrap_repetitions"]
    deltas = {condition: [] for condition in PRIMARY}
    for _ in range(repetitions):
        fold_deltas = {condition: [] for condition in PRIMARY}
        for record in records:
            labels = record["labels"]
            if scheme == "topic":
                unique, inverse = np.unique(record["topics"], return_inverse=True)
                sampled = rng.integers(0, len(unique), size=len(unique))
                weights = np.bincount(sampled, minlength=len(unique))[inverse]
            else:
                draws = np.concatenate([rng.choice(np.flatnonzero(labels == c), int(np.sum(labels == c)), replace=True) for c in (0, 1)])
                weights = np.bincount(draws, minlength=len(labels))
            for condition in PRIMARY:
                indices, before, after = record["conditions"][condition]
                y, w = labels[indices], weights[indices]
                if w[y == 0].sum() == 0 or w[y == 1].sum() == 0:
                    fold_deltas[condition].append(None)
                else:
                    fold_deltas[condition].append(float(roc_auc_score(y, after, sample_weight=w) - roc_auc_score(y, before, sample_weight=w)))
        for condition in PRIMARY:
            if all(value is not None for value in fold_deltas[condition]):
                deltas[condition].append(float(np.mean(fold_deltas[condition])))
    for condition in PRIMARY:
        values, saved = deltas[condition], recorded["conditions"][condition]
        demand(saved["valid_replicates"] == len(values) and saved["invalid_replicates"] == repetitions - len(values), "Bootstrap valid counts differ")
        available = bool(values and len(values) / repetitions >= .9)
        demand(saved["interval_available"] == available, "Bootstrap availability differs")
        low, high = np.quantile(values, [.025, .975]).tolist() if available else (None, None)
        close(low, saved["paired_delta_ci_2_5"], "Bootstrap lower differs")
        close(high, saved["paired_delta_ci_97_5"], "Bootstrap upper differs")
    return {condition: {"valid_replicates": len(values), "verified": True} for condition, values in deltas.items()}


def verify_raw(study, train, plan, reference, assignments_by_scheme, topics):
    summary, metrics_by_scheme = {}, {}
    for scheme, assignment in assignments_by_scheme.items():
        if not (study / f"{scheme}_gate.json").exists():
            continue
        records, metrics, feature_receipts = [], [], []
        for fold in range(3):
            folder = study / "folds" / f"{scheme}_{fold}"
            fitting = [row for i, row in enumerate(train) if assignment[i] != fold]
            held = [row for i, row in enumerate(train) if assignment[i] == fold]
            receipt = read_json(folder / "feature_fit.json")
            demand(receipt["fitting_ids"] == [row["id"] for row in fitting], "Fold feature fitting ID mismatch")
            demand(receipt["held_ids"] == [row["id"] for row in held], "Fold held ID mismatch")
            demand(set(receipt["fitting_ids"]).isdisjoint(receipt["held_ids"]), "Fit/held overlap")
            demand(receipt["model_sha256"] == sha256(folder / "features.joblib"), "Feature model hash mismatch")
            features = joblib.load(folder / "features.joblib")
            feature_receipts.extend(verify_idf(features, [row["text"] for row in fitting]))
            classifiers = {}
            n_crop = sum(len(row["text"].split()) >= 150 for row in fitting)
            for arm in ("baseline", "weak_crop"):
                recorded = read_json(folder / f"{arm}_fit.json")
                demand(recorded["model_sha256"] == sha256(folder / f"{arm}.joblib"), "Classifier hash mismatch")
                demand(recorded["n_originals"] == len(fitting), "Classifier original count mismatch")
                expected_rows = len(fitting) + (n_crop if arm == "weak_crop" else 0)
                demand(recorded["n_training_rows"] == expected_rows, "Classifier crop count mismatch")
                close(recorded["sample_weight_sum"], len(fitting), "Original-ID total training weight changed", 1e-9)
                if arm == "weak_crop":
                    parent_weights = np.ones(len(fitting))
                    eligible = np.array([i for i, row in enumerate(fitting) if len(row["text"].split()) >= 150], dtype=int)
                    parent_weights[eligible] = .9 + .1
                    demand(np.all(parent_weights == 1), "Per-original weight is not one")
                classifier = joblib.load(folder / f"{arm}.joblib")
                for parameter, value in reference["classifier"].items():
                    demand(classifier.get_params()[parameter] == value, "Classifier hyperparameter changed")
                demand(classifier.random_state == reference["seeds"]["model"], "Classifier seed changed")
                classifiers[arm] = classifier
            rows, record = verify_predictions(folder, held, features, classifiers["baseline"], classifiers["weak_crop"], scheme, fold)
            if scheme == "topic":
                record["topics"] = topics[assignment == fold]
            metrics.extend(rows)
            records.append(record)
            if (folder / "sealed_predictions.csv").exists():
                sealed_rows, sealed_record = verify_predictions(folder, held, features, classifiers["baseline"], classifiers["weak_crop"], scheme, fold, True)
                metrics.extend(sealed_rows)
                record["conditions"].update(sealed_record["conditions"])
        primary_metrics = [row for row in metrics if row["condition"] in PRIMARY]
        expected_gate, means = independently_gate(primary_metrics, scheme, plan)
        gate = read_json(study / f"{scheme}_gate.json")
        demand(gate["passed"] == expected_gate, "Recorded screening gate differs")
        for condition in PRIMARY:
            close(gate["conditions"][condition]["mean_delta"], means[condition]["mean"], "Gate macro mean differs")
        summary[scheme] = {"gate_passed": expected_gate, "means": means, "idf_checks": feature_receipts,
                           "bootstrap": verify_bootstrap(records, scheme, read_json(study / f"{scheme}_bootstrap.json"), plan)}
        metrics_by_scheme[scheme] = metrics
    return summary, metrics_by_scheme


def verify(study):
    study = Path(study).resolve()
    completion = read_json(study / "completion.json")
    plan, provenance = read_json(study / "plan.json"), read_json(study / "provenance.json")
    verify_hashes(ROOT, plan["expected_reference_sha256"])
    verify_hashes(study / "source", provenance["source_sha256"])
    # Atomic runtime JSON is canonicalized/sorted; provenance pins the exact
    # source-plan bytes, while semantic equality verifies the runtime copy.
    demand(sha256(study / "source/experiments/supervised/q5_crop_plan.json") == provenance["plan_sha256"], "Source plan hash differs")
    demand(read_json(study / "source/experiments/supervised/q5_crop_plan.json") == plan, "Plan/source snapshot differ")
    demand(provenance["uploads_performed"] == 0, "Unexpected upload")
    demand(provenance["synthetic_checks"]["svc_fits"] <= 8 and provenance["synthetic_checks"]["maximum_originals_per_fit"] <= 30, "Synthetic fit budget exceeded")
    train, validation = load_partitions()
    topics, random_folds, topic_folds, quality = verify_topics(study, train, plan)
    reference = read_json(study / "source" / plan["reference_plan"])
    assignments_by_scheme = {"random": random_folds}
    if topic_folds is not None:
        assignments_by_scheme["topic"] = topic_folds
    raw_summary, raw_metrics = verify_raw(study, train, plan, reference, assignments_by_scheme, topics)
    fits = read_json(study / "fit_receipts.json")
    demand(len(fits) <= plan["stages_and_svc_fit_budget"]["maximum_svc_fits"], "Competition SVC budget exceeded")
    demand([row["fit_number"] for row in fits] == list(range(1, len(fits) + 1)), "Fit numbering is not complete")
    demand(all(row["state"] == "completed" for row in fits), "A failed/invalid fit exists in completed experiment")
    demand(all(not any(note["category"] == "ConvergenceWarning" for note in row["warnings"]) for row in fits), "Invalid converged fit was counted as evidence")
    demand(completion["competition_svc_fits"] == len(fits), "Terminal fit count differs")
    demand(completion["uploads"] == 0 and completion["incumbent_retained"] is True, "Incumbent/upload boundary changed")
    prefix = study.name.removesuffix("_study")
    candidate_path = ROOT / "submissions" / f"{prefix}_weak_crop_full10000.csv"
    reason = completion.get("reason")
    expected_counts = {"topic_proxy_invalid_diagnostic_random_only": 6, "random_screen_failed": 6,
                       "topic_screen_failed": 12, "sealed_conditions_failed": 12, "calibrated_development_failed": 20}
    if completion["outcome"] == "stopped_no_export":
        demand(reason in expected_counts and len(fits) == expected_counts[reason], "Terminal stage/count mismatch")
        demand(completion["new_candidates"] == 0 and completion["test_text_accessed"] is False, "A stopped study accessed test/export")
        demand(not candidate_path.exists(), "A stopped study exported a candidate")
        if reason == "topic_proxy_invalid_diagnostic_random_only":
            demand(not quality["topic_valid"], "Invalid-group stop with valid groups")
            demand(all(row["stage"] == 1 for row in fits), "Later stage fitted despite invalid grouping")
            demand(not (study / "topic_gate.json").exists() and not (study / "sealed_gate.json").exists(), "Later stage evaluated despite invalid grouping")
            demand(not any((study.parent / f"{prefix}_{arm}_8000").exists() for arm in ("baseline", "weak_crop")), "Development model fitted despite invalid grouping")
        elif reason == "random_screen_failed":
            demand(quality["topic_valid"] and not raw_summary["random"]["gate_passed"], "Random-stop reason inconsistent")
        elif reason == "topic_screen_failed":
            demand(raw_summary["random"]["gate_passed"] and not raw_summary["topic"]["gate_passed"], "Topic-stop reason inconsistent")
        elif reason in ("sealed_conditions_failed", "calibrated_development_failed"):
            demand(raw_summary["random"]["gate_passed"] and raw_summary["topic"]["gate_passed"], "Later stage started before primary gates passed")
    else:
        demand(completion["outcome"] == "eligible_candidate_exported" and len(fits) == 24, "Unexpected outcome or fit count")
        demand(completion["new_candidates"] == 1 and candidate_path.exists(), "Missing eligible candidate")
        demand(read_json(study / "calibrated_gate.json")["passed"] is True, "Ineligible export")
        from human_ai.check_predictions import check
        check(candidate_path, "submission")  # Eligible export is the only test-text access path.
        metadata = read_json(candidate_path.with_suffix(".metadata.json"))
        demand(metadata["csv_sha256"] == sha256(candidate_path) and metadata["heldout_score"] is None, "Candidate metadata mismatch")
        model_path = study.parent / f"{prefix}_weak_crop_full10000/model.joblib"
        demand(metadata["model_sha256"] == sha256(model_path), "Candidate model hash mismatch")
        test = read_csv(raw_path("test.csv"))
        model = joblib.load(model_path)
        probabilities = model.predict_proba([row["text"] for row in test])[:, list(model.classes_).index(1)]
        submitted = {row["id"]: float(row["label"]) for row in read_csv(candidate_path)}
        demand(np.max(np.abs(probabilities - [submitted[row["id"]] for row in test])) <= 1e-12, "Reloaded candidate/CSV mismatch")
    return {"passed": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "study": str(study.relative_to(ROOT)), "outcome": completion["outcome"], "reason": reason,
            "competition_svc_fits": len(fits), "models_fitted_by_verifier": 0,
            "test_text_read_by_verifier": completion["outcome"] == "eligible_candidate_exported",
            "source_snapshot_hashes_checked": len(provenance["source_sha256"]),
            "topic_valid": quality["topic_valid"], "topic_quality_checks": quality["checks"],
            "raw_stages": raw_summary,
            "limitations": ["Review blindness is verified from frozen attestations and source control flow, not independently observable reviewer cognition.",
                            "No model is refitted; source/IDF/weight receipts and reloaded predictions verify the recorded training boundary without redoing optimization.",
                            "Conditional bootstrap excludes training variation and adaptive-selection correction."]}


def verify_whitespace_diagnostic(study):
    """Check the observed whitespace effect without fitting or new conditions."""
    study = Path(study).resolve()
    demand((study / "completion.json").exists(), "Verify only after the study completes")
    train, _ = load_partitions()
    assignment = assignments(study / "random_folds.csv", [row["id"] for row in train])
    metrics = read_json(study / "random_metrics.json")
    means = {name: {key: float(np.mean([row[key] for row in metrics if row["condition"] == name]))
                    for key in ("baseline_auroc", "weak_crop_auroc", "delta_auroc")}
             for name in PRIMARY}
    checks = []
    for fold in range(3):
        folder = study / "folds" / f"random_{fold}"
        receipt = read_json(folder / "feature_fit.json")
        demand(sha256(folder / "features.joblib") == receipt["model_sha256"], "Changed fold transformer")
        features = joblib.load(folder / "features.joblib")
        leaves = {leaf.analyzer: leaf for leaf in vectorizers(features)}
        texts = [row["text"] for i, row in enumerate(train) if assignment[i] == fold]
        changed = [" ".join(text.split()) for text in texts]
        demand(all(before.split() == after.split() for before, after in zip(texts, changed)), "Whitespace condition changes word sequence")
        word_difference = leaves["word"].transform(texts) - leaves["word"].transform(changed)
        char_difference = leaves["char"].transform(texts) - leaves["char"].transform(changed)
        word_difference.eliminate_zeros()
        char_difference.eliminate_zeros()
        demand(word_difference.nnz == 0, "Whitespace condition changes word features")
        checks.append({"fold": fold, "n": len(texts),
                       "n_text_changed": sum(a != b for a, b in zip(texts, changed)),
                       "tokens_preserved": True, "word_feature_difference_nnz": int(word_difference.nnz),
                       "char_feature_changed_rows": int(np.sum(np.diff(char_difference.tocsr().indptr) > 0)),
                       "char_feature_difference_nnz": int(char_difference.nnz),
                       "installed_char_whitespace_regex": leaves["char"]._white_spaces.pattern,
                       "feature_model_sha256": receipt["model_sha256"]})
    return {"passed": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "study": str(study.relative_to(ROOT)), "models_fitted": 0, "test_text_accessed": False,
            "verifier_source_sha256": sha256(__file__), "condition_macro_means": means,
            "baseline_whitespace_minus_clean_macro_auroc": means["whitespace_only"]["baseline_auroc"] - means["clean"]["baseline_auroc"],
            "weak_crop_whitespace_minus_clean_macro_auroc": means["whitespace_only"]["weak_crop_auroc"] - means["clean"]["weak_crop_auroc"],
            "per_fold_feature_checks": checks,
            "interpretation": "Whitespace-only preserves the token sequence and word features but changes character features. The pinned char analyzer collapses runs of at least two whitespace characters, retaining isolated newline/tab characters until this condition replaces them with a space.",
            "limitations": "Observed format sensitivity is a candidate next hypothesis, not proof of the Kaggle gap's cause or evidence that normalization training would improve it."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--whitespace-only", action="store_true", help="Verify the already observed format effect without rerunning bootstrap")
    args = parser.parse_args()
    demand(not args.output.exists(), "Use a new verification output path")
    with threadpool_limits(limits=1):
        receipt = verify_whitespace_diagnostic(args.study) if args.whitespace_only else verify(args.study)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary_keys = ("passed", "models_fitted", "baseline_whitespace_minus_clean_macro_auroc", "weak_crop_whitespace_minus_clean_macro_auroc") if args.whitespace_only else ("passed", "outcome", "reason", "competition_svc_fits", "models_fitted_by_verifier")
    print(json.dumps({key: receipt[key] for key in summary_keys}))


if __name__ == "__main__":
    main()
