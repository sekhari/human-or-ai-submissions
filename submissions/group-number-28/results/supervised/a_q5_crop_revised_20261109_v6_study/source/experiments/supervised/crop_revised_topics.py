"""Reuse the frozen topic proxy under an explicitly revised exploratory rule.

Only the historical content-topic count and maximum held-fold-size checks are
diagnostic. Topic assignments, reviews, fold IDs and seeds are reused exactly;
this module performs no model fitting and does not claim a new blind review.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from human_ai.data import ROOT, read_csv, sha256
from .crop_topics import matched_random_assignments
from .run_q5 import atomic_csv, atomic_json


DIAGNOSTIC_ONLY = {"minimum_interpretable_content_topics", "maximum_held_n"}
REQUIRED_PARENT_HASHES = {
    "topic_assignments.csv", "topic_folds.csv", "topic_review.json",
    "topic_review_packet.json", "topic_preparation.json",
    "topic_finalization.json", "plan.json", "completion.json",
}
COPIES = {
    "topic_assignments.csv": "topic_assignments.csv",
    "topic_folds.csv": "topic_folds.csv",
    "topic_review.json": "topic_review.json",
    "topic_review_packet.json": "topic_review_packet.json",
    "topic_proxy.joblib": "topic_proxy.joblib",
    "topic_preparation.json": "topic_preparation.json",
    "topic_preparation_context.json": "topic_preparation_context.json",
    "parent_topic_finalization.json": "topic_finalization.json",
    "parent_topic_quality.json": "topic_quality.json",
    "parent_plan.json": "plan.json",
    "parent_completion.json": "completion.json",
}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _json_hash(value):
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _input_digest(train):
    digest = hashlib.sha256()
    for row in train:
        digest.update(json.dumps([str(row["id"]), row["text"]], ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_hashes(base, hashes):
    for name, expected in hashes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe artifact name: {name}")
        actual = base / path
        if not actual.is_file() or sha256(actual) != expected:
            raise ValueError(f"Changed or missing reused artifact: {actual}")


def _copy_identical(source, target):
    if target.exists():
        if sha256(target) != sha256(source):
            raise ValueError(f"Cannot overwrite a changed inherited artifact: {target}")
    else:
        shutil.copy2(source, target)


def _fold_rows(scheme, assignment, labels, topics):
    rows = []
    for fold in range(3):
        held = assignment == fold
        fitting = ~held
        fit_topics = set(topics[fitting].tolist())
        held_topics = set(topics[held].tolist())
        rows.append({
            "scheme": scheme, "fold": fold,
            "fitting_n": int(fitting.sum()),
            "fitting_human": int(np.sum(fitting & (labels == 0))),
            "fitting_ai": int(np.sum(fitting & (labels == 1))),
            "held_n": int(held.sum()),
            "held_human": int(np.sum(held & (labels == 0))),
            "held_ai": int(np.sum(held & (labels == 1))),
            "fitting_topics_n": len(fit_topics), "held_topics_n": len(held_topics),
            "shared_topics_n": len(fit_topics & held_topics),
        })
    return rows


def prepare_revised_topics(study, train, plan):
    """Copy exact inherited groups and create the exact-count random control.

    Required plan fields are ``parent_study`` and ``parent_artifact_sha256``.
    The two softened names must be listed under
    ``topic_proxy.diagnostic_only_constraints``. All remaining topic settings
    and numerical historical thresholds must match the parent plan.
    """
    study = Path(study)
    parent = ROOT / plan["parent_study"]
    if study.resolve() == parent.resolve():
        raise ValueError("The revised run must not alter its parent directory")
    study.mkdir(parents=True, exist_ok=True)
    pins = dict(plan["parent_artifact_sha256"])
    if not REQUIRED_PARENT_HASHES <= set(pins):
        raise ValueError("Revised plan does not pin every required parent artifact")
    _verify_hashes(parent, pins)
    preparation = _read_json(parent / "topic_preparation.json")
    finalization = _read_json(parent / "topic_finalization.json")
    _verify_hashes(parent, preparation["artifact_sha256"])
    _verify_hashes(parent, finalization["artifact_sha256"])

    parent_plan = _read_json(parent / "plan.json")
    settings = plan["topic_proxy"]
    if set(settings.get("diagnostic_only_constraints", [])) != DIAGNOSTIC_ONLY:
        raise ValueError("Only the two authorized historical constraints may become diagnostic")
    for key in ("vectorizer", "svd", "post_svd_normalization", "kmeans", "folds",
                "quality_gates", "fold_quality_gates"):
        if settings[key] != parent_plan["topic_proxy"][key]:
            raise ValueError(f"A topic setting changed beyond diagnostic policy: {key}")
    if plan["seeds"]["matched_random_folds"] != parent_plan["seeds"]["matched_random_folds"]:
        raise ValueError("The exact-count random-control seed may not be searched")
    ids = [str(row["id"]) for row in train]
    if len(ids) != plan["data_boundaries"]["development_fitting_n"] or len(ids) != len(set(ids)):
        raise ValueError("Expected every canonical fitting original exactly once")
    input_digest = _input_digest(train)
    if input_digest != preparation["execution_contract"]["input_id_text_sha256"]:
        raise ValueError("Original fitting IDs/texts differ from the frozen parent input")
    assignment_rows = read_csv(parent / "topic_assignments.csv")
    fold_rows = read_csv(parent / "topic_folds.csv")
    if [row["id"] for row in assignment_rows] != ids or [row["id"] for row in fold_rows] != ids:
        raise ValueError("Parent topic/fold IDs differ from canonical fitting order")
    topics = np.array([int(row["topic"]) for row in assignment_rows], dtype=int)
    topic_folds = np.array([int(row["held_fold"]) for row in fold_rows], dtype=int)
    n_clusters = int(settings["kmeans"]["n_clusters"])
    if np.any(topics < 0) or np.any(topics >= n_clusters) or set(topic_folds.tolist()) != {0, 1, 2}:
        raise ValueError("Invalid inherited topic or held-fold index")
    reviews = _read_json(parent / "topic_review.json")
    packet = _read_json(parent / "topic_review_packet.json")
    if reviews["topic_review_packet_sha256"] != sha256(parent / "topic_review_packet.json"):
        raise ValueError("Parent review packet changed")
    if (reviews.get("labels_read_before_review_freeze") is not False
            or preparation.get("record_labels_read") is not False):
        raise ValueError("Parent records do not preserve the historical blind review boundary")
    by_reviewer = []
    for name in ("review_a", "review_b"):
        reviewed = reviews[name]
        if reviewed.get("reviewed_without_labels_or_scores") is not True:
            raise ValueError("Historical review lacks blindness attestation")
        by_topic = {int(row["topic"]): row for row in reviewed["clusters"]}
        if set(by_topic) != set(range(n_clusters)):
            raise ValueError("Historical review is incomplete")
        for cluster in packet["clusters"]:
            if set(by_topic[cluster["topic"]]["inspected_ids"]) != set(cluster["sample_ids"]):
                raise ValueError("Historical review sample IDs changed")
        by_reviewer.append(by_topic)
    combined = [
        {"topic": topic,
         "decision": by_reviewer[0][topic]["decision"]
         if by_reviewer[0][topic]["decision"] == by_reviewer[1][topic]["decision"] else "mixed"}
        for topic in range(n_clusters)
    ]
    if combined != reviews["combined"]:
        raise ValueError("Historical combined review violates its frozen disagreement rule")

    for destination, source in COPIES.items():
        _copy_identical(parent / source, study / destination)
    for name in ("blind_review_a.json", "blind_review_b.json"):
        if (parent / name).exists():
            _copy_identical(parent / name, study / name)
    contract = {
        "plan_content_sha256": _json_hash(plan), "input_id_text_sha256": input_digest,
        "n_originals": len(train), "parent_study": str(parent.relative_to(ROOT)),
        "parent_artifact_sha256": pins,
        "source_sha256": {
            "experiments/supervised/crop_revised_topics.py": sha256(__file__),
            "experiments/supervised/crop_topics.py": sha256(ROOT / "experiments/supervised/crop_topics.py"),
            "experiments/supervised/run_q5.py": sha256(ROOT / "experiments/supervised/run_q5.py"),
            "human_ai/data.py": sha256(ROOT / "human_ai/data.py"),
        },
    }
    reuse_path = study / "topic_reuse_receipt.json"
    inherited_hashes = {destination: {"parent_filename": source, "sha256": sha256(parent / source)}
                        for destination, source in COPIES.items()}
    if reuse_path.exists():
        reuse = _read_json(reuse_path)
        if reuse["execution_contract"] != contract or reuse["inherited_artifacts"] != inherited_hashes:
            raise ValueError("Cannot resume changed topic reuse contract")
    else:
        atomic_json(reuse_path, {
            "status": "adaptive_revised_protocol", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_contract": contract, "inherited_artifacts": inherited_hashes,
            "topic_grouping_fits_performed": 0, "new_blind_reviews_performed": False,
            "historical_blind_reviews_reused": True,
            "historical_preparation_receipts_copied_unchanged": True,
            "predictive_features_reused": False,
            "interpretation": "Previously observed diagnostics motivated the authorized revision. No fresh blindness or pristine preregistration is claimed.",
        })
    final_path = study / "topic_finalization.json"
    if final_path.exists():
        saved = _read_json(final_path)
        if saved["execution_contract"] != contract:
            raise ValueError("Cannot resume changed revised topic finalization")
        _verify_hashes(study, saved["artifact_sha256"])
        quality = _read_json(study / "topic_quality.json")
        random = np.array([int(row["held_fold"]) for row in read_csv(study / "random_folds.csv")])
        return {"topic_valid": quality["topic_valid"], "random_folds": random,
                "topic_folds": topic_folds if quality["topic_valid"] else None, "quality": quality}

    labels = np.array([int(row["label"]) for row in train], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Both original binary classes are required")
    q, bounds = settings["quality_gates"], settings["fold_quality_gates"]
    counts = np.bincount(topics, minlength=n_clusters)
    class_rows = []
    for topic in range(n_clusters):
        current = labels[topics == topic]
        n0, n1 = int(np.sum(current == 0)), int(np.sum(current == 1))
        class_rows.append({"topic": topic, "n": len(current), "human": n0, "ai": n1,
                           "dominant_class_fraction": max(n0, n1) / len(current) if len(current) else None})
    nmi = float(normalized_mutual_info_score(topics, labels, average_method=q["nmi_average_method"]))
    n_content = sum(row["decision"] == "content" for row in combined)
    n_style = sum(row["decision"] == "style-format" for row in combined)
    topic_details = _fold_rows("topic", topic_folds, labels, topics)
    random_folds = matched_random_assignments(labels, topic_folds, plan["seeds"]["matched_random_folds"])
    random_details = _fold_rows("matched_random", random_folds, labels, topics)
    for row in topic_details:
        row["historical_size_cap_passed"] = row["held_n"] <= bounds["maximum_held_n"]
        row["quality_passed"] = bool(
            row["held_n"] >= bounds["minimum_held_n"]
            and row["fitting_n"] > 0
            and min(row["held_human"], row["held_ai"]) >= bounds["minimum_held_n_per_class"]
            and bounds["minimum_held_ai_fraction"] <= row["held_ai"] / row["held_n"] <= bounds["maximum_held_ai_fraction"]
            and row["shared_topics_n"] == 0
        )
    checks = {
        "all_clusters_nonempty_and_large_enough": bool(np.all(counts >= q["minimum_cluster_n"])),
        "maximum_cluster_size": bool(np.all(counts <= q["maximum_cluster_n"])),
        "maximum_style_format_topics": n_style <= q["maximum_style_or_format_clusters"],
        "topic_label_nmi": nmi <= q["maximum_topic_label_nmi_inclusive"],
        "maximum_cluster_dominant_class_fraction": all(
            row["dominant_class_fraction"] is not None and row["dominant_class_fraction"] < q["maximum_cluster_dominant_class_fraction_exclusive"]
            for row in class_rows),
        "topic_groups_are_disjoint_across_folds": all(row["shared_topics_n"] == 0 for row in topic_details),
        "topic_fold_structural_balance": all(row["quality_passed"] for row in topic_details),
        "matched_random_class_and_total_counts": all(
            all(a[key] == b[key] for key in ("held_n", "held_human", "held_ai", "fitting_n", "fitting_human", "fitting_ai"))
            for a, b in zip(topic_details, random_details, strict=True)),
    }
    diagnostics = {
        "minimum_interpretable_content_topics": {
            "required_historically": q["minimum_content_topic_clusters"], "observed": n_content,
            "historical_check_passed": n_content >= q["minimum_content_topic_clusters"],
            "policy": "diagnostic_only",
        },
        "maximum_held_n": {
            "required_historically": bounds["maximum_held_n"],
            "held_fold_sizes": [row["held_n"] for row in topic_details],
            "historical_check_passed": all(row["historical_size_cap_passed"] for row in topic_details),
            "policy": "diagnostic_only",
        },
    }
    warnings = []
    if not diagnostics["minimum_interpretable_content_topics"]["historical_check_passed"]:
        warnings.append(f"Only {n_content} of {n_clusters} inherited clusters were unanimously content-interpretable; mixed/style groups remain in the analysis.")
    if not diagnostics["maximum_held_n"]["historical_check_passed"]:
        warnings.append("The inherited held-fold sizes are imbalanced; corresponding random folds exactly match each size and class count.")
    valid = all(checks.values())
    quality = {
        "status": "adaptive_revised_protocol", "topic_valid": bool(valid),
        "promotion_allowed_by_grouping": bool(valid), "checks": checks, "hard_checks": checks,
        "diagnostics": diagnostics, "diagnostic_warnings": warnings,
        "n_content_topics": n_content, "n_style_format_topics": n_style,
        "n_mixed_topics": n_clusters - n_content - n_style, "topic_label_nmi": nmi,
        "class_counts_by_topic": class_rows, "topic_fold_details": topic_details,
        "topic_folds_generated": True, "random_scheme": "matched_random",
        "review_sha256": sha256(study / "topic_review.json"),
        "quality_settings": q, "fold_quality_settings": bounds,
        "parent_study": str(parent.relative_to(ROOT)),
        "diagnostic_only_constraints": sorted(DIAGNOSTIC_ONLY),
        "interpretation": "Fixed inherited lexical-topic proxy with known mixed/style groups; all topics retained, not true domain/generator separation.",
    }
    atomic_json(study / "topic_quality.json", quality)
    atomic_csv(study / "random_folds.csv", ["id", "held_fold"],
               ({"id": row["id"], "held_fold": int(fold)} for row, fold in zip(train, random_folds, strict=True)))
    # Use a uniform CSV schema; diagnostic quality flags live in JSON above.
    count_rows = _fold_rows("topic", topic_folds, labels, topics) + random_details
    atomic_csv(study / "fold_counts.csv", list(count_rows[0]), count_rows)
    artifact_names = list(COPIES) + ["topic_reuse_receipt.json", "topic_quality.json", "random_folds.csv", "fold_counts.csv"]
    atomic_json(final_path, {
        "status": "adaptive_revised_protocol", "execution_contract": contract,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {name: sha256(study / name) for name in artifact_names},
        "topic_valid": bool(valid), "parent_study": str(parent.relative_to(ROOT)),
        "topic_grouping_fits_performed": 0,
    })
    return {"topic_valid": bool(valid), "random_folds": random_folds,
            "topic_folds": topic_folds if valid else None, "quality": quality}
