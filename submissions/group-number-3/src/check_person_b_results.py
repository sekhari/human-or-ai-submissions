"""Compare a fresh Person B run to the primary run; never fits a model."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import ROOT
from person_b import sha256


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/person_b")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-model-refactor", action="store_true",
                        help="Allow only models.py source to differ; parameters and predictions must agree.")
    args = parser.parse_args()
    reference, candidate = args.reference.resolve(), args.candidate.resolve()
    original, fresh = read(reference / "run_manifest.json"), read(candidate / "run_manifest.json")
    a, b = original["protocol"], fresh["protocol"]
    checks = {}
    for name in ("config", "config_sha256", "input_sha256", "python", "packages",
                 "model_parameters", "random_training_ids_sha256", "random_validation_ids_sha256"):
        checks[f"same_{name}"] = a[name] == b[name]
    changed_sources = sorted(name for name in a["code_sha256"]
                             if a["code_sha256"][name] != b["code_sha256"].get(name))
    checks["source_changes_within_declared_scope"] = (
        set(changed_sources) <= {"src/models.py"} if args.allow_model_refactor else not changed_sources)
    checks["test_set_unused_in_both_runs"] = original["test_csv_used"] is False and fresh["test_csv_used"] is False

    train = pd.read_csv(ROOT / "data/train.csv", dtype={"id": "string"}).set_index("id")
    splits = pd.read_csv(ROOT / "splits/splits.csv", dtype={"id": "string"})
    val_ids = set(splits.loc[splits.random_fold.eq("val"), "id"])
    pool_ids = set(splits.loc[splits.random_fold.eq("train"), "id"])
    checks["protocol_inputs_unchanged"] = all(
        sha256(ROOT / name) == digest for name, digest in a["input_sha256"].items())
    checks["fixed_shared_split_unchanged"] = sha256(ROOT / "splits/splits.csv") == a["input_sha256"]["splits/splits.csv"]

    baseline_a, baseline_b = read(reference / "baseline_results.json"), read(candidate / "baseline_results.json")
    q2_a, q2_b = read(reference / "q2_run_details.json"), read(candidate / "q2_run_details.json")
    q3_a, q3_b = read(reference / "q3_run_details.json"), read(candidate / "q3_run_details.json")
    for fold in ("random_fold", "ood_fold"):
        checks[f"baseline_{fold}_scores_and_sizes"] = all(
            abs(baseline_a[fold][key] - baseline_b[fold][key]) < 1e-12
            for key in ("training_rows", "validation_rows", "training_auroc", "validation_auroc"))
    checks["q2_scores_and_sizes"] = len(q2_a) == len(q2_b) == 5 and all(
        all(abs(x[key] - y[key]) < 1e-12 for key in
            ("training_rows", "validation_rows", "training_auroc", "validation_auroc"))
        for x, y in zip(q2_a, q2_b))
    for key in ("low_threshold", "high_threshold", "rounds", "pseudo_label_weight", "selected_rows",
                "pseudo_human_rows", "pseudo_ai_rows", "supervised_validation_auroc",
                "with_unlabeled_validation_auroc", "delta_validation_auroc"):
        checks[f"q3_same_{key}"] = abs(q3_a[key] - q3_b[key]) < 1e-12
    checks["q3_representation_frozen"] = q3_b["features_unchanged"] and (
        q3_b["feature_fingerprint_before"] == q3_b["feature_fingerprint_after"] ==
        baseline_b["random_fold"]["feature_fingerprint"])

    subsets = read(candidate / "q2_subset_ids.json")
    checks["same_q2_subset_ids"] = subsets == read(reference / "q2_subset_ids.json")
    previous = set()
    for size in a["config"]["training_sizes"]:
        ids = subsets[str(size)]
        selected = set(ids)
        checks[f"q2_{size}_balanced_nested_disjoint"] = (
            len(ids) == len(selected) == size and previous <= selected <= pool_ids and
            not selected & val_ids and train.loc[ids, "label"].value_counts().to_dict() == {0: size // 2, 1: size // 2})
        previous = selected

    expected_scores = {f"baseline_{fold}_predictions.csv": result["validation_auroc"]
                       for fold, result in baseline_b.items()}
    expected_scores.update({f"q2_n{row['training_rows']}_validation_predictions.csv": row["validation_auroc"] for row in q2_b})
    expected_scores["q3_validation_predictions.csv"] = q3_b["with_unlabeled_validation_auroc"]
    prediction_differences = {}
    for filename, expected_auc in expected_scores.items():
        left = pd.read_csv(reference / filename, dtype={"id": "string"}).set_index("id")
        right = pd.read_csv(candidate / filename, dtype={"id": "string"}).set_index("id")
        fold = "ood_fold" if "ood_fold" in filename else "random_fold"
        expected_ids = set(splits.loc[splits[fold].eq("val"), "id"])
        valid_ids = right.index.is_unique and set(right.index) == set(left.index) == expected_ids
        checks[filename + "_ids"] = valid_ids
        if not valid_ids:
            continue
        p = right.ai_probability.to_numpy()
        prediction_differences[filename] = float(np.max(np.abs(p - left.loc[right.index, "ai_probability"].to_numpy())))
        checks[filename + "_probabilities_labels_auc"] = bool(
            np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and
            np.array_equal(right.true_label.to_numpy(), train.loc[right.index, "label"].to_numpy()) and
            abs(roc_auc_score(right.true_label, p) - expected_auc) < 1e-12 and
            prediction_differences[filename] <= 1e-10)
    checks["pseudo_label_selection_identical"] = (
        (reference / "q3_selected_pseudo_labels.csv").read_bytes() ==
        (candidate / "q3_selected_pseudo_labels.csv").read_bytes())
    checks["all_new_fits_converged_without_warnings"] = all(
        not row["warnings"] and max(row["classifier_iterations"]) < 2000
        for row in list(baseline_b.values()) + q2_b + [q3_b])
    checks["completed_markers_match_candidate_protocol"] = all(
        read(candidate / f"{stage}_completed.json")["protocol_signature"] == fresh["protocol_signature"] and
        read(candidate / f"{stage}_completed.json")["status"] == "completed"
        for stage in ("baseline", "q2", "q3"))
    report = {"checked_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
              "reference_commit": original["originating_code_commit"],
              "candidate_commit": fresh["originating_code_commit"],
              "reference_protocol": original["protocol_signature"],
              "candidate_protocol": fresh["protocol_signature"],
              "changed_sources": changed_sources, "checks": {k: bool(v) for k, v in checks.items()},
              "prediction_max_abs_differences": prediction_differences,
              "all_passed": all(checks.values()),
              "scope": "Fresh fits on this Mac using the installed environment; no fresh installation or cross-platform claim."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks_passed": sum(checks.values()), "total_checks": len(checks),
                      "all_passed": report["all_passed"], "changed_sources": changed_sources,
                      "maximum_prediction_difference": max(prediction_differences.values()),
                      "failed": [key for key, value in checks.items() if not value]}, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
