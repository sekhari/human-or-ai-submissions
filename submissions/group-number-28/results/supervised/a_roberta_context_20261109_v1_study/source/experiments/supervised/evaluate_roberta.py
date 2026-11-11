"""Zero-fit paired evaluation of the frozen contextual-encoder experiment.

Ranks are compared only within one fitted boundary. Raw WC120 margins are
never mixed with encoder probabilities and no held-cohort score calibration,
weight fitting, or deployable blend is produced by this analysis.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from .crop_evaluation import PRIMARY_CONDITIONS, bootstrap_cv


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPICS = ROOT / "results/supervised/a_q5_crop_revised_20261109_v6_study/topic_assignments.csv"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate headers: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV: {path}")
    return rows


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("An empty report is not a complete experiment")
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_input(study, path):
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = (study / path, ROOT / path)
    present = [candidate for candidate in candidates if candidate.exists()]
    if not present:
        raise FileNotFoundError(f"Cannot resolve study input: {path}")
    return present[0]


def unique_by(rows, keys, source):
    mapping = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        if key in mapping:
            raise ValueError(f"Duplicate key {key} in {source}")
        mapping[key] = row
    return mapping


def numeric_scores(values):
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Expected finite one-dimensional scores")
    return scores


def average_ranks(values):
    """Average ranks for exact ties, independent of arbitrary tied row order."""
    values = numeric_scores(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    ends = np.r_[starts[1:], len(values)]
    sorted_ranks = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def rank_correlation(before, after):
    a, b = average_ranks(before), average_ranks(after)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def pair_complementarity(labels, baseline, encoder):
    """Tie-aware AUC credit repairs/regressions, never an ensemble estimate.

    Each positive/negative pair earns 1, 0.5, or 0 according to its score
    ordering. A repair/regression is its positive/negative change in credit.
    Their net, divided by all pairs, must equal the paired AUC difference.
    """
    labels = np.asarray(labels, dtype=np.int64)
    baseline, encoder = numeric_scores(baseline), numeric_scores(encoder)
    if labels.shape != baseline.shape or baseline.shape != encoder.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid paired label/score shapes")
    positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    n_pairs = len(positives) * len(negatives)
    if not n_pairs:
        raise ValueError("Pair comparison needs both classes")
    baseline_difference = baseline[positives, None] - baseline[None, negatives]
    encoder_difference = encoder[positives, None] - encoder[None, negatives]
    baseline_credit = (baseline_difference > 0).astype(np.float64) + 0.5 * (baseline_difference == 0)
    encoder_credit = (encoder_difference > 0).astype(np.float64) + 0.5 * (encoder_difference == 0)
    change = encoder_credit - baseline_credit
    gained = float(np.maximum(change, 0).sum())
    lost = float(np.maximum(-change, 0).sum())
    baseline_auc = float(roc_auc_score(labels, baseline))
    encoder_auc = float(roc_auc_score(labels, encoder))
    net = (gained - lost) / n_pairs
    if not np.isclose(net, encoder_auc - baseline_auc, rtol=0, atol=1e-12):
        raise AssertionError("Pairwise credit net does not reconstruct AUROC difference")
    return {
        "n_positive_negative_pairs": n_pairs,
        "baseline_pair_credit": float(baseline_credit.sum()),
        "encoder_pair_credit": float(encoder_credit.sum()),
        "repaired_pair_credit": gained,
        "regressed_pair_credit": lost,
        "repair_credit_fraction": gained / n_pairs,
        "regression_credit_fraction": lost / n_pairs,
        "net_credit_fraction": net,
        "both_strictly_inverted_pairs": int(np.sum((baseline_credit == 0) & (encoder_credit == 0))),
        "baseline_strictly_inverted_pairs": int(np.sum(baseline_credit == 0)),
        "encoder_strictly_inverted_pairs": int(np.sum(encoder_credit == 0)),
        "spearman_rank_correlation": rank_correlation(baseline, encoder),
    }


def classification_numbers(labels, baseline, logits, probability):
    labels = np.asarray(labels, dtype=np.int64)
    baseline, logits, probability = map(numeric_scores, (baseline, logits, probability))
    if any(values.shape != labels.shape for values in (baseline, logits, probability)):
        raise ValueError("Metric arrays do not align")
    if not np.isin(labels, [0, 1]).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid labels/probabilities")
    n_human, n_ai = int(np.sum(labels == 0)), int(np.sum(labels == 1))
    defined = bool(n_human and n_ai)
    before = float(roc_auc_score(labels, baseline)) if defined else None
    after = float(roc_auc_score(labels, logits)) if defined else None
    probability_auc = float(roc_auc_score(labels, probability)) if defined else None
    clipped = np.clip(probability, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    log_loss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))) if len(labels) else None
    return {
        "n": len(labels), "n_human": n_human, "n_ai": n_ai,
        "wc120_auroc": before, "roberta_auroc": after,
        "delta_auroc": after - before if defined else None,
        "roberta_probability_auroc": probability_auc,
        "probability_minus_logit_auroc": probability_auc - after if defined else None,
        "roberta_log_loss": log_loss,
        "roberta_brier": float(np.mean((probability - labels) ** 2)) if len(labels) else None,
    }


def read_fit(study, fit):
    key = fit["key"]
    directory = study / "fits" / key
    paths = {
        "train_ids": resolve_input(study, fit["train_ids_file"]),
        "held_ids": resolve_input(study, fit["held_ids_file"]),
        "baseline": resolve_input(study, fit["baseline_predictions_file"]),
        "predictions": directory / "predictions.csv",
        "held_metadata": directory / "held_metadata.csv",
        "train_metadata": directory / "train_metadata.csv",
    }
    ids = [row["id"] for row in read_csv(paths["held_ids"])]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate held ID in {key}")
    held = unique_by(read_csv(paths["held_metadata"]), ("id",), paths["held_metadata"])
    if set(ids) != {key[0] for key in held}:
        raise ValueError(f"Held metadata coverage mismatch in {key}")
    metadata = [held[(identifier,)] for identifier in ids]
    labels = np.asarray([int(row["label"]) for row in metadata], dtype=np.int64)
    n_words = np.asarray([int(row["n_words"]) for row in metadata], dtype=np.int64)
    predictions = unique_by(read_csv(paths["predictions"]), ("condition", "id"), paths["predictions"])
    baseline = unique_by(read_csv(paths["baseline"]), ("condition", "id"), paths["baseline"])
    expected = {(condition, identifier) for condition in PRIMARY_CONDITIONS for identifier, count in zip(ids, n_words)
                if condition != "prefix_100" or count >= 150}
    if set(predictions) != expected or set(baseline) != expected:
        raise ValueError(f"Unexpected condition/ID coverage in {key}")
    conditions = {}
    for condition in PRIMARY_CONDITIONS:
        indices = np.asarray([i for i, count in enumerate(n_words) if condition != "prefix_100" or count >= 150], dtype=np.int64)
        selected = [predictions[(condition, ids[i])] for i in indices]
        reference = [baseline[(condition, ids[i])] for i in indices]
        for i, predicted, old in zip(indices, selected, reference):
            if (predicted["fit_key"] != key or predicted["scheme"] != fit["scheme"] or
                    int(predicted["fold"]) != int(fit["fold"]) or
                    int(predicted["label"]) != labels[i] or int(old["label"]) != labels[i]):
                raise ValueError(f"Prediction metadata mismatch in {key}/{condition}/{ids[i]}")
        conditions[condition] = (
            indices,
            numeric_scores([row["baseline_score"] for row in reference]),
            numeric_scores([row["logit_margin"] for row in selected]),
            numeric_scores([row["p_ai"] for row in selected]),
        )
    train_metadata = read_csv(paths["train_metadata"])
    train_ids = [row["id"] for row in read_csv(paths["train_ids"])]
    metadata_train_ids = {row["id"] for row in train_metadata}
    if (len(train_ids) != len(set(train_ids)) or set(train_ids) != metadata_train_ids or
            len(metadata_train_ids) != len(train_metadata) or set(ids) & metadata_train_ids):
        raise ValueError(f"Duplicate/overlapping training metadata in {key}")
    return {
        "fit": fit, "ids": ids, "labels": labels, "metadata": metadata,
        "train_metadata": train_metadata, "conditions": conditions,
        "input_hashes": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
    }


def coverage_statistics(rows, maximum):
    full = np.asarray([int(row["token_count_full"]) for row in rows], dtype=np.int64)
    kept = np.asarray([int(row["tokens_kept"]) for row in rows], dtype=np.int64)
    truncated = np.asarray([str(row["truncated"]).lower() in ("true", "1") for row in rows])
    if not np.array_equal(truncated, full > maximum) or np.any(kept != np.minimum(full, maximum)):
        raise ValueError("Tokenizer metadata does not match the frozen truncation rule")
    return {
        "n": len(rows), "truncated_n": int(truncated.sum()),
        "truncated_fraction": float(truncated.mean()),
        "full_token_count_quantiles": {str(q): float(np.quantile(full, q)) for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 1)},
        "kept_token_fraction_of_total": float(kept.sum() / full.sum()),
        "definition": "Tokenizer counts include special tokens; kept=min(full,max_length), not whitespace words.",
    }


def subgroup_rows(record, maximum):
    fit, labels = record["fit"], record["labels"]
    fitting_words = np.asarray([int(row["n_words"]) for row in record["train_metadata"]])
    boundaries = np.quantile(fitting_words, [1 / 3, 2 / 3], method="linear")
    if "length_word_tertiles" in fit and not np.array_equal(boundaries, fit["length_word_tertiles"]):
        raise ValueError(f"Training-only length boundaries differ from frozen plan: {fit['key']}")
    words = np.asarray([int(row["n_words"]) for row in record["metadata"]])
    bins = np.digitize(words, boundaries, right=True)
    full_tokens = np.asarray([int(row["token_count_full"]) for row in record["metadata"]])
    _, baseline, logits, probability = record["conditions"]["clean"]
    masks = [("natural_word_length", name, bins == index) for index, name in enumerate(("short", "middle", "long"))]
    masks += [("token_coverage", "fully_covered", full_tokens <= maximum),
              ("token_coverage", "truncated", full_tokens > maximum)]
    result = []
    for kind, group, mask in masks:
        result.append({"fit_key": fit["key"], "scheme": fit["scheme"], "fold": int(fit["fold"]),
                       "condition": "clean", "group_kind": kind, "group": group,
                       "training_word_lower_boundary": float(boundaries[0]),
                       "training_word_upper_boundary": float(boundaries[1]),
                       **classification_numbers(labels[mask], baseline[mask], logits[mask], probability[mask])})
    return result


def aggregate_metrics(rows, complementarity):
    result = {}
    for scheme in ("random", "topic", "validation"):
        result[scheme] = {}
        for condition in PRIMARY_CONDITIONS:
            selected = sorted((row for row in rows if row["scheme"] == scheme and row["condition"] == condition), key=lambda row: row["fold"])
            expected_folds = [0] if scheme == "validation" else [0, 1, 2]
            if [row["fold"] for row in selected] != expected_folds:
                raise ValueError(f"Incomplete/duplicate folds for {scheme}/{condition}")
            paired = [row for row in complementarity if row["scheme"] == scheme and row["condition"] == condition]
            if any(row["delta_auroc"] is None for row in selected):
                raise ValueError(f"Undefined primary AUC in {scheme}/{condition}")
            deltas = [row["delta_auroc"] for row in selected]
            repair = float(np.mean([row["repair_credit_fraction"] for row in paired]))
            regression = float(np.mean([row["regression_credit_fraction"] for row in paired]))
            if not np.isclose(repair - regression, np.mean(deltas), rtol=0, atol=1e-12):
                raise AssertionError("Macro pair repair/regression does not match macro delta")
            result[scheme][condition] = {
                "folds": expected_folds, "held_n_by_fold": [row["n"] for row in selected],
                "wc120_macro_auroc": float(np.mean([row["wc120_auroc"] for row in selected])),
                "roberta_macro_auroc": float(np.mean([row["roberta_auroc"] for row in selected])),
                "macro_delta_auroc": float(np.mean(deltas)), "fold_delta_aurocs": deltas,
                "minimum_fold_delta": min(deltas), "maximum_fold_delta": max(deltas),
                "positive_folds": sum(value > 0 for value in deltas),
                "macro_repair_credit_fraction": repair, "macro_regression_credit_fraction": regression,
                "aggregation": "Equal-weight mean of within-fit metrics; no pooled cross-fit score AUC.",
            }
    return result


def analyze(study, output):
    plan_path = study / "plan.json"
    plan = json.loads(plan_path.read_text())
    fits = plan["fits"]
    expected = {f"{scheme}_{fold}" for scheme in ("random", "topic") for fold in range(3)} | {"canonical_8000"}
    if {fit["key"] for fit in fits} != expected or len(fits) != 7:
        raise ValueError("The complete frozen seven-fit comparison is required")
    records = [read_fit(study, fit) for fit in fits]
    topic_path = resolve_input(study, plan.get("topic_assignments_file", DEFAULT_TOPICS))
    topic_rows = unique_by(read_csv(topic_path), ("id",), topic_path)
    maximum = int(plan["recipe"]["max_length"])
    per_fold, complementarity, subgroups, coverage = [], [], [], {}
    bootstrap_records = {scheme: [] for scheme in ("random", "topic", "validation")}
    inputs = {"plan.json": sha256(plan_path), "topic_assignments": {"path": str(topic_path), "sha256": sha256(topic_path)}}
    for record in records:
        fit, labels = record["fit"], record["labels"]
        identifiers = {"fit_key": fit["key"], "scheme": fit["scheme"], "fold": int(fit["fold"])}
        for condition, (indices, baseline, logits, probability) in record["conditions"].items():
            per_fold.append({**identifiers, "condition": condition,
                             **classification_numbers(labels[indices], baseline, logits, probability)})
            complementarity.append({**identifiers, "condition": condition,
                                    **pair_complementarity(labels[indices], baseline, logits)})
        subgroups.extend(subgroup_rows(record, maximum))
        coverage[fit["key"]] = {"training": coverage_statistics(record["train_metadata"], maximum),
                               "held": coverage_statistics(record["metadata"], maximum)}
        bootstrap_record = {"fold": int(fit["fold"]), "labels": labels,
                            "conditions": {condition: value[:3] for condition, value in record["conditions"].items()}}
        if fit["scheme"] == "topic":
            bootstrap_record["topics"] = np.asarray([int(topic_rows[(identifier,)]["topic"]) for identifier in record["ids"]])
        bootstrap_records[fit["scheme"]].append(bootstrap_record)
        inputs[fit["key"]] = record["input_hashes"]
    aggregate = aggregate_metrics(per_fold, complementarity)
    bootstrap_plan = {
        "seeds": {name: int(plan["seeds"][name]) for name in ("random_bootstrap", "topic_bootstrap", "validation_bootstrap")},
        "statistics": {"bootstrap_repetitions": int(plan["statistics"]["bootstrap_repetitions"]),
                       "minimum_valid_bootstrap_fraction_all_schemes": float(plan["statistics"]["minimum_valid_bootstrap_fraction_all_schemes"]),
                       "interval_limitations": ["Conditional on fixed models/splits; no training variation or adaptive selection correction.",
                                                "Overlapping fitting folds are not independent replications.",
                                                "Whole-topic sensitivity uses only the existing 12 lexical-topic/style proxy groups.",
                                                "Dependent text transformations are not independent datasets; intervals are descriptive and unadjusted for multiplicity."]},
    }
    bootstraps = {scheme: bootstrap_cv(value, scheme, bootstrap_plan) for scheme, value in bootstrap_records.items()}
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_fold_metrics.csv", per_fold)
    write_csv(output / "pair_complementarity.csv", complementarity)
    write_csv(output / "clean_subgroups.csv", subgroups)
    write_json(output / "aggregate_metrics.json", aggregate)
    write_json(output / "paired_bootstrap.json", bootstraps)
    write_json(output / "coverage_summary.json", coverage)
    write_json(output / "input_hashes.json", inputs)
    write_json(output / "analysis_receipt.json", {
        "status": "complete", "fits": len(records), "primary_condition_rows": len(per_fold),
        "analysis_source_sha256": sha256(__file__), "bootstrap_source_sha256": sha256(Path(__file__).with_name("crop_evaluation.py")),
        "new_model_fits": 0, "encoder_primary_score": "logit_margin", "probability_score_secondary": "p_ai",
        "raw_margin_probability_mixing": False, "held_cohort_score_transform": False,
        "new_candidate_or_blend": False, "scientific_quality_gate_applied": False,
        "pair_decomposition": "Positive/negative changes in 1/0.5/0 pair credit, divided by all AI-human pairs. Descriptive evidence, not an ensemble score.",
        "log_loss_clipping": "float64 machine epsilon at both probability boundaries",
        "bootstrap_settings": bootstrap_plan,
        "output_sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    })
    print(json.dumps({"status": "complete", "output": str(output), "clean": {scheme: aggregate[scheme]["clean"] for scheme in aggregate}}, indent=2))


class MathematicalChecks(unittest.TestCase):
    def test_pair_credit_delta_with_ties_and_unequal_classes(self):
        y = np.array([0, 0, 1, 1, 1])
        before = np.array([0.2, 0.4, 0.4, 0.8, 0.1])
        after = np.array([0.2, 0.4, 0.3, 0.8, 0.4])
        result = pair_complementarity(y, before, after)
        self.assertEqual(result["n_positive_negative_pairs"], 6)
        self.assertGreater(result["repaired_pair_credit"], 0)
        self.assertGreater(result["regressed_pair_credit"], 0)
        self.assertAlmostEqual(result["net_credit_fraction"], roc_auc_score(y, after) - roc_auc_score(y, before), places=14)

    def test_average_ties_and_monotone_correlation(self):
        np.testing.assert_array_equal(average_ranks([0.2, 0.2, 0.1, 0.9]), [1.5, 1.5, 0, 3])
        self.assertAlmostEqual(rank_correlation([1, 2, 2, 5], [2, 3, 3, 9]), 1)
        self.assertIsNone(rank_correlation([1, 1], [2, 3]))

    def test_missing_class_kept_undefined_and_probability_saturation_visible(self):
        single = classification_numbers([1, 1], [1, 2], [2, 3], [0.8, 0.9])
        self.assertIsNone(single["roberta_auroc"])
        saturated = classification_numbers([0, 1], [1, 2], [30, 31], [1.0, 1.0])
        self.assertEqual(saturated["roberta_auroc"], 1)
        self.assertEqual(saturated["roberta_probability_auroc"], 0.5)
        self.assertTrue(np.isfinite(saturated["roberta_log_loss"]))

    def test_probability_and_duplicate_key_validation(self):
        with self.assertRaises(ValueError):
            classification_numbers([0, 1], [0, 1], [0, 1], [0, 1.1])
        with self.assertRaises(ValueError):
            unique_by([{"id": "a"}, {"id": "a"}], ("id",), "synthetic")
        with self.assertRaises(ValueError):
            pair_complementarity([0, 0], [0, 1], [0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(MathematicalChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    if args.study_dir is None:
        parser.error("--study-dir is required unless --self-test is used")
    study = args.study_dir.resolve()
    analyze(study, args.output_dir.resolve() if args.output_dir else study / "analysis")


if __name__ == "__main__":
    main()
