"""Paired, no-fit evaluation for the fixed weak-crop protocol.

Every condition retains indices into original held documents. Bootstrap draws
are made on those originals (or whole topics), then reused for every model arm
and condition. Raw margins are compared only within the fitted fold.

Run ``python -m experiments.supervised.crop_evaluation --self-test`` for the
embedded synthetic checks. They perform zero classifier or feature fits.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from .q5_sensitivity import apply_condition


PRIMARY_CONDITIONS = (
    "clean", "quote_and_dash", "digit_runs_to_zero", "whitespace_only", "prefix_100",
)
SEALED_CONDITIONS = ("suffix_100", "middle_100")


def make_conditions(texts, include_sealed=False):
    """Return ``condition: (local original indices, transformed texts)``.

    These fixed definitions deliberately match q5_crop_plan.json. The caller
    must not request sealed conditions until its declared primary gates pass.
    """
    texts = list(texts)
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Every original passage must be a string")
    indices = np.arange(len(texts), dtype=np.int64)
    words = [text.split() for text in texts]
    eligible = np.asarray([i for i, tokens in enumerate(words) if len(tokens) >= 150], dtype=np.int64)
    conditions = {
        "clean": (indices.copy(), texts.copy()),
        "quote_and_dash": (
            indices.copy(),
            [apply_condition(apply_condition(text, "quote_typography"), "dash_typography") for text in texts],
        ),
        "digit_runs_to_zero": (indices.copy(), [re.sub(r"\d+", "0", text) for text in texts]),
        "whitespace_only": (indices.copy(), [" ".join(tokens) for tokens in words]),
        "prefix_100": (eligible.copy(), [" ".join(words[i][:100]) for i in eligible]),
    }
    if include_sealed:
        conditions["suffix_100"] = (eligible.copy(), [" ".join(words[i][-100:]) for i in eligible])
        conditions["middle_100"] = (
            eligible.copy(),
            [" ".join(words[i][(len(words[i]) - 100) // 2:(len(words[i]) - 100) // 2 + 100]) for i in eligible],
        )
    return conditions


def _checked_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Expected one-dimensional binary labels")
    return labels.astype(np.int64, copy=False)


def _checked_condition(value, n_original):
    if len(value) != 3:
        raise ValueError("Each margin condition must be (original_indices, baseline, weak_crop)")
    indices = np.asarray(value[0])
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Condition indices must be a one-dimensional integer array")
    indices = indices.astype(np.int64, copy=False)
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n_original):
        raise ValueError("Condition indices must be unique and refer to held originals")
    before, after = (np.asarray(scores, dtype=np.float64) for scores in value[1:])
    if before.shape != (len(indices),) or after.shape != before.shape:
        raise ValueError("Each arm needs one score per condition index")
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("Margins must be finite")
    return indices, before, after


def evaluate_margins(scheme, fold, labels, condition_margins, minimum_per_class=50):
    """Calculate schema-stable paired AUC rows, retaining insufficient counts."""
    labels = _checked_labels(labels)
    rows = []
    for name, value in condition_margins.items():
        indices, before, after = _checked_condition(value, len(labels))
        y = labels[indices]
        n_human, n_ai = int(np.sum(y == 0)), int(np.sum(y == 1))
        defined = n_human > 0 and n_ai > 0
        before_auc = float(roc_auc_score(y, before)) if defined else None
        after_auc = float(roc_auc_score(y, after)) if defined else None
        rows.append({
            "scheme": str(scheme), "fold": int(fold), "condition": str(name),
            "n": len(indices), "n_human": n_human, "n_ai": n_ai,
            "baseline_auroc": before_auc, "weak_crop_auroc": after_auc,
            "delta_auroc": after_auc - before_auc if defined else None,
            "evidence_sufficient": bool(min(n_human, n_ai) >= minimum_per_class),
        })
    return rows


def _is_true(value):
    return value is True or value == "True" or value == "true"


def _number(value):
    if value is None or value == "":
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _condition_summary(rows, scheme, condition):
    selected = [row for row in rows if row["scheme"] == scheme and row["condition"] == condition]
    folds = [int(row["fold"]) for row in selected]
    complete = len(selected) == 3 and sorted(folds) == [0, 1, 2]
    selected = sorted(selected, key=lambda row: int(row["fold"]))
    sufficient = complete and all(_is_true(row["evidence_sufficient"]) for row in selected)
    deltas = [_number(row["delta_auroc"]) for row in selected]
    sufficient = sufficient and all(delta is not None for delta in deltas)
    return {
        "complete_three_folds": bool(complete),
        "evidence_sufficient": bool(sufficient),
        "folds": folds,
        "fold_deltas": deltas,
        "mean_delta": float(np.mean(deltas)) if sufficient else None,
        "minimum_fold_delta": float(min(deltas)) if sufficient else None,
        "maximum_fold_delta": float(max(deltas)) if sufficient else None,
        "positive_folds": sum(delta > 0 for delta in deltas) if sufficient else None,
    }


def _at_least(value, threshold):
    return value is not None and value >= threshold


def aggregate_screen(rows, scheme, plan):
    """Apply the declared primary gates, requiring each of folds 0/1/2 once."""
    kind = "topic" if scheme == "topic" else "random"
    if scheme not in ("topic", "random", "diagnostic_random", "random_diagnostic"):
        raise ValueError(f"Unknown screening scheme: {scheme}")
    limits = plan["screening_gates"][kind]
    summaries = {name: _condition_summary(rows, scheme, name) for name in PRIMARY_CONDITIONS}
    checks = {"all_conditions_have_sufficient_three_folds": all(row["evidence_sufficient"] for row in summaries.values())}
    clean, prefix = summaries["clean"], summaries["prefix_100"]
    checks.update({
        "clean_mean": _at_least(clean["mean_delta"], limits["clean_minimum_mean_delta"]),
        "clean_any_fold": _at_least(clean["minimum_fold_delta"], limits["clean_minimum_any_fold_delta"]),
        "prefix_mean": _at_least(prefix["mean_delta"], limits["prefix_minimum_mean_delta"]),
        "prefix_positive_folds": _at_least(prefix["positive_folds"], limits["prefix_minimum_positive_folds"]),
    })
    if "prefix_minimum_any_fold_delta" in limits:
        checks["prefix_any_fold"] = _at_least(prefix["minimum_fold_delta"], limits["prefix_minimum_any_fold_delta"])
    for name in ("quote_and_dash", "digit_runs_to_zero", "whitespace_only"):
        checks[f"{name}_mean"] = _at_least(summaries[name]["mean_delta"], limits["quote_digit_whitespace_minimum_each_mean_delta"])
        checks[f"{name}_any_fold"] = _at_least(summaries[name]["minimum_fold_delta"], limits["quote_digit_whitespace_minimum_any_fold_delta"])
    passed = all(checks.values())
    return {
        "scheme": scheme, "gate_family": kind, "passed": bool(passed),
        "status": "passed" if passed else "failed_or_insufficient_evidence",
        "conditions": summaries, "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "summary_definition": "Equal-weight mean of within-fold AUROC arm differences; never pooled raw margins.",
    }


def sealed_screen(rows, plan, schemes=("random", "topic")):
    """Apply suffix/middle non-regression gates in both valid split schemes."""
    limits = plan["screening_gates"]["sealed_conditions_after_primary_pass"]
    summaries, checks = {}, {}
    for scheme in schemes:
        summaries[scheme] = {}
        for name in SEALED_CONDITIONS:
            value = _condition_summary(rows, scheme, name)
            summaries[scheme][name] = value
            checks[f"{scheme}_{name}_sufficient"] = value["evidence_sufficient"]
            checks[f"{scheme}_{name}_mean"] = _at_least(value["mean_delta"], limits["suffix_middle_minimum_each_mean_delta_in_each_scheme"])
            checks[f"{scheme}_{name}_any_fold"] = _at_least(value["minimum_fold_delta"], limits["suffix_middle_minimum_any_fold_delta"])
    passed = all(checks.values())
    return {"passed": bool(passed), "status": "passed" if passed else "failed_or_insufficient_evidence",
            "conditions": summaries, "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "interpretation": limits["claim"]}


def _auc_cache(labels, scores):
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    groups = np.cumsum(np.r_[True, np.diff(ordered_scores) != 0]) - 1 if len(order) else np.array([], dtype=int)
    return order, groups.astype(np.int64), labels[order]


def _weighted_cached_auc(cache, weights):
    """AUC with nonnegative bootstrap multiplicities, including exact ties."""
    order, groups, labels = cache
    ordered_weights = weights[order]
    positive = np.bincount(groups, weights=ordered_weights * labels)
    negative = np.bincount(groups, weights=ordered_weights * (1 - labels))
    n_positive, n_negative = positive.sum(), negative.sum()
    if n_positive == 0 or n_negative == 0:
        return None
    concordance = np.dot(positive, np.cumsum(negative) - 0.5 * negative)
    return float(concordance / (n_positive * n_negative))


def bootstrap_cv(fold_records, scheme, plan, include_sealed=False):
    """Paired conditional macro-AUC intervals without fitting or pooling folds.

    A record is ``{fold, labels, conditions: {name: (indices, B0, B1)}}``.
    Topic records additionally require a topic array aligned with ``labels``.
    For ``scheme='validation'`` one full-data record is supported, using the
    declared validation seed; no fold-3 screening assumption is imposed here.
    """
    if not fold_records:
        raise ValueError("At least one held fold record is required")
    topic_mode = scheme == "topic"
    seed_key = "topic_bootstrap" if topic_mode else ("validation_bootstrap" if scheme == "validation" else "random_bootstrap")
    repetitions = int(plan["statistics"]["bootstrap_repetitions"])
    minimum_fraction = float(plan["statistics"]["minimum_valid_bootstrap_fraction_all_schemes"])
    names = list(PRIMARY_CONDITIONS) + (list(SEALED_CONDITIONS) if include_sealed else [])
    records = []
    for record in fold_records:
        labels = _checked_labels(record["labels"])
        prepared = {}
        for name in names:
            if name not in record["conditions"]:
                raise ValueError(f"Missing condition {name} in bootstrap fold {record['fold']}")
            indices, before, after = _checked_condition(record["conditions"][name], len(labels))
            prepared[name] = (indices, _auc_cache(labels[indices], before), _auc_cache(labels[indices], after))
        groups = None
        if topic_mode:
            topics = np.asarray(record["topics"])
            if topics.shape != labels.shape:
                raise ValueError("Topic assignments must align with held original IDs")
            unique, groups = np.unique(topics, return_inverse=True)
            n_topics = len(unique)
        else:
            n_topics = None
        records.append({"fold": int(record["fold"]), "labels": labels, "conditions": prepared,
                        "topic_index": groups, "n_topics": n_topics})
    if len(set(record["fold"] for record in records)) != len(records):
        raise ValueError("Bootstrap folds must be unique")
    rng = np.random.default_rng(plan["seeds"][seed_key])
    replicate_deltas = {name: [] for name in names}
    points = {name: [] for name in names}
    for record in records:
        for name in names:
            indices, before, after = record["conditions"][name]
            unit = np.ones(len(indices))
            a, b = _weighted_cached_auc(before, unit), _weighted_cached_auc(after, unit)
            points[name].append(b - a if a is not None and b is not None else None)
    for _ in range(repetitions):
        fold_deltas = {name: [] for name in names}
        for record in records:
            labels = record["labels"]
            if topic_mode:
                draws = rng.integers(0, record["n_topics"], size=record["n_topics"])
                multiplicity = np.bincount(draws, minlength=record["n_topics"])
                weights = multiplicity[record["topic_index"]]
            else:
                parts = [np.flatnonzero(labels == value) for value in (0, 1)]
                draw = np.concatenate([rng.choice(part, len(part), replace=True) for part in parts if len(part)])
                weights = np.bincount(draw, minlength=len(labels))
            # The same original-ID multiplicities go to both arms and all
            # conditions. Eligibility subsets do not get independent draws.
            for name in names:
                indices, before, after = record["conditions"][name]
                a = _weighted_cached_auc(before, weights[indices])
                b = _weighted_cached_auc(after, weights[indices])
                fold_deltas[name].append(b - a if a is not None and b is not None else None)
        for name in names:
            if all(value is not None for value in fold_deltas[name]):
                replicate_deltas[name].append(float(np.mean(fold_deltas[name])))
    intervals = {}
    for name in names:
        values = replicate_deltas[name]
        valid_fraction = len(values) / repetitions
        available = bool(values and valid_fraction >= minimum_fraction)
        low, high = np.quantile(values, [.025, .975]).tolist() if available else (None, None)
        intervals[name] = {
            "point_macro_delta_auroc": float(np.mean(points[name])) if all(value is not None for value in points[name]) else None,
            "valid_replicates": len(values), "invalid_replicates": repetitions - len(values),
            "valid_fraction": valid_fraction, "interval_available": available,
            "paired_delta_ci_2_5": low, "paired_delta_ci_97_5": high,
        }
    return {
        "scheme": scheme, "seed": int(plan["seeds"][seed_key]), "requested_replicates": repetitions,
        "n_folds": len(records), "conditions": intervals,
        "method": "Within-fold whole-topic paired bootstrap" if topic_mode else "Within-fold class-stratified original-ID paired bootstrap",
        "no_replacement_of_invalid_replicates": True,
        "limitations": plan["statistics"]["interval_limitations"],
    }


class EvaluationChecks(unittest.TestCase):
    """Synthetic mathematical/boundary regression tests, with zero fits."""

    def setUp(self):
        self.plan = json.loads((Path(__file__).with_name("q5_crop_plan.json")).read_text())
        self.plan["statistics"]["bootstrap_repetitions"] = 100

    def test_transformations_preserve_fixed_eligibility_and_windows(self):
        tokens = [f"word{i}" for i in range(201)]
        original = "\n  ".join(tokens)
        texts = [original, " ".join(tokens[:149]), "“Hi”—123 and ٤٥"]
        plain = make_conditions(texts)
        self.assertEqual(tuple(plain), PRIMARY_CONDITIONS)
        self.assertEqual(plain["clean"][1][0], original)
        changed = make_conditions(texts, include_sealed=True)
        for name in ("prefix_100", "suffix_100", "middle_100"):
            np.testing.assert_array_equal(changed[name][0], [0])
            self.assertEqual(len(changed[name][1][0].split()), 100)
        self.assertEqual(changed["middle_100"][1][0].split(), tokens[50:150])
        self.assertEqual(changed["suffix_100"][1][0].split(), tokens[-100:])
        self.assertEqual(changed["digit_runs_to_zero"][1][2], "“Hi”—0 and 0")
        self.assertEqual(changed["quote_and_dash"][1][2], '"Hi"-123 and ٤٥')

    def test_invalid_scores_or_indices_fail_and_small_groups_are_insufficient(self):
        y = np.array([0, 1])
        condition = {"clean": (np.arange(2), np.array([0., 1.]), np.array([0., 1.]))}
        row = evaluate_margins("random", 0, y, condition)[0]
        self.assertFalse(row["evidence_sufficient"])
        self.assertEqual(row["delta_auroc"], 0)
        with self.assertRaises(ValueError):
            evaluate_margins("random", 0, y, {"clean": (np.array([0, 0]), [0, 1], [0, 1])})
        with self.assertRaises(ValueError):
            evaluate_margins("random", 0, y, {"clean": (np.arange(2), [0, np.nan], [0, 1])})
        empty = evaluate_margins("random", 0, y, {"prefix_100": (np.array([], dtype=int), [], [])})[0]
        self.assertIsNone(empty["baseline_auroc"])

    def _passing_rows(self, schemes=("random", "topic"), sealed=False):
        return [{"scheme": scheme, "fold": fold, "condition": name, "evidence_sufficient": True,
                 "delta_auroc": .006 if name == "prefix_100" else 0.}
                for scheme in schemes for fold in range(3)
                for name in (PRIMARY_CONDITIONS + SEALED_CONDITIONS if sealed else PRIMARY_CONDITIONS)]

    def test_gates_require_unique_complete_folds_and_reject_bad_floor(self):
        rows = self._passing_rows()
        self.assertTrue(aggregate_screen(rows, "random", self.plan)["passed"])
        self.assertTrue(aggregate_screen(rows, "topic", self.plan)["passed"])
        self.assertFalse(aggregate_screen(rows + [rows[0]], "random", self.plan)["passed"])
        self.assertFalse(aggregate_screen(rows[1:], "random", self.plan)["passed"])
        changed = copy.deepcopy(rows)
        changed[0]["delta_auroc"] = -.007
        self.assertFalse(aggregate_screen(changed, "random", self.plan)["passed"])
        self.assertFalse(sealed_screen(rows, self.plan)["passed"])
        self.assertTrue(sealed_screen(self._passing_rows(sealed=True), self.plan)["passed"])

    def test_weighted_auc_matches_expanded_and_weighted_sklearn_with_ties(self):
        y = np.array([0, 1, 0, 1, 1])
        scores = np.array([.3, .3, .1, .8, .4])
        weights = np.array([2, 3, 0, 2, 1])
        cached = _weighted_cached_auc(_auc_cache(y, scores), weights)
        self.assertAlmostEqual(cached, roc_auc_score(y, scores, sample_weight=weights), places=14)
        expanded = np.repeat(np.arange(len(y)), weights)
        self.assertAlmostEqual(cached, roc_auc_score(y[expanded], scores[expanded]), places=14)
        self.assertIsNone(_weighted_cached_auc(_auc_cache(y, scores), np.array([1, 0, 2, 0, 0])))

    def _record(self, fold=0, topics=False):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0., 1., 2., 3.]) + 100 * fold
        conditions = {name: (np.arange(4), scores, scores) for name in PRIMARY_CONDITIONS}
        record = {"fold": fold, "labels": y, "conditions": conditions}
        if topics:
            record["topics"] = np.array([0, 0, 1, 1])
        return record

    def test_bootstrap_preserves_pairs_and_reproduces_shared_draws(self):
        records = [self._record(fold) for fold in range(3)]
        first = bootstrap_cv(records, "random", self.plan)
        self.assertEqual(first, bootstrap_cv(records, "random", self.plan))
        for item in first["conditions"].values():
            self.assertEqual(item["point_macro_delta_auroc"], 0.)
            self.assertEqual(item["valid_replicates"], 100)
            self.assertEqual(item["paired_delta_ci_2_5"], 0.)
            self.assertEqual(item["paired_delta_ci_97_5"], 0.)

    def test_topic_bootstrap_reports_invalid_without_drawing_until_success(self):
        result = bootstrap_cv([self._record(topics=True)], "topic", self.plan)
        for item in result["conditions"].values():
            self.assertEqual(item["valid_replicates"] + item["invalid_replicates"], 100)
            self.assertLess(item["valid_fraction"], .9)
            self.assertFalse(item["interval_available"])
            self.assertIsNone(item["paired_delta_ci_2_5"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__], verbosity=2)
    else:
        parser.print_help()
