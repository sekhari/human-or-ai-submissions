"""Audit existing submissions and observable train/test shift without fitting.

Run from the repository root with the recorded modeling Python::

    /opt/anaconda3/bin/python -m experiments.supervised.audit_public_predictions \
        --output results/supervised/a_public_gap_20261109_v1_predictions

Outputs contain aggregate statistics and file hashes, never passages or test
labels. Shift and subgroup results are exploratory diagnostics. They do not
estimate hidden-test AUROC or establish a cause for the public score gap.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score

from human_ai.check_predictions import check
from human_ai.data import ROOT, load_partitions, raw_path, read_csv, sha256
from .richness_features import RichnessFeatures


FEATURES = (
    "word_count", "character_count", "newline_rate", "curly_quote_rate",
    "ascii_quote_rate", "em_dash_rate", "digit_rate", "nonascii_rate",
    "uppercase_rate", "mattr50", "normalized_entropy", "bigram_repetition",
)
RUNS = {
    "word_original": "a_q1_20261107_v1_svm",
    "word_expanded": "a_q1_expanded_20261108_v2_svm",
    "wc120": "a_q5_20261108_v1_wc120",
    "wc120_quotes": "a_q5_quotes_20261108_v3_wc120_norm",
    "c60_richness": "a_q5_richness_20261108_v4_c60_richness",
}


def extract(records):
    richness = RichnessFeatures().transform([r["text"] for r in records])
    values = []
    for row, lexical in zip(records, richness, strict=True):
        text = row["text"]
        denominator = max(len(text), 1)
        values.append([
            len(text.split()), len(text), text.count("\n") / denominator,
            sum(text.count(c) for c in "“”‘’") / denominator,
            sum(text.count(c) for c in "\"'") / denominator,
            text.count("—") / denominator,
            sum(c.isdigit() for c in text) / denominator,
            sum(ord(c) > 127 for c in text) / denominator,
            sum(c.isupper() for c in text) / denominator,
            *lexical,
        ])
    return np.asarray(values, dtype=np.float64)


def normalized_texts(records):
    return [re.sub(r"\s+", " ", r["text"]).strip().lower() for r in records]


def aligned(path, records, column):
    predictions = {r["id"]: r for r in read_csv(path)}
    return np.asarray([float(predictions[r["id"]][column]) for r in records])


def summary(values):
    return {
        "n": len(values), "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "quantiles_0_10_50_90_100": np.quantile(values, [0, .1, .5, .9, 1]).tolist(),
        "fraction_positive": float(np.mean(values > 0)),
    }


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new audit directory: {output}")
    records = {name: read_csv(raw_path(name + ".csv")) for name in ("train", "test", "unlabeled")}
    fitting, validation = load_partitions()
    labels = np.asarray([int(r["label"]) for r in records["train"]])
    validation_labels = np.asarray([int(r["label"]) for r in validation])
    data = {name: extract(rows) for name, rows in records.items()}
    fit_data, validation_data = extract(fitting), extract(validation)
    hashes = {str(raw_path(name + ".csv").relative_to(ROOT)): sha256(raw_path(name + ".csv")) for name in records}
    normalized = {name: normalized_texts(rows) for name, rows in records.items()}
    integrity = {
        "data": {name: {"n": len(values), "empty": sum(not x for x in values),
                         "duplicate_extras_after_lowercase_whitespace_normalization": len(values) - len(set(values))}
                 for name, values in normalized.items()},
        "train_test_normalized_exact_overlap": len(set(normalized["train"]) & set(normalized["test"])),
        "train_unlabeled_normalized_exact_overlap": len(set(normalized["train"]) & set(normalized["unlabeled"])),
        "submissions": {},
    }
    predictions, held_predictions = {}, {}
    for key, run_id in RUNS.items():
        submission = ROOT / "submissions" / (run_id + "_full10000.csv")
        held = ROOT / "results" / "supervised" / run_id / "validation_predictions.csv"
        integrity["submissions"][key] = {"valid_rows": check(submission, "submission"),
                                           "valid_validation_rows": check(held, "validation")}
        for path in (submission, held):
            hashes[str(path.relative_to(ROOT))] = sha256(path)
        predictions[key] = aligned(submission, records["test"], "label")
        held_predictions[key] = aligned(held, validation, "p_ai")

    feature_results = {}
    for i, name in enumerate(FEATURES):
        human, ai = data["train"][labels == 0, i], data["train"][labels == 1, i]
        test_values = data["test"][:, i]
        nearest_class = min((human, ai), key=lambda values: abs(values.mean() - test_values.mean()))
        standard_error = np.sqrt(np.var(nearest_class, ddof=1) / len(nearest_class)
                                 + np.var(test_values, ddof=1) / len(test_values))
        feature_results[name] = {
            **{dataset: summary(values[:, i]) for dataset, values in data.items()},
            "human_train": summary(human), "ai_train": summary(ai),
            "training_class_auroc_raw_direction": float(roc_auc_score(labels, data["train"][:, i])),
            "train_test_ks_statistic": float(ks_2samp(data["train"][:, i], test_values).statistic),
            "train_unlabeled_ks_statistic": float(ks_2samp(data["train"][:, i], data["unlabeled"][:, i]).statistic),
            "test_mean_outside_train_class_mean_range": bool(test_values.mean() < min(human.mean(), ai.mean())
                                                            or test_values.mean() > max(human.mean(), ai.mean())),
            "test_minus_nearest_train_class_mean_over_independent_mean_se":
                float((test_values.mean() - nearest_class.mean()) / standard_error) if standard_error else None,
        }

    prediction_results = {}
    for key, values in predictions.items():
        prediction_results[key] = {
            **summary(values),
            "fraction_at_least_half": float(np.mean(values >= .5)),
            "test_spearman_vs_wc120": float(spearmanr(predictions["wc120"], values).statistic),
            "test_half_threshold_disagreements_vs_wc120": int(np.sum((values >= .5) != (predictions["wc120"] >= .5))),
            "validation_auroc_source_8000_model": float(roc_auc_score(validation_labels, held_predictions[key])),
            "validation_spearman_vs_wc120": float(spearmanr(held_predictions["wc120"], held_predictions[key]).statistic),
        }
    integrity["word_original_and_expanded_max_abs_prediction_difference"] = float(np.max(abs(predictions["word_original"] - predictions["word_expanded"])))

    # Boundaries use only the canonical 8,000-row fitting partition. Repeated
    # boundaries can produce empty bins; report rather than manufacture groups.
    subgroups = []
    for i in (0, 2, 6, 9, 10, 11):
        boundaries = np.unique(np.quantile(fit_data[:, i], [.25, .5, .75]))
        held_bins = np.digitize(validation_data[:, i], boundaries, right=True)
        test_bins = np.digitize(data["test"][:, i], boundaries, right=True)
        for bin_number in range(len(boundaries) + 1):
            mask = held_bins == bin_number
            y = validation_labels[mask]
            aucs = {key: float(roc_auc_score(y, values[mask])) if len(set(y)) == 2 else None
                    for key, values in held_predictions.items()}
            subgroups.append({
                "feature": FEATURES[i], "training_quartile_boundaries": boundaries.tolist(),
                "bin": bin_number, "boundary_convention": "np.digitize(right=True)",
                "validation_n": int(mask.sum()), "validation_ai_n": int(y.sum()),
                "test_n": int(np.sum(test_bins == bin_number)),
                "validation_auroc": aucs,
            })

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "source_sha256": {"experiments/supervised/audit_public_predictions.py": sha256(__file__)},
        "input_sha256": hashes, "models_fitted": 0,
        "limitations": [
            "No test labels are available; all test-set statistics are unlabeled diagnostics.",
            "Feature shift does not establish which shift caused the score gap.",
            "The reused validation set is a development set and subgroup comparisons are exploratory.",
            "A test mean outside both training class means is inconsistent with exact population-level pure label shift for that feature; sampling uncertainty and source composition remain.",
            "Approximate mean standard errors assume independent observations and omit multiple-comparison adjustment.",
            "Predicted AI fractions do not estimate the true test class balance.",
        ],
    }
    output.mkdir(parents=True)
    for name, value in (("metadata", metadata), ("integrity", integrity), ("feature_shift", feature_results),
                        ("prediction_comparison", prediction_results), ("validation_subgroups", subgroups)):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(f"Saved audit to {output}; no model fits, refits or uploads")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
