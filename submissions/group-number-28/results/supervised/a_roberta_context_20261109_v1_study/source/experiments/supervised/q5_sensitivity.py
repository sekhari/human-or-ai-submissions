"""Measure paired typography sensitivity of already fitted Q5 models.

The transformations are evaluation diagnostics, not a training intervention or
evidence that the hidden test distribution contains these particular changes.
"""
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import roc_auc_score

from human_ai.data import sha256
from .run_q1 import predict_ai
from .run_q5 import atomic_csv, atomic_json


CONDITIONS = ("original", "quote_typography", "dash_typography")
QUOTE_TRANSLATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
})
DASH_TRANSLATION = str.maketrans({"\u2013": "-", "\u2014": "-"})


def apply_condition(text, condition):
    """Apply exactly one frozen character mapping without changing case/words."""
    if condition == "original":
        return text
    if condition == "quote_typography":
        return text.translate(QUOTE_TRANSLATION)
    if condition == "dash_typography":
        return text.translate(DASH_TRANSLATION)
    raise ValueError(f"Unknown typography condition: {condition}")


def _probabilities(model, documents):
    probabilities = np.asarray(predict_ai(model, documents), dtype=float)
    if probabilities.shape != (len(documents),):
        raise ValueError("Expected one class-1 probability per document")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid predicted probabilities")
    return probabilities


def evaluate_sensitivity(study, models, validation, plan):
    """Write paired predictions and diagnostics without fitting any estimator.

``models`` maps safe artifact names to fitted calibrated classifiers. Every
condition retains the original ID order and true labels. AUROC for a changed
subset is undefined when that subset contains fewer than two true classes.
"""
    study = Path(study)
    output = study / "format_predictions"
    output.mkdir(parents=True, exist_ok=True)
    text = [row["text"] for row in validation]
    labels = np.asarray([int(row["label"]) for row in validation])
    ids = [str(row["id"]) for row in validation]
    if len(ids) != len(set(ids)) or set(labels.tolist()) != {0, 1}:
        raise ValueError("Sensitivity data must contain unique IDs and both binary labels")
    threshold = float(plan.get("evaluation", {}).get("error_threshold", 0.5))
    conditions = {name: [apply_condition(document, name) for document in text]
                  for name in CONDITIONS}
    rows, hashes, grouped = [], {}, {}
    for name, model in models.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Unsafe model artifact name: {name}")
        baseline = _probabilities(model, text)
        baseline_auc = float(roc_auc_score(labels, baseline))
        grouped[name] = {}
        for condition, changed_text in conditions.items():
            probabilities = baseline if condition == "original" else _probabilities(model, changed_text)
            changed = np.asarray([before != after for before, after in zip(text, changed_text, strict=True)])
            n_changed = int(changed.sum())
            changed_has_both_labels = n_changed > 0 and len(set(labels[changed].tolist())) == 2
            subset_auc = (float(roc_auc_score(labels[changed], probabilities[changed]))
                          if changed_has_both_labels else None)
            subset_delta = (subset_auc - float(roc_auc_score(labels[changed], baseline[changed]))
                            if changed_has_both_labels else None)
            auc = float(roc_auc_score(labels, probabilities))
            row = {
                "model": name, "condition": condition, "n": len(validation),
                "n_changed": n_changed, "changed_fraction": n_changed / len(validation),
                "auroc": auc, "delta_auroc": auc - baseline_auc,
                "mean_abs_probability_change": float(np.mean(np.abs(probabilities - baseline))),
                "flip_rate_at_0_5": float(np.mean((probabilities >= threshold) != (baseline >= threshold))),
                "changed_subset_auroc": subset_auc, "changed_subset_delta_auroc": subset_delta,
                "n_changed_human": int(np.sum(changed & (labels == 0))),
                "n_changed_ai": int(np.sum(changed & (labels == 1))),
            }
            rows.append(row)
            grouped[name][condition] = row
            path = output / f"{name}_{condition}.csv"
            atomic_csv(path, ["id", "label", "p_ai"],
                       ({"id": identifier, "label": str(int(label)), "p_ai": format(float(probability), ".17g")}
                        for identifier, label, probability in zip(ids, labels, probabilities, strict=True)))
            hashes[str(path.relative_to(study))] = sha256(path)
    if not rows:
        raise ValueError("At least one fitted model is required")
    atomic_csv(study / "format_sensitivity.csv", list(rows[0]), rows)
    result = {
        "conditions": list(CONDITIONS), "threshold": threshold,
        "n_validation": len(validation), "rows": rows, "models": grouped,
        "prediction_sha256": hashes,
        "paired_by_original_id": True, "new_model_fits_performed": False,
        "limitations": "Evaluation-only typography sensitivity, not a normalized training intervention or proof of hidden-domain robustness. Variants are dependent observations.",
    }
    atomic_json(study / "format_sensitivity.json", result)
    return result
