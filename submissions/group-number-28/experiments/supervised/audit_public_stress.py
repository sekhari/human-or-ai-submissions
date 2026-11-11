"""Frozen, no-refit stress diagnostics after user-reported public scores.

No model selection, fitting, test-label retrieval, or candidate export. Raw
passages never appear in the generated artifacts. These constructed conditions
are authorship-preservation hypotheses, not verified simulations of Kaggle data.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from human_ai.data import ROOT, load_partitions, read_csv, sha256, write_csv
from experiments.supervised.q5_sensitivity import apply_condition
from experiments.supervised.run_q1 import predict_ai


MODELS = {
    "wc120": "a_q5_20261108_v1_wc120",
    "c60": "a_q5_20261108_v1_c60",
    "c60_richness": "a_q5_richness_20261108_v4_c60_richness",
    "wc120_quotes": "a_q5_quotes_20261108_v3_wc120_norm",
}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def delta_interval(y, before, after, draws):
    differences = [roc_auc_score(y[index], after[index]) - roc_auc_score(y[index], before[index])
                   for index in draws]
    return np.quantile(differences, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/supervised/a_public_gap_20261109_v1_stress")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    _, validation = load_partitions()
    labels = np.array([int(row["label"]) for row in validation])
    text = [row["text"] for row in validation]
    ids = [row["id"] for row in validation]
    long_indices = np.array([i for i, passage in enumerate(text) if len(passage.split()) >= 150])
    model_paths = {name: ROOT / "results/supervised" / run_id / "model.joblib"
                   for name, run_id in MODELS.items()}
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "fit_performed": False,
        "conditions": {
            "quote_and_dash": "Apply existing frozen quote_typography then dash_typography mappings together; full canonical validation IDs.",
            "digit_runs_to_zero": "Python Unicode-aware re.sub(r'\\d+', '0', passage); full canonical validation IDs.",
            "prefix_100_on_min150": "Keep first 100 whitespace-split words joined by one ASCII space; compare with unchanged original text on exactly canonical IDs having >=150 whitespace words.",
        },
        "authorship_preservation_hypotheses": [
            "Quote/dash mapping changes typography while preserving authorship and most content.",
            "Digit masking preserves original authorship but removes potentially meaningful numerical content; a stress diagnostic, not realistic attack evidence.",
            "A prefix retains original authorship but removes content and changes length/whitespace; not a pure length intervention or proven hidden-test attack.",
        ],
        "models": {name: {"run_id": MODELS[name], "model_sha256": sha256(path),
                           "reference_predictions_sha256": sha256(path.parent / "validation_predictions.csv")}
                   for name, path in model_paths.items()},
        "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(__file__),
                          "experiments/supervised/q5_sensitivity.py": sha256(ROOT / "experiments/supervised/q5_sensitivity.py"),
                          "data/splits/manifest.json": sha256(ROOT / "data/splits/manifest.json")},
        "n_validation": len(validation), "n_prefix_subset": len(long_indices),
        "bootstrap": {"repetitions": 500, "seed": 20261109,
                      "method": "Paired, class-stratified percentile bootstrap of original document IDs; fixed models/split, no selection or training-variation correction."},
        "limitations": "Post-public-feedback exploratory diagnostics. No observed condition identifies hidden domains, attacks, or causes of the public gap. No candidate selection or export.",
    }
    # This immutable plan is persisted before models are loaded or scored.
    save_json(output / "plan.json", plan)
    write_csv(output / "prefix_subset_ids.csv", ["id"], ({"id": ids[i]} for i in long_indices))
    conditions = {
        "quote_and_dash": ([apply_condition(apply_condition(t, "quote_typography"), "dash_typography") for t in text], np.arange(len(text))),
        "digit_runs_to_zero": ([re.sub(r"\d+", "0", t) for t in text], np.arange(len(text))),
        "prefix_100_on_min150": ([" ".join(text[i].split()[:100]) for i in long_indices], long_indices),
    }
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    bootstrap = {}
    for key, indices in (("all", np.arange(len(text))), ("prefix", long_indices)):
        y = labels[indices]
        zero, one = np.where(y == 0)[0], np.where(y == 1)[0]
        bootstrap[key] = [np.concatenate((rng.choice(zero, len(zero), replace=True), rng.choice(one, len(one), replace=True)))
                          for _ in range(500)]
    rows, agreement, prediction_rows = [], {}, []
    with threadpool_limits(limits=1):
        for name, path in model_paths.items():
            print(f"Scoring {name}", flush=True)
            model = joblib.load(path)
            before_full = np.asarray(predict_ai(model, text))
            reference_by_id = {row["id"]: float(row["p_ai"]) for row in read_csv(path.parent / "validation_predictions.csv")}
            maximum_error = float(np.max(np.abs(before_full - np.array([reference_by_id[identifier] for identifier in ids]))))
            if maximum_error > 1e-12:
                raise ValueError(f"Saved model reference mismatch: {name}: {maximum_error}")
            agreement[name] = {"maximum_absolute_error_vs_saved_validation": maximum_error}
            for condition, (changed_text, indices) in conditions.items():
                before, y = before_full[indices], labels[indices]
                after = np.asarray(predict_ai(model, changed_text))
                difference = after - before
                low, high = delta_interval(y, before, after, bootstrap["prefix" if condition == "prefix_100_on_min150" else "all"])
                row = {"model": name, "condition": condition, "n": len(indices),
                       "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
                       "n_text_changed": sum(changed != text[i] for changed, i in zip(changed_text, indices)),
                       "original_auroc": float(roc_auc_score(y, before)),
                       "changed_auroc": float(roc_auc_score(y, after)),
                       "delta_auroc": float(roc_auc_score(y, after) - roc_auc_score(y, before)),
                       "delta_ci_2_5": low, "delta_ci_97_5": high,
                       "score_spearman": float(spearmanr(before, after).statistic),
                       "mean_absolute_probability_change": float(np.mean(np.abs(difference))),
                       "maximum_absolute_probability_change": float(np.max(np.abs(difference))),
                       "n_threshold_0_5_flips": int(np.sum((before >= 0.5) != (after >= 0.5))),
                       "flip_fraction": float(np.mean((before >= 0.5) != (after >= 0.5)))}
                rows.append(row)
                prediction_rows.extend({"id": ids[i], "label": int(y[j]), "model": name,
                                        "condition": condition, "original_p_ai": float(before[j]),
                                        "changed_p_ai": float(after[j])} for j, i in enumerate(indices))
                print(f"  {condition}: {row['original_auroc']:.6f} -> {row['changed_auroc']:.6f}", flush=True)
    write_csv(output / "summary.csv", list(rows[0]), rows)
    write_csv(output / "predictions.csv", list(prediction_rows[0]), prediction_rows)
    save_json(output / "results.json", {"plan_sha256": sha256(output / "plan.json"),
                                       "reference_agreement": agreement, "rows": rows,
                                       "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                       "artifact_sha256": {p.name: sha256(p) for p in output.glob("*.csv")}})


if __name__ == "__main__":
    main()
