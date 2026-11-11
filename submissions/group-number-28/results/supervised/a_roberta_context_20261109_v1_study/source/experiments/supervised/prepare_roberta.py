"""Freeze one local encoder comparison without fitting any model.

The saved canonical WC120 model is used for inference only, in its historical
environment. Neural fitting is performed separately in python_course.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from human_ai.data import load_partitions, sha256, write_csv
from .crop_evaluation import make_conditions, PRIMARY_CONDITIONS

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "results/supervised/a_q5_crop_revised_20261109_v6_study"
MODEL = ROOT / "results/supervised/a_q5_20261108_v1_wc120/model.joblib"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(study):
    study.mkdir(parents=True, exist_ok=False)
    (study / "ids").mkdir()
    (study / "baselines").mkdir()
    (study / "frozen_inputs").mkdir()
    train, held = load_partitions()
    train_ids = {row["id"] for row in train}
    labels = {r["id"]: int(r["label"]) for r in train + held}
    input_hashes = {}
    def pin(path):
        input_hashes[str(path.relative_to(ROOT))] = sha256(path)
    for path in [ROOT / "data/splits/manifest.json", ROOT / "data/splits/train_ids.csv",
                 ROOT / "data/splits/validation_ids.csv", MODEL]:
        pin(path)
    for name in ["random_folds.csv", "topic_folds.csv", "topic_assignments.csv", "fold_counts.csv"]:
        pin(PARENT / name)
        shutil.copy2(PARENT / name, study / "frozen_inputs" / name)
    fitting_specs = []
    for scheme in ["random", "topic"]:
        fold_rows = read_csv(PARENT / f"{scheme}_folds.csv")
        assignment = {row["id"]: int(row["held_fold"]) for row in fold_rows}
        if len(assignment) != 8000 or set(assignment) != train_ids:
            raise ValueError("Prior fold IDs differ from canonical fitting IDs")
        for fold in range(3):
            key = f"{scheme}_{fold}"
            fitting = [r for r in train if assignment[r["id"]] != fold]
            testing = [r for r in train if assignment[r["id"]] == fold]
            source = PARENT / "folds" / key / "primary_predictions.csv"
            pin(source)
            cached = read_csv(source)
            expected = {(name, testing[i]["id"]) for name, (indices, _) in
                        make_conditions([r["text"] for r in testing]).items() for i in indices}
            keys = [(r["condition"], r["id"]) for r in cached]
            if len(set(keys)) != len(keys) or set(keys) != expected:
                raise ValueError(f"Prior baseline coverage differs: {key}")
            baseline = []
            for r in cached:
                if int(r["label"]) != labels[r["id"]]:
                    raise ValueError("Baseline label mismatch")
                baseline.append({"id": r["id"], "label": r["label"], "condition": r["condition"],
                                 "baseline_score": r["baseline_margin_not_probability"]})
            write_csv(study / "baselines" / f"{key}.csv", ["id", "label", "condition", "baseline_score"], baseline)
            fitting_specs.append((key, scheme, fold, fitting, testing, "raw_svm_margin"))

    # Zero fitting: the canonical reference already exists and has frozen labels.
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        model = joblib.load(MODEL)
        if list(model.classes_) != [0, 1]:
            raise ValueError("Unexpected baseline class orientation")
        canonical = []
        for name, (indices, texts) in make_conditions([r["text"] for r in held]).items():
            scores = model.predict_proba(texts)[:, 1]
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite saved-model inference")
            for i, score in zip(indices, scores):
                canonical.append({"id": held[i]["id"], "label": held[i]["label"], "condition": name,
                                  "baseline_score": format(float(score), ".17g")})
    clean_path = ROOT / "results/supervised/a_q5_20261108_v1_wc120/validation_predictions.csv"
    stress_path = ROOT / "results/supervised/a_public_gap_20261109_v1_stress/predictions.csv"
    pin(clean_path)
    pin(stress_path)
    expected = {("clean", r["id"]): float(r["p_ai"]) for r in read_csv(clean_path)}
    for r in read_csv(stress_path):
        if r["model"] == "wc120":
            name = "prefix_100" if r["condition"] == "prefix_100_on_min150" else r["condition"]
            if name in PRIMARY_CONDITIONS:
                expected[(name, r["id"])] = float(r["changed_p_ai"])
    errors = [abs(float(r["baseline_score"]) - expected[(r["condition"], r["id"])])
              for r in canonical if (r["condition"], r["id"]) in expected]
    error = max(errors)
    if error > 1e-10:
        raise ValueError(f"Saved baseline differs from previous prediction cache: {error}")
    write_csv(study / "baselines/canonical_8000.csv", ["id", "label", "condition", "baseline_score"], canonical)
    save(study / "baseline_inference_receipt.json", {
        "new_model_fits": 0, "model_sha256": sha256(MODEL), "python": sys.executable,
        "elapsed_seconds": time.perf_counter() - started, "prediction_rows": len(canonical),
        "matched_historical_prediction_rows": len(errors), "maximum_historical_error": error,
        "probability_orientation": "P(AI=1)", "fit_validation_or_test_inputs_used": False,
    })
    fitting_specs.append(("canonical_8000", "validation", 0, train, held, "calibrated_probability"))
    fits = []
    generated_hashes = {}
    for key, scheme, fold, fitting, testing, score_type in fitting_specs:
        id_files = {}
        for part, records in [("train", fitting), ("held", testing)]:
            path = study / "ids" / f"{key}_{part}.csv"
            write_csv(path, ["id"], [{"id": r["id"]} for r in records])
            id_files[f"{part}_ids_file"] = str(path.relative_to(study))
            generated_hashes[str(path.relative_to(study))] = sha256(path)
        baseline_path = study / "baselines" / f"{key}.csv"
        generated_hashes[str(baseline_path.relative_to(study))] = sha256(baseline_path)
        cuts = np.quantile([len(r["text"].split()) for r in fitting], [1 / 3, 2 / 3], method="linear").tolist()
        fits.append({"key": key, "scheme": scheme, "fold": fold, **id_files,
                     "training_n": len(fitting), "held_n": len(testing),
                     "baseline_predictions_file": str(baseline_path.relative_to(study)),
                     "baseline_score_type": score_type, "length_word_tertiles": cuts,
                     "expected_optimizer_updates": 2 * ((len(fitting) + 15) // 16)})
    download = json.loads((ROOT / "results/supervised/a_roberta_mps_20261109_v1/model_download.json").read_text())
    plan = {
        "protocol_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "One fixed pretrained contextual candidate versus WC120, complete matched random/topic comparison and reused canonical development evaluation",
        "status_at_freeze": "Before any scientific RoBERTa fit or held-model prediction",
        "hypothesis": "A supplied-label fine-tuned contextual encoder may improve held-topic ranking and repair WC120 errors; hidden-test improvement is unknown",
        "comparison_scope": "Model-family comparison, not an isolated causal effect of pretraining or architecture",
        "python_executable": "/opt/anaconda3/envs/python_course/bin/python",
        "model": {"name": download["repo_id"], "snapshot": download["snapshot_path"],
                  "revision": download["resolved_revision"], "weights_sha256": download["file_sha256"]["model.safetensors"],
                  "weights_use": "Generic base checkpoint; never device-test weights", "license": "MIT (pinned model card)"},
        "recipe": {"epochs": 2, "microbatch": 4, "gradient_accumulation": 4, "max_length": 512,
                   "learning_rate": 2e-5, "weight_decay": 0.01, "max_grad_norm": 1.0,
                   "seed": 42, "attention": "sdpa", "dtype": "float32", "lr_schedule": "constant",
                   "optimizer": "AdamW", "foreach": False, "fused": False, "padding": "max_length",
                   "weight_decay_scope": "All trainable parameters, including bias/normalization, as in the device test",
                   "partial_accumulation": "Retain every original; weight microbatch mean loss by microbatch_n/actual_effectivebatch_n",
                   "checkpoint_selection": "Final prescribed epoch only; no held-loss evaluation or early stopping",
                   "cpu_threads": 10},
        "fits": fits,
        "conditions": list(PRIMARY_CONDITIONS), "include_sealed": False,
        "prefix_eligibility": "Original whitespace word count >=150; first100 words joined with one ASCII space",
        "tokenizer_coverage": "Train coverage before fitting, held coverage only after final epoch; full token length includes specials; record truncation at512",
        "ranking_score": "RoBERTa raw class1 minus class0 logits, to avoid FP32 probability saturation ties",
        "statistics": {"primary": "Equal-weight mean of three within-fold clean AUROCs per scheme and paired deltas against identical-ID WC120 predictions",
                       "secondary": ["prefix100 and other fixed sensitivity conditions", "per-fold ranking repairs and regressions with ties at0.5", "rank correlation", "natural-length and512truncation subgroups"],
                       "bootstrap_repetitions": 1000, "minimum_valid_bootstrap_fraction_all_schemes": 0.9,
                       "interval_scope": "Conditional on fitted models, fixed/adaptively reused splits; no training variation, selection or multiplicity correction",
                       "topic_interval": "Resample whole frozen lexical topic clusters within each fixed held fold; record undefined replicates without redrawing",
                       "pooling": "Never pool cross-fold raw margins as primary AUROC"},
        "seeds": {"model": 42, "random_bootstrap": 548, "topic_bootstrap": 549, "validation_bootstrap": 550},
        "boundaries": {"canonical_split_changes": False, "external_labelled_data": False,
                       "unlabeled_or_test_learning": False, "held_evaluation_before_final_epoch": False,
                       "held_set_is_untouched": False, "topic_proxy_is_true_domain_label": False,
                       "blend_fitting_or_export": False, "automatic_submission": False},
        "completion_policy": "Finish all7 fixed fits regardless of quality. Stop only for data, numeric, hardware or software failures; preserve failed attempts and resume identical unfinished work. No score thresholds or architecture/seed/epoch/weight search.",
        "input_files_sha256": input_hashes, "prepared_files_sha256": generated_hashes,
        "preparer_source_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
    }
    save(study / "plan.json", plan)
    save(study / "preparation_receipt.json", {"plan_sha256": sha256(study / "plan.json"), "fits": len(fits),
                                            "new_model_fits": 0, "canonical_training_n": len(train),
                                            "canonical_validation_n": len(held), "held_test_predictions_generated": False,
                                            "baseline_only_canonical_inference": True})
    print(json.dumps({"study": str(study), "plan_sha256": sha256(study / "plan.json"),
                      "new_fits": 0, "planned_neural_fits": len(fits), "baseline_reload_error": error}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.study_dir.resolve())
