"""Independently inspect a RoBERTa study without importing Torch or fitting.

Checks saved boundaries, training receipts and held-prediction arithmetic. This
is an artifact audit, not a fresh fit, an operator profiler, or a new holdout.
"""
import argparse
from bisect import bisect_left, bisect_right
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from human_ai.data import load_partitions


PRIOR = ROOT / "results/supervised/a_q5_crop_revised_20261109_v6_study"
CONDITIONS = ("clean", "quote_and_dash", "digit_runs_to_zero", "whitespace_only", "prefix_100")
NATIVE_FALLBACK = re.compile(
    r"not currently supported on[^\n]*mps|will fall back[^\n]*cpu|"
    r"mps[^\n]*(?:fallback|fall back|falling back)[^\n]*cpu", re.IGNORECASE,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing/duplicate CSV columns: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return rows


def resolve(study, relative):
    path = Path(relative)
    return path if path.is_absolute() else study / path


def ids_digest(ids):
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def auc(labels, scores):
    """Mann-Whitney AUROC with exact half-credit for tied scores."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("Aligned nonempty labels/scores are required")
    if any(label not in (0, 1) for label in labels) or any(not math.isfinite(value) for value in scores):
        raise ValueError("Binary labels and finite scores are required")
    negatives = sorted(score for label, score in zip(labels, scores) if label == 0)
    positives = [score for label, score in zip(labels, scores) if label == 1]
    if not negatives or not positives:
        return None
    credit = sum((bisect_left(negatives, score) + bisect_right(negatives, score)) / 2
                 for score in positives)
    return credit / (len(negatives) * len(positives))


class Checks:
    def __init__(self):
        self.passed = 0
        self.errors = []

    def require(self, condition, message):
        if condition:
            self.passed += 1
        else:
            self.errors.append(message)

    def close(self, actual, expected, message, tolerance=1e-9):
        self.require(actual is not None and expected is not None
                     and math.isfinite(float(actual)) and math.isfinite(float(expected))
                     and math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance), message)


def verify_boundaries(study, fit, train_records, validation_records, checks):
    train = [row["id"] for row in read_csv(resolve(study, fit["train_ids_file"]))]
    held = [row["id"] for row in read_csv(resolve(study, fit["held_ids_file"]))]
    canonical_train = {row["id"] for row in train_records}
    canonical_validation = {row["id"] for row in validation_records}
    checks.require(len(train) == len(set(train)) and len(held) == len(set(held)), f"{fit['key']}: duplicate IDs")
    checks.require(not set(train) & set(held), f"{fit['key']}: train/held overlap")
    if fit["scheme"] == "validation":
        checks.require(set(train) == canonical_train and set(held) == canonical_validation,
                       f"{fit['key']}: canonical split differs")
    else:
        prior = read_csv(PRIOR / f"{fit['scheme']}_folds.csv")
        expected_held = {row["id"] for row in prior if int(row["held_fold"]) == fit["fold"]}
        checks.require(set(held) == expected_held and set(train) == canonical_train - expected_held,
                       f"{fit['key']}: differs from preserved prior fold memberships")
        checks.require(not (set(train) | set(held)) & canonical_validation,
                       f"{fit['key']}: canonical validation appears in CV")
    return train, held


def verify_prepared(study, plan, fits, train_records, validation_records, checks):
    """Independently check preparation before any neural training is required."""
    receipt = read_json(study / "preparation_receipt.json")
    checks.require(receipt["plan_sha256"] == sha256(study / "plan.json")
                   and receipt["new_model_fits"] == 0 and receipt["fits"] == 7,
                   "Preparation receipt/protocol hash differs")
    for name, expected in plan["input_files_sha256"].items():
        checks.require(sha256(ROOT / name) == expected, f"Prior input hash differs: {name}")
    for name, expected in plan["prepared_files_sha256"].items():
        checks.require(sha256(study / name) == expected, f"Prepared input hash differs: {name}")
    checks.require(plan["conditions"] == list(CONDITIONS) and plan["include_sealed"] is False,
                   "Unexpected condition set or previously sealed conditions opened")
    checks.require(plan["boundaries"]["blend_fitting_or_export"] is False
                   and plan["boundaries"]["automatic_submission"] is False
                   and plan["boundaries"]["held_evaluation_before_final_epoch"] is False,
                   "Unexpected fitting/evaluation/export boundary")
    by_id = {row["id"]: row for row in train_records + validation_records}
    boundaries = {}
    for fit in fits:
        train, held = verify_boundaries(study, fit, train_records, validation_records, checks)
        boundaries[fit["key"]] = (train, held)
        checks.require(fit["training_n"] == len(train) and fit["held_n"] == len(held)
                       and fit["expected_optimizer_updates"] == 2 * math.ceil(len(train) / 16),
                       f"{fit['key']}: fit count/update arithmetic differs")
        rows = read_csv(resolve(study, fit["baseline_predictions_file"]))
        observed = {(row["condition"], row["id"]): row for row in rows}
        expected = {(condition, identifier) for condition in CONDITIONS for identifier in held
                    if condition != "prefix_100" or len(by_id[identifier]["text"].split()) >= 150}
        checks.require(len(rows) == len(observed) and set(observed) == expected,
                       f"{fit['key']}: prepared baseline condition/ID coverage differs")
        checks.require(all(row["id"] in by_id and row["label"] == by_id[row["id"]]["label"]
                           and math.isfinite(float(row["baseline_score"])) for row in rows),
                       f"{fit['key']}: prepared baseline labels/scores differ")
        if fit["scheme"] != "validation":
            prior_rows = read_csv(PRIOR / "folds" / fit["key"] / "primary_predictions.csv")
            prior = {(row["condition"], row["id"]): row for row in prior_rows}
            checks.require(set(prior) == expected and all(
                observed[key]["label"] == prior[key]["label"]
                and float(observed[key]["baseline_score"]) == float(prior[key]["baseline_margin_not_probability"])
                for key in expected), f"{fit['key']}: prepared CV baseline scores differ from preserved WC120")
        else:
            prior = {row["id"]: row for row in read_csv(ROOT / "results/supervised/a_q5_20261108_v1_wc120/validation_predictions.csv")}
            checks.require(all(float(observed[("clean", identifier)]["baseline_score"]) == float(prior[identifier]["p_ai"])
                               for identifier in held), "Canonical clean WC120 probabilities differ")
            baseline_receipt = read_json(study / "baseline_inference_receipt.json")
            checks.require(baseline_receipt["new_model_fits"] == 0
                           and baseline_receipt["maximum_historical_error"] <= 1e-10
                           and baseline_receipt["probability_orientation"] == "P(AI=1)",
                           "Canonical baseline inference/reproduction receipt failed")
    return boundaries


def verify_predictions(study, fit, held, by_id, checks):
    folder = study / "fits" / fit["key"]
    rows = read_csv(folder / "predictions.csv")
    checks.require(bool(rows) and set(rows[0]) ==
                   {"fit_key", "scheme", "fold", "condition", "id", "label", "p_ai", "logit_margin"},
                   f"{fit['key']}: unexpected held-prediction schema")
    keys = [(row["condition"], row["id"]) for row in rows]
    checks.require(len(keys) == len(set(keys)), f"{fit['key']}: duplicate condition/ID prediction")
    expected = {(condition, identifier) for condition in CONDITIONS for identifier in held
                if condition != "prefix_100" or len(by_id[identifier]["text"].split()) >= 150}
    checks.require(set(keys) == expected, f"{fit['key']}: condition/ID prediction coverage differs")
    for row in rows:
        identifier = row["id"]
        checks.require(identifier in held and int(row["label"]) == int(by_id[identifier]["label"]),
                       f"{fit['key']}: held label/ID differs for {identifier}")
        checks.require(row["fit_key"] == fit["key"] and row["scheme"] == fit["scheme"]
                       and int(row["fold"]) == fit["fold"], f"{fit['key']}: prediction fit identity differs")
        probability, margin = float(row["p_ai"]), float(row["logit_margin"])
        checks.require(math.isfinite(probability) and 0 <= probability <= 1 and math.isfinite(margin),
                       f"{fit['key']}: nonfinite/out-of-range score for {identifier}")
        sigmoid = (1 / (1 + math.exp(-margin))) if margin >= 0 else (math.exp(margin) / (1 + math.exp(margin)))
        checks.close(probability, sigmoid, f"{fit['key']}: probability orientation differs for {identifier}", 1e-6)
    metadata = read_csv(folder / "held_metadata.csv")
    checks.require(bool(metadata) and set(metadata[0]) ==
                   {"id", "label", "n_words", "token_count_full", "tokens_kept", "truncated"},
                   f"{fit['key']}: unexpected held metadata schema")
    checks.require(len(metadata) == len(held) and {row["id"] for row in metadata} == set(held),
                   f"{fit['key']}: held metadata coverage differs")
    for row in metadata:
        identifier = row["id"]
        full, kept = int(row["token_count_full"]), int(row["tokens_kept"])
        checks.require(int(row["label"]) == int(by_id[identifier]["label"])
                       and int(row["n_words"]) == len(by_id[identifier]["text"].split()),
                       f"{fit['key']}: natural-length metadata differs for {identifier}")
        checks.require(full >= 2 and kept == min(full, 512)
                       and row["truncated"].lower() == str(full > 512).lower(),
                       f"{fit['key']}: tokenizer-coverage metadata inconsistent for {identifier}")
    baseline = read_csv(resolve(study, fit["baseline_predictions_file"]))
    baseline_map = {(row["condition"], row["id"]): row for row in baseline}
    checks.require(len(baseline) == len(baseline_map) and set(baseline_map) == expected,
                   f"{fit['key']}: baseline condition/ID alignment differs")
    result = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        labels = [int(row["label"]) for row in selected]
        scores = [float(row["logit_margin"]) for row in selected]
        baseline_scores = []
        for row in selected:
            before = baseline_map[(condition, row["id"])]
            checks.require(before["label"] == row["label"], f"{fit['key']}: baseline label mismatch")
            baseline_scores.append(float(before["baseline_score"]))
        new_auc, baseline_auc = auc(labels, scores), auc(labels, baseline_scores)
        result.append({"fit_key": fit["key"], "scheme": fit["scheme"], "fold": fit["fold"],
                       "condition": condition, "n": len(selected), "n_human": labels.count(0),
                       "n_ai": labels.count(1), "roberta_auc": new_auc,
                       "baseline_auc": baseline_auc,
                       "delta_auc": new_auc - baseline_auc if new_auc is not None and baseline_auc is not None else None})
    return result


def verify_provenance_lineage(study, fit, config, provenance, checks):
    """Allow only the documented post-training CPU-transfer mechanical repair."""
    worker_name = "experiments/supervised/train_roberta.py"
    current_worker = provenance["source_sha256"][worker_name]
    checks.require(provenance["plan_sha256"] == sha256(study / "plan.json"),
                   f"{fit['key']}: source provenance plan hash differs")
    amendments_path = study / "mechanical_amendments.json"
    if provenance["before_first_neural_fit"] is True:
        checks.require(config["source_sha256"] == current_worker,
                       f"{fit['key']}: actual worker differs from before-fit snapshot")
        return
    recovery = study / "mechanical_recovery_001"
    initial = read_json(recovery / "provenance.json")
    amendments = read_json(amendments_path)["amendments"]
    checks.require(initial["before_first_neural_fit"] is True
                   and initial["plan_sha256"] == provenance["plan_sha256"],
                   f"{fit['key']}: original before-fit source/plan lineage missing")
    checks.require(set(initial["source_sha256"]) == set(provenance["source_sha256"])
                   and all(initial["source_sha256"][name] == value for name, value in provenance["source_sha256"].items()
                           if name != worker_name),
                   f"{fit['key']}: non-worker scientific sources changed during mechanical repair")
    for name, expected in initial["source_sha256"].items():
        checks.require(sha256(recovery / "initial_source" / name) == expected,
                       f"{fit['key']}: archived before-fit source differs: {name}")
    allowed = [row for row in amendments if row.get("id") == "evaluation_cpu_transfer_001"
               and row.get("old_worker_sha256") == initial["source_sha256"][worker_name]
               and row.get("new_worker_sha256") == current_worker
               and row.get("allowed_change") == "evaluation_cpu_transfer_cast_only"
               and row.get("applies_to_fit_keys") == ["random_0"]]
    checks.require(len(amendments) == len(allowed) == 1,
                   f"{fit['key']}: unexpected or unapproved mechanical source amendment")
    if config["source_sha256"] == current_worker:
        return
    checks.require(fit["key"] == "random_0" and config["source_sha256"] == initial["source_sha256"][worker_name],
                   f"{fit['key']}: original config source is outside the approved recovery lineage")
    preservation = read_json(recovery / "preservation_receipt.json")
    checks.require(preservation["fit_key"] == "random_0" and preservation["completed_epochs"] == 2
                   and preservation["completed_optimizer_updates"] == fit["expected_optimizer_updates"]
                   and preservation["new_model_fits_required"] == 0
                   and preservation["seed_or_recipe_changed"] is False
                   and preservation["held_quality_scores_available_before_repair"] is False,
                   "Recovered fit changed the trained state, recipe or quality-selection boundary")
    for relative, expected in preservation["original_files_sha256"].items():
        checks.require(sha256(recovery / "random_0_before_resume" / Path(relative).name) == expected,
                       f"Archived failed-attempt evidence differs: {relative}")
    protected = {"config.json", "train_ids.csv", "held_ids.csv", "train_metadata.csv", "held_metadata.csv",
                 "training_steps.csv", "epoch_receipts.json", "memory_samples.csv", "memory_summary.json",
                 "epoch_1_order.csv", "epoch_2_order.csv"}
    for name in protected:
        relative = f"fits/random_0/{name}"
        checks.require(sha256(study / relative) == preservation["original_files_sha256"][relative],
                       f"Post-training recovery rewrote protected training evidence: {name}")
    resumes = [read_json(path) for path in (study / "fits/random_0").glob("resume_receipt_*.json")]
    checks.require(any(row["completed_epoch"] == 2
                       and row["checkpoint_sha256"] == preservation["resume_checkpoint_sha256"]
                       and row["config_sha256"] == preservation["original_files_sha256"]["fits/random_0/config.json"]
                       for row in resumes), "Recovery did not load the pinned completed-epoch2 checkpoint")


def verify_training(study, fit, train, held, by_id, plan, checks):
    folder = study / "fits" / fit["key"]
    config = read_json(folder / "config.json")
    metrics = read_json(folder / "metrics.json")
    completion = read_json(folder / "completion.json")
    receipt = read_json(folder / "validation_receipt.json")
    fallback = read_json(folder / "fallback_receipt.json")
    steps = read_csv(folder / "training_steps.csv")
    epochs = read_json(folder / "epoch_receipts.json")
    checks.require(config["fit"] == fit and config["recipe"] == plan["recipe"]
                   and config["model"] == plan["model"], f"{fit['key']}: worker settings differ from frozen plan")
    checks.require(config["plan_sha256"] == sha256(study / "plan.json")
                   and config["train_input_ids_sha256"] == sha256(resolve(study, fit["train_ids_file"]))
                   and config["held_input_ids_sha256"] == sha256(resolve(study, fit["held_ids_file"])),
                   f"{fit['key']}: worker plan/ID hashes differ")
    checks.require(config["canonical_split_manifest_sha256"] == sha256(ROOT / "data/splits/manifest.json"),
                   f"{fit['key']}: canonical manifest hash differs")
    provenance = read_json(study / "provenance.json")
    verify_provenance_lineage(study, fit, config, provenance, checks)
    for name, expected in provenance["source_sha256"].items():
        checks.require(sha256(study / "source" / name) == expected,
                       f"{fit['key']}: frozen executing source hash differs: {name}")
    checks.require(Path(config["python"]).resolve() == Path("/opt/anaconda3/envs/python_course/bin/python").resolve()
                   and config["mps_available"] is True and config["mps_built"] is True
                   and config["fallback_environment_before_torch_import"] == "0",
                   f"{fit['key']}: Python/MPS/fallback setting differs")
    checks.require(config["torch_cpu_threads"] == plan["recipe"]["cpu_threads"]
                   and config["test_access"] is False and config["train_from_device_probe_weights"] is False,
                   f"{fit['key']}: thread/test/warm-start scope differs")
    checks.require(config["training_id_order_sha256"] == ids_digest(train)
                   and config["held_id_order_sha256"] == ids_digest(held),
                   f"{fit['key']}: input ID order digest differs")
    for part, ids in (("train", train), ("held", held)):
        copied = read_csv(folder / f"{part}_ids.csv")
        checks.require([row["id"] for row in copied] == ids
                       and all(row["label"] == by_id[row["id"]]["label"] for row in copied),
                       f"{fit['key']}: copied {part} IDs/labels differ")
    train_meta = read_csv(folder / "train_metadata.csv")
    checks.require([row["id"] for row in train_meta] == train,
                   f"{fit['key']}: training tokenizer metadata order/coverage differs")
    counts = {}
    for row in train_meta:
        identifier = row["id"]
        full, kept = int(row["token_count_full"]), int(row["tokens_kept"])
        checks.require(row["label"] == by_id[identifier]["label"]
                       and int(row["n_words"]) == len(by_id[identifier]["text"].split())
                       and full >= 2 and kept == min(full, 512)
                       and row["truncated"].lower() == str(full > 512).lower(),
                       f"{fit['key']}: training tokenizer metadata inconsistent for {identifier}")
        counts[identifier] = kept
    updates_per_epoch = math.ceil(len(train) / 16)
    checks.require(len(steps) == 2 * updates_per_epoch == fit["expected_optimizer_updates"]
                   and [int(epoch["epoch"]) for epoch in epochs] == [1, 2],
                   f"{fit['key']}: missing/extra epochs or optimizer updates")
    for epoch in epochs:
        number = int(epoch["epoch"])
        order_rows = read_csv(folder / f"epoch_{number}_order.csv")
        order = [row["id"] for row in order_rows]
        expected_order = list(train)
        random.Random(plan["recipe"]["seed"] + number).shuffle(expected_order)
        checks.require(len(order) == len(set(order)) == len(train) and set(order) == set(train)
                       and order == expected_order and ids_digest(order) == epoch["epoch_permutation_sha256"]
                       and epoch["epoch_order_file"] == f"epoch_{number}_order.csv"
                       and sha256(folder / epoch["epoch_order_file"]) == epoch["epoch_order_file_sha256"],
                       f"{fit['key']}: epoch {number} original coverage/permutation digest differs")
        selected = [row for row in steps if int(row["epoch"]) == number]
        checks.require(epoch["originals_seen"] == epoch["unique_originals_seen"] == len(train)
                       and epoch["optimizer_updates"] == len(selected) == updates_per_epoch
                       and epoch["held_model_inputs_used"] is False and epoch["held_metrics_evaluated"] is False,
                       f"{fit['key']}: epoch {number} count/held-input receipt differs")
        for index, row in enumerate(selected):
            group = order[index * 16:(index + 1) * 16]
            sizes = [len(group[start:start + 4]) for start in range(0, len(group), 4)]
            checks.require(int(row["optimizer_update_in_epoch"]) == index + 1
                           and int(row["optimizer_update_global"]) == (number - 1) * updates_per_epoch + index + 1
                           and int(row["documents"]) == len(group)
                           and json.loads(row["microbatch_sizes_json"]) == sizes
                           and row["original_id_sequence_sha256"] == ids_digest(group)
                           and int(row["effective_tokens"]) == sum(counts[identifier] for identifier in group),
                           f"{fit['key']}: epoch {number}/update {index + 1} schedule/count/hash differs")
            checks.require(all(row[field] == "True" for field in (
                "loss_and_gradient_norm_finite", "logits_on_mps", "gradient_norm_on_mps")),
                f"{fit['key']}: epoch {number}/update {index + 1} numeric/device check failed")
            checks.require(float(row["seconds"]) > 0 and all(math.isfinite(float(row[field]))
                           and float(row[field]) >= 0 for field in ("seconds", "mean_loss", "preclip_gradient_norm")),
                           f"{fit['key']}: epoch {number}/update {index + 1} invalid numeric value")
        checks.close(epoch["training_update_seconds"], sum(float(row["seconds"]) for row in selected),
                     f"{fit['key']}: epoch {number} time sum differs")
        checks.close(epoch["mean_loss"], sum(float(row["mean_loss"]) * int(row["documents"]) for row in selected) / len(train),
                     f"{fit['key']}: epoch {number} weighted loss differs")
        checks.require(float(epoch["epoch_loop_wall_seconds"]) >= float(epoch["training_update_seconds"]),
                       f"{fit['key']}: epoch {number} loop time shorter than updates")
    checks.require(metrics["status"] == completion["status"] == "complete"
                   and completion["metrics_sha256"] == sha256(folder / "metrics.json")
                   and metrics["n_train"] == len(train) and metrics["n_held"] == len(held)
                   and metrics["epochs"] == completion["epochs"] == 2,
                   f"{fit['key']}: completion counts/hashes differ")
    checks.require(metrics["optimizer_updates"] == completion["optimizer_updates"] == len(steps)
                   and metrics["training_documents_processed"] == 2 * len(train),
                   f"{fit['key']}: total optimizer/document count differs")
    checks.close(metrics["training_update_seconds"], sum(float(row["seconds"]) for row in steps),
                 f"{fit['key']}: total training seconds differ")
    checks.close(metrics["training_epoch_loop_seconds"], sum(float(epoch["epoch_loop_wall_seconds"]) for epoch in epochs),
                 f"{fit['key']}: total epoch loop seconds differ")
    checks.require(all(receipt[field] is True for field in (
        "forward_backward_optimizer_executed", "all_numeric_checks_passed", "all_gradients_mps",
        "encoder_and_classifier_parameters_updated")) and metrics["all_gradients_mps"] is True
        and receipt["completed_epochs_before_held_inference"] == 2
        and receipt["held_early_stopping_or_epoch_selection"] is False,
        f"{fit['key']}: full-update/final-epoch evaluation checks failed")
    changes = metrics["monitored_parameter_max_changes"]
    checks.require(len(changes) == 2 and any("encoder.layer.0" in key for key in changes)
                   and any("classifier.out_proj.weight" in key for key in changes)
                   and all(math.isfinite(value) and value > 0 for value in changes.values()),
                   f"{fit['key']}: encoder/classifier update evidence failed")
    states = metrics["optimizer_state_devices"]
    checks.require(bool(states) and all(key.startswith("step:") or key.split(":")[1] == "mps" for key in states)
                   and any(key.startswith("exp_avg:mps:") for key in states)
                   and any(key.startswith("exp_avg_sq:mps:") for key in states),
                   f"{fit['key']}: AdamW MPS moment state differs")
    checks.require(metrics["reload_max_logit_error"] == receipt["reload_max_logit_error"]
                   and 0 <= metrics["reload_max_logit_error"] <= receipt["reload_tolerance"] == 1e-6,
                   f"{fit['key']}: checkpoint reload error failed")
    checkpoint = folder / "model.pt"
    checks.require(checkpoint.stat().st_size == metrics["checkpoint_bytes"]
                   and sha256(checkpoint) == metrics["checkpoint_sha256"],
                   f"{fit['key']}: scientific checkpoint bytes/hash differs")
    for name, expected in receipt["files_sha256"].items():
        checks.require(sha256(folder / name) == expected, f"{fit['key']}: receipt output hash differs: {name}")
    warnings = read_json(folder / "warnings.json")
    fallback_warnings = [row for row in warnings if any(word in row["message"].lower()
                         for word in ("fallback", "fall back", "falling back", "not currently supported on the mps"))]
    checks.require(fallback["environment"] == "0" and fallback["captured_fallback_warning_count"] == 0
                   and fallback_warnings == fallback["captured_fallback_warnings"] == [],
                   f"{fit['key']}: Python unsupported-op fallback warning observed")
    receipts = [row for row in read_json(study / "execution_receipts.json") if row["fit_key"] == fit["key"]]
    checks.require(bool(receipts) and receipts[-1]["return_code"] == 0,
                   f"{fit['key']}: successful process exit receipt missing")
    native_matches = []
    for process_receipt in receipts:
        for part in ("stdout", "stderr"):
            path = ROOT / process_receipt[f"{part}_path"]
            checks.require(sha256(path) == process_receipt[f"{part}_sha256"],
                           f"{fit['key']}: native {part} hash differs")
            native_matches.extend(NATIVE_FALLBACK.findall(path.read_text(errors="replace")))
    checks.require(native_matches == [], f"{fit['key']}: native unsupported-op fallback warning observed")
    memory = read_json(folder / "memory_summary.json")
    samples = read_csv(folder / "memory_samples.csv")
    checks.require(memory["n_samples"] == len(samples) and bool(samples) and memory["sample_errors"] == [],
                   f"{fit['key']}: memory observation count/errors differ")
    for name, extrema in memory["all_training"].items():
        values = [int(row[name]) for row in samples]
        checks.require(extrema == {"minimum": min(values), "maximum": max(values)},
                       f"{fit['key']}: sampled memory extrema differ: {name}")
    return metrics


def verify_metric_outputs(study, reconstructed, checks):
    path = study / "analysis/per_fold_metrics.csv"
    if not path.is_file():
        return False
    rows = read_csv(path)
    keyed = {(row["fit_key"], row["condition"]): row for row in rows}
    expected_keys = {(row["fit_key"], row["condition"]) for row in reconstructed}
    checks.require(len(rows) == len(keyed) and expected_keys <= set(keyed), "Analysis metric keys duplicate/missing")
    for result in reconstructed:
        row = keyed[(result["fit_key"], result["condition"])]
        for field in ("n", "n_human", "n_ai"):
            checks.require(int(row[field]) == result[field], f"Analysis {result['fit_key']}/{result['condition']} {field} differs")
        for field, expected_field in (("roberta_auroc", "roberta_auc"), ("wc120_auroc", "baseline_auc"), ("delta_auroc", "delta_auc")):
            checks.close(float(row[field]), result[expected_field], f"Analysis {result['fit_key']}/{result['condition']} {field} differs")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--fit-key")
    parser.add_argument("--preflight", action="store_true", help="Check frozen inputs/boundaries without requiring a fit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = Checks()
    result = {"verifier": "Zero-fit standard-library artifact audit", "verifier_source_sha256": sha256(__file__),
              "verifier_python": sys.executable, "study_dir": str(args.study_dir.resolve()),
              "limitations": ["No model loading, checkpoint inference or fresh fit is performed.",
                              "Saved evidence and source review do not constitute an operator-by-operator GPU profile.",
                              "Reused topic/style proxy and validation are developmental, not untouched hidden-domain estimates."]}
    try:
        study = args.study_dir.resolve()
        plan = read_json(study / "plan.json")
        recipe = plan["recipe"]
        for key, expected in {"epochs": 2, "microbatch": 4, "gradient_accumulation": 4,
                              "max_length": 512, "learning_rate": 2e-5, "weight_decay": .01,
                              "max_grad_norm": 1.0, "seed": 42, "attention": "sdpa", "dtype": "float32"}.items():
            checks.require(recipe[key] == expected, f"Frozen recipe differs: {key}")
        model = plan["model"]
        snapshot = Path(model["snapshot"])
        checks.require(model["revision"] == "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
                       and snapshot.name == model["revision"], "Unexpected generic base checkpoint revision")
        checks.require(sha256(snapshot / "model.safetensors") == model["weights_sha256"], "Generic base weights hash differs")
        train_records, validation_records = load_partitions()
        by_id = {row["id"]: row for row in train_records + validation_records}
        fits = plan["fits"]
        checks.require({fit["key"] for fit in fits} ==
                       {"random_0", "random_1", "random_2", "topic_0", "topic_1", "topic_2", "canonical_8000"}
                       and len(fits) == 7, "Expected all seven prescribed fits")
        if args.fit_key:
            fits = [fit for fit in fits if fit["key"] == args.fit_key]
            checks.require(len(fits) == 1, "Unknown/duplicate fit key")
        boundaries = verify_prepared(study, plan, fits, train_records, validation_records, checks)
        reconstructed = []
        if not args.preflight:
            for fit in fits:
                train, held = boundaries[fit["key"]]
                metrics = verify_training(study, fit, train, held, by_id, plan, checks)
                fit_metrics = verify_predictions(study, fit, held, by_id, checks)
                for metric in fit_metrics:
                    recorded = metrics["condition_metrics"][metric["condition"]]
                    checks.close(recorded["auroc_logit_margin"], metric["roberta_auc"],
                                 f"{fit['key']}/{metric['condition']}: worker AUROC differs")
                reconstructed.extend(fit_metrics)
            result["analysis_metrics_present_and_checked"] = verify_metric_outputs(study, reconstructed, checks)
        result["audit_scope"] = "preflight" if args.preflight else (args.fit_key or "all_prescribed_fits")
        result["reconstructed_metrics"] = reconstructed
    except Exception as error:
        checks.errors.append(f"Verification could not complete: {type(error).__name__}: {error}")
    result.update(status="passed" if not checks.errors else "failed", passed_checks=checks.passed, errors=checks.errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Preserve the previous verification receipt: {args.output}")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "passed_checks": checks.passed,
                      "errors": checks.errors, "output": str(args.output)}))
    if checks.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
