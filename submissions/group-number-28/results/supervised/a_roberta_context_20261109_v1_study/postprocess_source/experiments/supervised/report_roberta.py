"""Render completed RoBERTa evidence as English Markdown, without model code.

This reporting utility is intentionally outside the frozen training dependency
graph. It never loads model weights, predicts, fits, or modifies study evidence.
The new report and generation receipt retain this utility's source hash.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("clean", "quote_and_dash", "digit_runs_to_zero", "whitespace_only", "prefix_100")
FIT_KEYS = tuple(f"{scheme}_{fold}" for scheme in ("random", "topic") for fold in range(3)) + ("canonical_8000",)
SCHEME_LABELS = {"random": "Matched random", "topic": "Topic proxy", "validation": "Reused canonical validation"}
NATIVE_FALLBACK = re.compile(r"not currently supported on[^\n]*mps|will fall back[^\n]*cpu|mps[^\n]*(?:fallback|fall back|falling back)[^\n]*cpu", re.IGNORECASE)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("A report cannot present nonfinite measured values")
    return value


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |"] +
                      ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def number(value, precision=6, signed=False):
    return format(numeric(value), f"{'+' if signed else ''}.{precision}f")


def percentage(value):
    return f"{100 * numeric(value):.2f}%"


class Inputs:
    def __init__(self):
        self.hashes = {}

    def record(self, path):
        path = Path(path).resolve()
        self.hashes[str(path)] = sha256(path)
        return path

    def json(self, path):
        return json.loads(self.record(path).read_text())

    def csv(self, path):
        with self.record(path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"Missing or duplicate headers: {path}")
            rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"Malformed CSV: {path}")
        return rows


def linked(output, path, label):
    relative = os.path.relpath(Path(path).resolve(), output.parent.resolve())
    return f"[{label}]({quote(relative, safe='/._-')})"


def resolve_log(study, path):
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = (ROOT / path, study / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Native log is missing: {path}")


def delta_interval(result):
    if not result["interval_available"]:
        return f"unavailable ({result['valid_replicates']} valid)"
    return f"[{number(result['paired_delta_ci_2_5'], signed=True)}, {number(result['paired_delta_ci_97_5'], signed=True)}]"


def sensitivity_tradeoff_sentence(aggregate):
    descriptions = []
    for condition, label in (("whitespace_only", "Whitespace-only"), ("prefix_100", "Joined prefix100")):
        losses = [f"{SCHEME_LABELS[scheme].lower()} {number(aggregate[scheme][condition]['macro_delta_auroc'], signed=True)}"
                  for scheme in ("random", "topic", "validation")
                  if numeric(aggregate[scheme][condition]["macro_delta_auroc"]) < 0]
        if losses:
            descriptions.append(f"{label} regressions versus WC120: " + "; ".join(losses))
    return ". ".join(descriptions) + (". These losses remain part of the reported model-family tradeoff." if descriptions else "")


def execution_span(receipts):
    """Reconstruct the full timestamp span, distinct from summed process time."""
    spans = []
    for receipt in receipts:
        finished = datetime.fromisoformat(receipt["finished_at_utc"].replace("Z", "+00:00"))
        if finished.tzinfo is None:
            raise ValueError("Execution receipt timestamps must have an explicit timezone")
        finished = finished.astimezone(timezone.utc)
        seconds = numeric(receipt["process_wall_seconds"])
        if seconds < 0:
            raise ValueError("Negative process duration")
        spans.append((finished - timedelta(seconds=seconds), finished))
    if not spans:
        raise ValueError("Native execution receipts are required")
    first, last = min(span[0] for span in spans), max(span[1] for span in spans)
    return first, last, (last - first).total_seconds()


def require_complete(study, inputs):
    plan = inputs.json(study / "plan.json")
    fits = plan["fits"]
    if len(fits) != 7 or {fit["key"] for fit in fits} != set(FIT_KEYS):
        raise ValueError("A report requires the full prescribed seven-fit experiment")
    training = inputs.json(study / "training_completion.json")
    if training["status"] != "complete" or training["fits"] != 7:
        raise ValueError("Training is incomplete")
    analysis = inputs.json(study / "analysis/analysis_receipt.json")
    if analysis["status"] != "complete" or analysis["fits"] != 7 or analysis["primary_condition_rows"] != 35:
        raise ValueError("Analysis is incomplete")
    for name, expected in analysis["output_sha256"].items():
        if sha256(study / "analysis" / name) != expected:
            raise ValueError(f"Recorded analysis output hash changed: {name}")
    per_fit = {}
    for fit in fits:
        key = fit["key"]
        folder = study / "fits" / key
        completion = inputs.json(folder / "completion.json")
        metrics_path = folder / "metrics.json"
        metrics = inputs.json(metrics_path)
        if (completion["status"] != "complete" or metrics["status"] != "complete" or
                completion["metrics_sha256"] != sha256(metrics_path) or
                completion["fit_key"] != key or metrics["fit_key"] != key or
                completion["optimizer_updates"] != metrics["optimizer_updates"]):
            raise ValueError(f"Incomplete or inconsistent fit: {key}")
        memory = inputs.json(folder / "memory_summary.json")
        if memory["sample_errors"]:
            raise ValueError(f"Memory sampler errors require explicit review: {key}")
        fallback = inputs.json(folder / "fallback_receipt.json")
        per_fit[key] = {"definition": fit, "metrics": metrics, "memory": memory, "fallback": fallback}
    receipts = inputs.json(study / "execution_receipts.json")
    for key in FIT_KEYS:
        selected = [row for row in receipts if row["fit_key"] == key]
        if not selected or selected[-1]["return_code"] != 0:
            raise ValueError(f"Successful native process receipt missing: {key}")
        per_fit[key]["process_receipts"] = selected
    return plan, per_fit, receipts, analysis


def mechanical_recovery_notes(study, inputs, per_fit):
    """Read explicit recovery lineage; never infer a favorable restart."""
    path = study / "mechanical_amendments.json"
    if not path.is_file():
        return [], []
    amendments = inputs.json(path)["amendments"]
    notes, evidence = [], [(path, "Mechanical source amendments")]
    for amendment in amendments:
        preservation_path = Path(amendment["preservation_receipt"])
        if not preservation_path.is_absolute():
            preservation_path = study / preservation_path
        preservation = inputs.json(preservation_path)
        key = preservation["fit_key"]
        if key not in per_fit:
            raise ValueError(f"Recovery receipt names an unknown fit: {key}")
        notes.append(
            f"Recovery `{amendment['id']}` for `{key}`: {preservation['reason']}. "
            f"The preservation receipt records {preservation['completed_epochs']} completed epochs and "
            f"{preservation['completed_optimizer_updates']:,} completed optimizer updates, "
            f"{preservation['new_model_fits_required']} new model fits required, "
            f"seed/recipe changed: {str(preservation['seed_or_recipe_changed']).lower()}, "
            f"held-quality scores available before repair: {str(preservation['held_quality_scores_available_before_repair']).lower()}. "
            f"The registered repair scope is `{amendment['allowed_change']}`; old/new worker hashes and the saved checkpoint hash are preserved in the linked lineage."
        )
        metrics = per_fit[key]["metrics"]
        if "training_updates_in_this_process" in metrics:
            notes.append(f"The final `{key}` process records {metrics['training_updates_in_this_process']:,} new optimizer updates in that process; its full epoch-loop totals retain the saved completed training. This distinguishes recovery execution cost from scientific training cost.")
        evidence.append((preservation_path, f"{key} preserved pre-repair evidence"))
    return notes, evidence


def render(study, output, verification=None):
    inputs = Inputs()
    plan, per_fit, receipts, _ = require_complete(study, inputs)
    aggregate = inputs.json(study / "analysis/aggregate_metrics.json")
    bootstrap = inputs.json(study / "analysis/paired_bootstrap.json")
    coverage = inputs.json(study / "analysis/coverage_summary.json")
    metrics_rows = inputs.csv(study / "analysis/per_fold_metrics.csv")
    pairs = inputs.csv(study / "analysis/pair_complementarity.csv")
    expected = {(key, condition) for key in FIT_KEYS for condition in CONDITIONS}
    keyed_rows = {(row["fit_key"], row["condition"]): row for row in metrics_rows}
    keyed_pairs = {(row["fit_key"], row["condition"]): row for row in pairs}
    if len(metrics_rows) != 35 or set(keyed_rows) != expected or len(pairs) != 35 or set(keyed_pairs) != expected:
        raise ValueError("Full per-fit metrics or complementarity evidence is missing/duplicated")
    repetitions = int(plan["statistics"]["bootstrap_repetitions"])
    if any(int(value["requested_replicates"]) != repetitions for value in bootstrap.values()):
        raise ValueError("Bootstrap replicate count differs from the frozen plan")
    audit = inputs.json(verification) if verification else None
    if audit and (audit["status"] != "passed" or audit.get("audit_scope") != "all_prescribed_fits"):
        raise ValueError("The supplied final verification must pass for all prescribed fits")
    recovery_notes, recovery_evidence = mechanical_recovery_notes(study, inputs, per_fit)

    # Scan existing native output, without importing the frozen worker or Torch.
    native_matches = []
    for receipt in receipts:
        for part in ("stdout", "stderr"):
            path = inputs.record(resolve_log(study, receipt[f"{part}_path"]))
            if sha256(path) != receipt[f"{part}_sha256"]:
                raise ValueError(f"Native log hash changed: {path}")
            native_matches.extend(NATIVE_FALLBACK.findall(path.read_text(errors="replace")))
    fallback_count = sum(int(item["fallback"]["captured_fallback_warning_count"]) for item in per_fit.values())

    random_delta = numeric(aggregate["random"]["clean"]["macro_delta_auroc"])
    topic_delta = numeric(aggregate["topic"]["clean"]["macro_delta_auroc"])
    if random_delta > 0 and topic_delta > 0:
        lead = "RoBERTa improves mean clean ranking against WC120 under both existing split schemes."
    elif random_delta < 0 and topic_delta < 0:
        lead = "RoBERTa trails WC120 in mean clean ranking under both existing split schemes."
    else:
        lead = "The clean-ranking comparison has different tradeoffs across the existing split schemes."
    total_process_seconds = sum(numeric(row["process_wall_seconds"]) for row in receipts)
    first_attempt_start, last_attempt_finish, execution_span_seconds = execution_span(receipts)
    success_process_seconds = sum(numeric(item["process_receipts"][-1]["process_wall_seconds"]) for item in per_fit.values())
    epoch_loop_seconds = sum(numeric(item["metrics"]["training_epoch_loop_seconds"]) for item in per_fit.values())
    update_seconds = sum(numeric(item["metrics"]["training_update_seconds"]) for item in per_fit.values())
    evaluation_seconds = sum(numeric(item["metrics"]["evaluation_seconds"]) for item in per_fit.values())
    inference_seconds = sum(numeric(condition["inference_seconds"]) for item in per_fit.values()
                            for condition in item["metrics"]["condition_metrics"].values())
    driver_max = max(numeric(item["memory"]["all_training"]["mps_driver_allocated_bytes"]["maximum"]) for item in per_fit.values())
    tensor_max = max(numeric(item["memory"]["all_training"]["mps_current_allocated_bytes"]["maximum"]) for item in per_fit.values())
    rss_max = max(numeric(item["memory"]["all_training"]["process_rss_bytes"]["maximum"]) for item in per_fit.values())
    lifetime_rss_max = max(numeric(item["metrics"]["process_lifetime_max_rss_bytes"]) for item in per_fit.values())

    comparison = []
    for scheme in ("random", "topic", "validation"):
        for condition in CONDITIONS:
            row, interval = aggregate[scheme][condition], bootstrap[scheme]["conditions"][condition]
            if not math.isclose(numeric(row["macro_delta_auroc"]), numeric(interval["point_macro_delta_auroc"]), abs_tol=1e-12):
                raise ValueError(f"Bootstrap and macro delta disagree: {scheme}/{condition}")
            comparison.append([SCHEME_LABELS[scheme], condition, number(row["wc120_macro_auroc"]),
                               number(row["roberta_macro_auroc"]), number(row["macro_delta_auroc"], signed=True),
                               delta_interval(interval), f"{row['positive_folds']}/{len(row['folds'])}"])

    all_folds = []
    clean_pairs = []
    cost = []
    coverage_rows = []
    for key in FIT_KEYS:
        item = per_fit[key]
        for condition in CONDITIONS:
            row = keyed_rows[(key, condition)]
            all_folds.append([key, condition, row["n"], number(row["wc120_auroc"]), number(row["roberta_auroc"]), number(row["delta_auroc"], signed=True)])
        pair = keyed_pairs[(key, "clean")]
        clean_pairs.append([key, percentage(pair["repair_credit_fraction"]), percentage(pair["regression_credit_fraction"]),
                            number(pair["net_credit_fraction"], signed=True),
                            number(pair["spearman_rank_correlation"], precision=4) if pair["spearman_rank_correlation"] != "" else "undefined",
                            pair["both_strictly_inverted_pairs"], pair["n_positive_negative_pairs"]])
        m, memory = item["metrics"], item["memory"]["all_training"]
        measured_inference = sum(numeric(condition["inference_seconds"]) for condition in m["condition_metrics"].values())
        cost.append([key, f"{m['n_train']:,}", f"{m['n_held']:,}", m["optimizer_updates"],
                     number(numeric(m["training_epoch_loop_seconds"]) / 60, 2), number(measured_inference, 2),
                     number(m["evaluation_seconds"], 2),
                     number(sum(numeric(row["process_wall_seconds"]) for row in item["process_receipts"]) / 60, 2),
                     number(numeric(item["process_receipts"][-1]["process_wall_seconds"]) / 60, 2),
                     number(numeric(memory["mps_driver_allocated_bytes"]["maximum"]) / 2**30, 3)])
        for boundary in ("training", "held"):
            c = coverage[key][boundary]
            coverage_rows.append([key, boundary, f"{c['n']:,}", c["truncated_n"], percentage(c["truncated_fraction"]),
                                  number(c["full_token_count_quantiles"]["0.5"], 1),
                                  number(c["full_token_count_quantiles"]["0.95"], 1), percentage(c["kept_token_fraction_of_total"])])

    evidence_paths = [
        (study / "plan.json", "Frozen plan"), (study / "provenance.json", "Executing-source provenance"),
        (study / "analysis/per_fold_metrics.csv", "All per-fit metrics"),
        (study / "analysis/aggregate_metrics.json", "Macro comparisons"),
        (study / "analysis/paired_bootstrap.json", "Paired bootstrap intervals"),
        (study / "analysis/pair_complementarity.csv", "Pair-credit and rank-correlation evidence"),
        (study / "analysis/clean_subgroups.csv", "Clean natural-length and truncation subgroups"),
        (study / "analysis/coverage_summary.json", "Tokenizer coverage"),
        (study / "execution_receipts.json", "Native process costs and log hashes"),
        (study / "analysis/analysis_receipt.json", "Analysis receipt"),
    ]
    if verification:
        evidence_paths.append((verification, "Final independent verification"))
    evidence_paths.extend(recovery_evidence)
    for path, _ in evidence_paths:
        inputs.record(path)
    evidence = "\n".join(f"- {linked(output, path, label)}." for path, label in evidence_paths)
    recipe = plan["recipe"]
    audit_text = (f"The supplied independent zero-fit audit passed {audit['passed_checks']:,} checks for all seven fits. "
                  "It inspects saved evidence without a fresh fit or fresh checkpoint inference."
                  if audit else "No final independent verification receipt was supplied to this generator; no audit-pass claim is made here.")
    fallback_text = ("No unsupported-op CPU fallback warning was observed in the captured Python warnings or scanned native logs across these exercised paths."
                     if fallback_count == 0 and not native_matches else
                     f"Fallback observations require review: {fallback_count} captured Python warning(s), {len(native_matches)} native log match(es).")
    model_name = plan["model"]["name"]
    total_updates = sum(int(item["metrics"]["optimizer_updates"]) for item in per_fit.values())
    attempt_text = (f"There were {len(receipts)} native process attempts. Their total measured wall time was "
                    f"{total_process_seconds / 60:.2f} minutes; the final successful attempts total "
                    f"{success_process_seconds / 60:.2f} minutes. Failed/interrupted attempts, if any, remain included in the former total.")
    failed_attempts = [row for row in receipts if int(row["return_code"]) != 0]
    mechanical_text = (f"There were {len(failed_attempts)} failed/interrupted native attempt(s), affecting "
                       + ", ".join(sorted({row["fit_key"] for row in failed_attempts}))
                       + ". The per-fit epoch-loop costs describe the completed prescribed epochs; the native all-attempt costs also retain failed-attempt time. A recovery-only successful process can be short because its training epochs were already saved. See the execution receipts and recovery evidence for the mechanical cause and exact continuation boundary."
                       if failed_attempts else "No failed native process attempt is recorded in the execution receipts.")
    lines = [
        f"# RoBERTa contextual experiment — {plan['created_at_utc'][:10]}", "",
        f"**{lead}** The paired clean mean differences are {number(random_delta, signed=True)} for matched random "
        f"and {number(topic_delta, signed=True)} for the topic proxy. All seven prescribed fits and all five evaluation conditions completed; unfavorable folds are retained.", "",
        *([sensitivity_tradeoff_sentence(aggregate), ""] if sensitivity_tradeoff_sentence(aggregate) else []),
        "This is one fixed model-family comparison using supplied labels. It does not isolate a causal effect of pretraining or architecture, and no hidden-test score was measured or inferred. No quality threshold stopped or promoted a fit, no fusion weights were learned, and no submission was exported.", "",
        "## Frozen recipe and boundaries", "",
        f"Generic `{model_name}` revision `{plan['model']['revision']}`, FP32 SDPA, {recipe['epochs']} epochs, "
        f"constant learning rate {recipe['learning_rate']:g}, AdamW weight decay {recipe['weight_decay']:g}, gradient clip {recipe['max_grad_norm']:g}, "
        f"seed {recipe['seed']}, {recipe['max_length']} fixed padded tokens, microbatch {recipe['microbatch']} with "
        f"{recipe['gradient_accumulation']} accumulation steps (effective batch {recipe['microbatch'] * recipe['gradient_accumulation']}). "
        "Partial updates retain every original and weight each microbatch by its actual share of the update. Every fit starts from the pinned generic checkpoint; device-test weights are not reused.", "",
        f"Training uses `{plan['python_executable']}` on this M4 Pro. The last prescribed epoch is evaluated without held-loss early stopping or epoch selection. The six CV fits reuse the exact matched random/topic memberships within the canonical 8,000 originals; the additional fit uses the preserved 8,000/2,000 development boundary. No test/unlabeled data or external authorship labels enter this experiment.", "",
        "Primary AUROC uses the raw AI-minus-human logit margin. Probabilities are computed from the FP32 logits after conversion to CPU float64 and are recorded as secondary outputs. WC120 uses the preserved within-fold SVM margins for CV and the historical calibrated probability on canonical validation. No raw-margin/probability average or held-cohort score transform is used.", "",
        "## Main comparisons", "",
        table(["Evaluation", "Condition", "WC120 AUROC", "RoBERTa AUROC", "Delta", "Paired 95% interval", "Positive folds"], comparison), "",
        "CV values are equal-weight means of three within-fold AUROCs; scores from different fitted models are never pooled for a primary AUROC. Canonical validation is one reused development set, not an untouched confirmation. Prefix100 uses the same original IDs with at least 150 whitespace words, evaluating their first 100 words joined with one ASCII space. It both shortens the passage and normalizes its separators, so a prefix100 decline cannot be attributed solely to reduced context.", "",
        f"Intervals condition on these fitted models and fixed/adaptively reused splits. Matched random and canonical draws resample class-stratified original IDs; topic draws resample whole frozen topic clusters separately inside each held fold. The same original multiplicities are paired across models and conditions. Undefined replicates are counted and not redrawn. The {repetitions:,}-replicate intervals are descriptive, without training-variation, adaptive-selection or multiplicity adjustment; overlap among fitting folds prevents treating folds as independent replications.", "",
        "## All prescribed fold and condition results", "",
        table(["Fit", "Condition", "Held n", "WC120 AUROC", "RoBERTa AUROC", "Delta"], all_folds), "",
        "The other conditions normalize quote/dash characters, replace Unicode digit runs with zero, or normalize whitespace. They are dependent sensitivity variants, not separate domains or evidence that the hidden test uses those transformations. No previously sealed suffix/middle condition was opened.", "",
        "## Ranking complementarity", "",
        table(["Clean fit", "Repair credit", "Regression credit", "Net / all pairs", "Spearman rho", "Both inverted", "AI-human pairs"], clean_pairs), "",
        "Each AI-human pair earns 1 for correct order, 0.5 for an exact tie and 0 for an inversion. Repair/regression columns are positive/negative changes in that credit divided by all pairs; their difference reconstructs the displayed AUROC delta. A pair may be half-repaired or half-regressed. The same documents contribute many pairs, so pair counts are not independent sample sizes. Joint inversions and rank correlation describe error overlap; they do not establish that a deployable blend would gain. A probability combination would need training-only calibration and a separate evaluation boundary for any learned combination weights.", "",
        "## Tokenizer coverage and natural length", "",
        table(["Fit", "Boundary", "Originals", "Truncated", "Truncated fraction", "Full tokens median", "Full tokens p95", "Token mass retained"], coverage_rows), "",
        "Tokenizer counts include special tokens; truncation is full token count above 512 and kept tokens are min(full count,512). Token mass retained is an aggregate count ratio, not a guarantee that omitted content is irrelevant. Fixed 512-token padding still incurs computation for shorter passages. Held coverage is constructed after the final training update. Natural-length subgroup boundaries are whitespace-word tertiles learned only on the corresponding fitting originals; all subgroup counts and undefined single-class results remain in the linked numeric evidence.", "",
        "## Measured resource cost", "",
        table(["Fit", "Train n", "Held n", "Updates", "Epoch loops, min", "5-condition inference, s", "Evaluation total, s", "All native attempts, min", "Final successful process, min", "MPS driver max, GiB"], cost), "",
        f"The seven fits executed {total_updates:,} optimizer updates. Their completed epoch loops total {epoch_loop_seconds / 60:.2f} minutes "
        f"({update_seconds / 60:.2f} minutes inside synchronized training-update timers). Five-condition inference totals {inference_seconds / 60:.2f} minutes; "
        f"the broader evaluation windows total {evaluation_seconds / 60:.2f} minutes. {attempt_text}", "",
        f"The reconstructed wall-clock span from the first attempt's inferred start ({first_attempt_start.isoformat()}) to the last finish ({last_attempt_finish.isoformat()}) was {execution_span_seconds / 60:.2f} minutes. It includes gaps between attempts, including a mechanical-repair pause if one occurred. Start times are inferred from each recorded UTC finish minus its measured process duration; this is a timestamp-span reconstruction, not another training timer.", "",
        mechanical_text, "",
        *[paragraph for note in recovery_notes for paragraph in (note, "")],
        "These timing columns overlap and must not be added: evaluation includes held tokenization and condition inference, while native process wall time includes setup, training, epoch checkpointing, evaluation and final save/reload. The native-process sum excludes gaps between subprocesses; the reconstructed span includes those gaps. Both exclude the earlier model download and classical-baseline preparation and the later analysis/auditing/report generation. Neither is the entire research-session duration.", "",
        f"Across the sampled training windows, the largest MPS driver allocation was {driver_max / 2**30:.3f} GiB, "
        f"MPS tensor allocation {tensor_max / 2**30:.3f} GiB and process RSS {rss_max / 2**30:.3f} GiB. "
        f"The separate whole-process RSS high-water mark was {lifetime_rss_max / 2**30:.3f} GiB. "
        "These are maxima over separate observations, not jointly additive components. Unified-memory measurements overlap; do not sum RSS and MPS memory. The sampler covers training and epoch checkpoint windows and stops before held inference and final save/reload. These are observed sampled extrema rather than exact GPU peaks; the process lifetime RSS has broader scope.", "",
        fallback_text + " Generic fallback was disabled before importing PyTorch. Ordinary CPU tokenization, dispatch, scalar diagnostics and AdamW step bookkeeping remain expected; this is not an operator-by-operator GPU profile or a zero-CPU claim.", "",
        "## Verification, interpretation and remaining unknowns", "",
        audit_text, "",
        "The saved numerical checks cover finite forward/backward updates, MPS gradients and optimizer moments, changed encoder/classifier parameters, and final checkpoint reload agreement on two clean held inputs after training. This checks numerical consistency, not independent-environment reproduction or full-model task correctness.", "",
        "The topic split remains a lexical-topic/style proxy with only 12 existing clusters, including mixed/style groups; it is not a true source, generator or deployment-domain holdout. These are development comparisons after earlier diagnostic exploration, one recipe and one seed. They cannot establish private-leaderboard transfer or eliminate source/near-duplicate leakage. Better local AUROC would not guarantee a higher competition score; unfavorable local results do not rule out other encoders or training recipes.", "",
        "New shared assets are seven independently bounded contextual checkpoints, paired per-original predictions on five conditions, verified training receipts, tokenizer coverage and measured experiment cost. Decisions about follow-up architectures, crop handling or calibrated combination can now use this evidence. Hidden-domain transfer, training variation and the value of a properly evaluated combination remain unverified.", "",
        "## Machine evidence", "", evidence, "",
        "Model binaries and native logs are local ignored artifacts. The committed-size evidence records numeric results, IDs, hashes and frozen sources without raw passages. No automatic Kaggle or GitHub action is performed by this report utility.", "",
    ]
    return "\n".join(lines), inputs.hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    study, output = args.study_dir.resolve(), args.output.resolve()
    receipt = args.receipt.resolve() if args.receipt else study / "report_receipt.json"
    if output.exists() or receipt.exists():
        raise FileExistsError("Preserve existing reports and generation receipts; choose new paths")
    text, hashes = render(study, output, args.verification.resolve() if args.verification else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    result = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "reporter_source_sha256": sha256(__file__), "study_dir": str(study),
              "output": str(output), "output_sha256": sha256(output), "input_files_sha256": hashes,
              "model_imports_or_fits": False, "frozen_evidence_modified": False,
              "report_outside_frozen_training_dependency_graph": True}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "output": str(output), "receipt": str(receipt)}))


if __name__ == "__main__":
    main()
