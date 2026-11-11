"""Independently verify a completed MPS device benchmark without loading a model.

Uses the Python standard library only. This validates the recorded evidence and
arithmetic; it does not rerun training or prove that unobserved operators never
use CPU implementations.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
NATIVE_FALLBACK = re.compile(
    r"not currently supported on[^\n]*mps|will fall back[^\n]*cpu|"
    r"mps[^\n]*(?:fallback|fall back|falling back)[^\n]*cpu", re.IGNORECASE
)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return rows


class Checks:
    def __init__(self):
        self.passed = 0
        self.errors = []

    def require(self, condition, message):
        if condition:
            self.passed += 1
        else:
            self.errors.append(message)

    def close(self, actual, expected, message):
        self.require(math.isfinite(float(actual)) and math.isfinite(float(expected))
                     and math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9), message)


def verify(args, checks):
    folder = args.run_dir.resolve()
    plan = load_json(folder / "plan.json")
    metrics = load_json(folder / "metrics.json")
    receipt = load_json(folder / "validation_receipt.json")
    fallback = load_json(folder / "fallback_receipt.json")
    warnings = load_json(folder / "warnings.json")
    summary = load_json(folder / "memory_summary.json")
    steps = load_csv(folder / "steps.csv")
    samples = load_csv(folder / "memory_samples.csv")
    originals = load_csv(folder / "benchmark_original_ids.csv")
    stdout = args.stdout_log.read_text(errors="replace")
    stderr = args.stderr_log.read_text(errors="replace")

    checks.require(not (folder / "failure.json").exists(), "A failure receipt exists")
    checks.require(metrics["status"] == "completed" and metrics["benchmark_only"] is True,
                   "Run must be a completed device benchmark")
    checks.require(plan["benchmark_only"] is True and plan["task_auroc_evaluated"] is False
                   and plan["validation_or_test_inputs_used"] is False, "Unexpected model-scoring scope")
    checks.require(Path(plan["python"]).resolve() == Path("/opt/anaconda3/envs/python_course/bin/python").resolve(),
                   "Benchmark did not use python_course")
    checks.require(plan["mps_built"] is True and plan["mps_available"] is True,
                   "MPS availability was not confirmed")
    checks.require(plan["dtype"] == "float32" and plan["attention_implementation"] == "sdpa",
                   "Unexpected precision or attention implementation")
    frozen_source = folder.parent / "source/experiments/supervised/benchmark_roberta_mps.py"
    source = frozen_source if frozen_source.is_file() else ROOT / "experiments/supervised/benchmark_roberta_mps.py"
    checks.require(plan["source_sha256"] == sha256(source),
                   "Worker source differs from the executed source hash")
    checks.require(plan["canonical_split_manifest_sha256"] == sha256(ROOT / "data/splits/manifest.json"),
                   "Canonical split manifest hash differs")
    checks.require(plan["selected_ids_sha256"] == sha256(folder / "benchmark_original_ids.csv"),
                   "Selected-ID artifact hash differs")
    snapshot = Path(plan["snapshot"])
    weights = snapshot / "model.safetensors"
    checks.require(weights.stat().st_size == plan["weights_bytes"] and sha256(weights) == plan["weights_sha256"],
                   "Pinned input weights differ from the recorded weights")
    checks.require(snapshot.name == plan["snapshot_revision"], "Snapshot revision does not match its directory")
    for name, expected in receipt["files_sha256"].items():
        checks.require(Path(name).name == name and sha256(folder / name) == expected,
                       f"Validation receipt hash mismatch: {name}")
    checkpoint = folder / "model.pt"
    checks.require(checkpoint.stat().st_size == metrics["checkpoint_bytes"]
                   and sha256(checkpoint) == metrics["checkpoint_sha256"], "Benchmark checkpoint hash differs")

    expected_fields = {"id", "label"}
    checks.require(bool(originals) and set(originals[0]) == expected_fields, "ID artifact has unexpected columns")
    ids = [row["id"] for row in originals]
    canonical = {row["id"] for row in load_csv(ROOT / "data/splits/train_ids.csv")}
    held = {row["id"] for row in load_csv(ROOT / "data/splits/validation_ids.csv")}
    checks.require(len(ids) == len(set(ids)) == plan["n_unique_benchmark_originals"],
                   "Selected originals count or uniqueness differs")
    checks.require(set(ids) <= canonical and not set(ids) & held, "Benchmark includes non-fitting IDs")
    checks.require(all(row["label"] in ("0", "1") for row in originals), "Unexpected benchmark labels")
    batch, accum, length = plan["batch_size"], plan["gradient_accumulation"], plan["sequence_length"]
    effective_batch = batch * accum
    checks.require(plan["effective_batch_size"] == effective_batch, "Effective batch arithmetic differs")
    checks.require(plan["warmup_optimizer_updates"] == 5 and plan["timed_optimizer_updates"] == 20,
                   "Expected exactly 5 warmup and 20 measured optimizer updates")
    checks.require(len(steps) == 25, "Step artifact does not contain 25 optimizer updates")
    cursor = 0
    for index, row in enumerate(steps):
        warmup = index < 5
        checks.require(row["phase"] == ("warmup" if warmup else "timed")
                       and int(row["optimizer_update"]) == (index + 1 if warmup else index - 4),
                       f"Unexpected step ordering at row {index + 1}")
        checks.require(int(row["documents"]) == effective_batch
                       and int(row["padded_tokens"]) == effective_batch * length,
                       f"Step document/padded-token count differs at row {index + 1}")
        counts = json.loads(row["effective_token_counts_json"])
        checks.require(len(counts) == effective_batch and all(isinstance(value, int) and 0 < value <= length for value in counts)
                       and sum(counts) == int(row["effective_tokens"]),
                       f"Step effective-token count differs at row {index + 1}")
        processed = [ids[(cursor + j) % len(ids)] for j in range(effective_batch)]
        cursor += effective_batch
        checks.require(hashlib.sha256("\n".join(processed).encode()).hexdigest() == row["original_id_sequence_sha256"],
                       f"Step ID sequence differs at row {index + 1}")
        checks.require(row["loss_and_gradient_norm_finite"] == "True", f"Step numeric check failed at row {index + 1}")
        for name in ("seconds", "mean_loss", "preclip_gradient_norm"):
            value = float(row[name])
            checks.require(math.isfinite(value) and value > 0, f"Invalid {name} at row {index + 1}")
        for name in row:
            if name.startswith("host_") and name.endswith("_seconds"):
                checks.require(math.isfinite(float(row[name])) and float(row[name]) >= 0,
                               f"Invalid host duration {name} at row {index + 1}")

    timed = [row for row in steps if row["phase"] == "timed"]
    durations = [float(row["seconds"]) for row in timed]
    seconds = sum(durations)
    docs = sum(int(row["documents"]) for row in timed)
    checks.require(metrics["timed_optimizer_updates"] == len(timed) == 20
                   and metrics["timed_microsteps"] == 20 * accum, "Measured update or microstep count differs")
    checks.close(metrics["timed_update_seconds_sum"], seconds, "Update duration sum differs")
    for name, value in (("mean_update_seconds", statistics.mean(durations)),
                        ("median_update_seconds", statistics.median(durations)),
                        ("minimum_update_seconds", min(durations)),
                        ("maximum_update_seconds", max(durations)),
                        ("documents_processed", docs), ("documents_per_second", docs / seconds),
                        ("effective_tokens_per_second", sum(int(row["effective_tokens"]) for row in timed) / seconds),
                        ("padded_tokens_per_second", sum(int(row["padded_tokens"]) for row in timed) / seconds)):
        checks.close(metrics[name], value, f"Metric arithmetic differs: {name}")
    checks.require(math.isfinite(metrics["timed_loop_wall_seconds"]) and metrics["timed_loop_wall_seconds"] >= seconds,
                   "Complete-loop walltime is shorter than measured updates")
    checks.close(metrics["end_to_end_loop_documents_per_second"], docs / metrics["timed_loop_wall_seconds"],
                 "Complete-loop throughput differs")

    checks.require(metrics["memory"] == summary, "Metrics and standalone memory summaries differ")
    checks.require(summary["sample_errors"] == [] and summary["n_samples"] == len(samples) and len(samples) > 0,
                   "Memory sampling failed or sample count differs")
    fields = ("mps_current_allocated_bytes", "mps_driver_allocated_bytes", "process_rss_bytes",
              "system_available_bytes", "system_swap_used_bytes")
    for phase in ("all_training", "timed"):
        subset = samples if phase == "all_training" else [row for row in samples if row["phase"] == "timed"]
        checks.require(bool(subset), f"No memory samples for {phase}")
        for name in fields:
            values = [int(row[name]) for row in subset]
            checks.require(all(value >= 0 for value in values), f"Negative memory sample: {name}")
            checks.require(summary[phase][name] == {"minimum": min(values), "maximum": max(values)},
                           f"Sampled memory extrema differ: {phase}/{name}")
    checks.require(metrics["all_model_gradients_mps"] is True
                   and receipt["gradients_on_mps"] is True, "Model gradient device check failed")
    checks.require(receipt["forward_backward_optimizer_executed"] is True
                   and receipt["all_numeric_checks_passed"] is True
                   and receipt["encoder_and_classifier_parameters_updated"] is True, "Training path receipt did not pass")
    deltas = metrics["monitored_parameter_max_updates"]
    checks.require(len(deltas) == 2 and any("encoder.layer.0" in name for name in deltas)
                   and any("classifier.out_proj.weight" in name for name in deltas)
                   and all(math.isfinite(value) and value > 0 for value in deltas.values()),
                   "Encoder/classifier parameter update evidence differs")
    state = metrics["optimizer_state_devices"]
    checks.require(bool(state) and all(key.split(":")[0] == "step" or key.split(":")[1] == "mps" for key in state),
                   "Optimizer non-step tensor state left MPS")
    checks.require(any(key.startswith("exp_avg:mps:") for key in state)
                   and any(key.startswith("exp_avg_sq:mps:") for key in state), "AdamW MPS moment state is absent")
    probabilities = {}
    for name in ("initial_eval_probabilities", "final_eval_probabilities", "reloaded_eval_probabilities"):
        values = receipt[name]
        checks.require(len(values) == min(2, len(ids)) and all(len(row) == 2 for row in values),
                       f"Unexpected shape for {name}")
        checks.require(all(math.isfinite(value) and 0 <= value <= 1 for row in values for value in row)
                       and all(math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6) for row in values),
                       f"Invalid probability distribution: {name}")
        probabilities[name] = values
    final = probabilities["final_eval_probabilities"]
    restored = probabilities["reloaded_eval_probabilities"]
    reload_error = max(abs(a - b) for row_a, row_b in zip(final, restored) for a, b in zip(row_a, row_b))
    checks.close(receipt["reload_max_error"], reload_error, "Receipt reload difference differs")
    checks.close(metrics["reload_max_probability_error"], reload_error, "Metrics reload difference differs")
    checks.require(reload_error <= receipt["reload_tolerance"] <= 1e-6, "Reload difference exceeds tolerance")

    checks.require(plan["fallback_environment_before_torch_import"] == "0" and fallback["environment"] == "0"
                   and fallback["unsupported_op_fallback_enabled"] is False, "Unsupported-op fallback was enabled")
    matched_warnings = [note for note in warnings if any(word in note["message"].lower()
                       for word in ("fallback", "fall back", "falling back", "not currently supported on the mps"))]
    checks.require(fallback["captured_fallback_warnings"] == matched_warnings
                   and fallback["captured_fallback_warning_count"] == len(matched_warnings), "Fallback warning receipt differs")
    checks.require(not matched_warnings, "Captured Python fallback warning requires manual review")
    native_matches = NATIVE_FALLBACK.findall(stdout + "\n" + stderr)
    checks.require(not native_matches, "Native stdout/stderr contains MPS CPU-fallback evidence")
    log_receipts = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("phase") in ("warmup", "timed") and "update" in value:
            log_receipts.append(value)
    checks.require(len(log_receipts) == 25, "stdout does not record 25 completed updates")
    checks.require(any('"status": "completed"' in line for line in stdout.splitlines()),
                   "stdout has no final completion receipt")

    projection = None
    if (batch, length, accum) == (4, 512, 4):
        fold_source = ROOT / "results/supervised/a_q5_crop_revised_20261109_v6_study/fold_counts.csv"
        folds = load_csv(fold_source)
        cv_train_sizes = [int(row["fitting_n"]) for row in folds]
        checks.require(len(canonical) == 8000 and len(folds) == 6
                       and sorted(cv_train_sizes) == [4646, 4646, 5394, 5394, 5960, 5960],
                       "Existing full/CV dataset sizes differ from the intended projection")
        full_updates = 2 * math.ceil(len(canonical) / 16)
        cv_updates = 2 * sum(math.ceil(size / 16) for size in cv_train_sizes)
        mean_loop = metrics["timed_loop_wall_seconds"] / 20
        projection = {
            "effective_batch_size": 16, "epochs": 2,
            "full_8000_optimizer_updates": full_updates,
            "six_outer_fits_optimizer_updates": cv_updates,
            "six_outer_fits_train_sizes": cv_train_sizes,
            "fold_sizes_source": str(fold_source.relative_to(ROOT)),
            "fold_sizes_source_sha256": sha256(fold_source),
            "seconds_per_update_measured": seconds / 20,
            "seconds_per_update_including_benchmark_diagnostics": mean_loop,
            "full_8000_update_seconds": full_updates * seconds / 20,
            "full_8000_loop_seconds": full_updates * mean_loop,
            "six_outer_fits_update_seconds": cv_updates * seconds / 20,
            "six_outer_fits_loop_seconds": cv_updates * mean_loop,
            "excluded": ["download", "loading", "tokenization", "evaluation", "checkpoint I/O"],
            "uncertainty": "20 measured updates; no confidence interval or full-epoch thermal measurement. CV sizing assumes the existing matched random/topic training folds.",
        }
    return {
        "run_dir": str(folder), "configuration": {"batch_size": batch, "sequence_length": length, "grad_accum": accum},
        "verified_input_files_sha256": {"stdout": sha256(args.stdout_log), "stderr": sha256(args.stderr_log),
                                        "validation_receipt": sha256(folder / "validation_receipt.json")},
        "observed_native_fallback_matches": native_matches,
        "projection": projection,
        "limitations": ["This verifier inspects saved evidence without loading the model or rerunning any fit.",
                        "No observed unsupported-op CPU fallback is supported only for this exercised path.",
                        "Ordinary CPU activity and AdamW scalar step bookkeeping are expected.",
                        "Memory extrema are sampled observed maxima; unified-memory measurements overlap."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = Checks()
    result = {}
    try:
        result = verify(args, checks)
    except Exception as error:
        checks.errors.append(f"Verification could not complete: {type(error).__name__}: {error}")
    result.update(status="passed" if not checks.errors else "failed", passed_checks=checks.passed,
                  errors=checks.errors, verifier_source_sha256=sha256(__file__),
                  verifier_python=sys.executable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "passed_checks": checks.passed, "errors": checks.errors,
                      "output": str(args.output.resolve())}))
    return 0 if not checks.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
