"""Audit a full-label RoBERTa submission without model imports or fitting."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re

from human_ai.data import ROOT, load_partitions, raw_path, read_csv, sha256
from human_ai.check_predictions import check
from .verify_roberta import Checks, ids_digest


def verify(directory, output):
    if output.exists():
        raise FileExistsError(output)
    load = lambda name: json.loads((directory / name).read_text())
    plan, config, metrics, completion = [load(n) for n in ("plan.json", "config.json", "metrics.json", "completion.json")]
    submission = Path(plan["output"])
    metadata = json.loads(submission.with_suffix(".metadata.json").read_text())
    checks = Checks()
    checks.require(completion["status"] == metrics["status"] == "complete", "Export incomplete")
    checks.require(check(submission, "submission") == 3000, "Invalid submission schema/IDs/probabilities")
    checks.require(config["plan_sha256"] == sha256(directory / "plan.json"), "Plan changed")
    for name, digest in plan["source_sha256"].items():
        checks.require(sha256(ROOT / name) == sha256(directory / "source" / name) == digest, f"Source changed: {name}")
    source = Path(plan["source_study"])
    for file, key in [("plan.json", "source_plan_sha256"), ("FINAL_AUDIT.json", "source_audit_sha256"),
                      ("analysis/aggregate_metrics.json", "source_metrics_sha256")]:
        checks.require(sha256(source / file) == plan[key], f"Selection evidence changed: {file}")
    train, validation = load_partitions()
    originals = train + validation
    ids = [r["id"] for r in originals]
    labels = {r["id"]: r["label"] for r in originals}
    saved = read_csv(directory / "train_ids.csv")
    checks.require([r["id"] for r in saved] == ids and len(set(ids)) == 10000, "Full-label ID order/coverage differs")
    checks.require(all(r["label"] == labels[r["id"]] for r in saved), "Full-label targets differ")
    checks.require(config["class_counts"] == {"0": 5000, "1": 5000}, "Incorrect class support")
    checks.require(sha256(directory / "train_ids.csv") == plan["training_ids_sha256"], "Training IDs changed")
    checks.require(sha256(raw_path("train.csv")) == plan["train_source_sha256"], "Training source changed")
    checks.require(plan["input_preprocessing"] == "Raw text; preserve original whitespace", "Preprocessing changed")
    checks.require(not plan["warm_start_from_development_checkpoint"] and not plan["test_fitting"], "Invalid warm-start/fitting scope")
    checks.require(metrics["held_out_evaluation"] is None and metadata["held_out_score"] is None, "False held-out estimate after full refit")
    checks.require(metadata["public_score"] is None and not metadata["uploaded_to_kaggle"], "Unmeasured leaderboard score/upload claim")
    steps = read_csv(directory / "training_steps.csv")
    checks.require(len(steps) == metrics["optimizer_updates"] == 1250, "Incorrect update count")
    epochs = load("epochs.json")
    checks.require(len(epochs) == 2, "Incorrect epoch count")
    for epoch in (1, 2):
        expected = list(range(10000))
        random.Random(42 + epoch).shuffle(expected)
        expected_ids = [ids[i] for i in expected]
        order = [r["id"] for r in read_csv(directory / f"epoch_{epoch}_order.csv")]
        checks.require(order == expected_ids, "Epoch order differs from prescribed seed")
        selected = [r for r in steps if int(r["epoch"]) == epoch]
        checks.require(len(selected) == 625 and sum(int(r["documents"]) for r in selected) == 10000, "Incomplete epoch")
        entry = epochs[epoch-1]
        checks.require(entry["originals"] == entry["unique_originals"] == 10000 and entry["updates"] == 625, "Epoch receipt differs")
        checks.require(entry["order_sha256"] == ids_digest(order), "Epoch digest differs")
        weighted_loss = 0.
        for index, row in enumerate(selected, start=1):
            checks.require(int(row["update_in_epoch"]) == index and int(row["update_global"]) == (epoch-1)*625+index, "Update order differs")
            checks.require(row["original_id_sequence_sha256"] == ids_digest(order[(index-1)*16:index*16]), "Update IDs differ")
            checks.require(int(row["documents"]) == 16 and json.loads(row["microbatch_sizes_json"]) == [4]*4, "Accumulation differs")
            checks.require(row["numeric_finite"] == row["logits_on_mps"] == "True", "Numeric/device check failed")
            checks.require(math.isfinite(float(row["loss"])) and math.isfinite(float(row["gradient_norm"])) and float(row["seconds"]) > 0, "Nonfinite training evidence")
            weighted_loss += float(row["loss"]) / 625
        checks.close(weighted_loss, entry["mean_loss"], "Epoch loss does not reconstruct", 1e-12)
    checks.require(all(float(x) > 0 for x in metrics["parameter_max_changes"].values()), "Encoder/classifier did not update")
    devices = metrics["optimizer_state_devices"]
    checks.require(devices.get("exp_avg:mps") == devices.get("exp_avg_sq:mps") == 201, "Optimizer moments did not stay on MPS")
    checks.require(metrics["reload_probability_max_error_all_test"] <= 1e-6 and metrics["reload_margin_max_error_all_test"] <= 1e-6, "Full test reload discrepancy")
    test = read_csv(raw_path("test.csv"))
    test_ids = [r["id"] for r in test]
    csv_rows, prediction_rows = read_csv(submission), read_csv(directory / "test_predictions.csv")
    checks.require([r["id"] for r in csv_rows] == [r["id"] for r in prediction_rows] == test_ids, "Test order differs")
    for submitted, predicted in zip(csv_rows, prediction_rows, strict=True):
        p, margin = float(predicted["p_ai"]), float(predicted["logit_margin"])
        expected = 1/(1+math.exp(-margin)) if margin >= 0 else math.exp(margin)/(1+math.exp(margin))
        checks.close(p, float(submitted["label"]), "Submission probability differs", 1e-14)
        checks.close(p, expected, "Class-1 probability orientation differs", 1e-12)
    checks.require(sha256(submission) == metrics["csv_sha256"] == metadata["csv_sha256"] == completion["csv_sha256"], "CSV checksum differs")
    checks.require(sha256(directory / "model.pt") == metrics["checkpoint_sha256"] == metadata["checkpoint_sha256"], "Model checksum differs")
    checks.require(sha256(raw_path("test.csv")) == metrics["test_source_sha256"], "Test source changed")
    checks.require(sha256(directory / "metrics.json") == completion["metrics_sha256"], "Metrics changed")
    checks.require(load("fallback_receipt.json")["python_fallback_warnings"] == [], "Python CPU fallback warning")
    receipt = load("execution_receipt.json")
    checks.require(receipt["return_code"] == 0, "Native process failed")
    for part in ("stdout", "stderr"):
        path = ROOT / receipt[f"{part}_path"]
        checks.require(sha256(path) == receipt[f"{part}_sha256"], "Native log checksum changed")
        checks.require(not re.search(r"not currently supported on[^\n]*mps|will fall back[^\n]*cpu|mps[^\n]*(?:fallback|fall back)[^\n]*cpu", path.read_text(), re.I), "Native CPU fallback warning")
    result = {"status": "passed" if not checks.errors else "failed", "passed_checks": checks.passed,
              "errors": checks.errors, "n_train": 10000, "n_test": 3000, "optimizer_updates": 1250,
              "csv": str(submission), "csv_sha256": sha256(submission), "source_sha256": sha256(__file__),
              "audit_scope": "Saved full-label training/evidence/CSV arithmetic and hashes; no model import, prediction or fitting",
              "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if checks.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir.resolve(), args.output.resolve())
