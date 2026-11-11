"""Zero-fit independent audit of the exploratory equal-word prefix contrast."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re

from human_ai.data import load_partitions, sha256
from .verify_roberta import Checks, auc, read_csv

ROOT = Path(__file__).resolve().parents[2]


def quantile(values, p):
    ordered = sorted(values)
    x = (len(ordered) - 1) * p
    i, j = math.floor(x), math.ceil(x)
    return ordered[i] + (x - i) * (ordered[j] - ordered[i])


def verify(study, output):
    if output.exists():
        raise FileExistsError(output)
    directory = study / "exploratory_prefix_whitespace"
    fit = study / "fits/canonical_8000"
    read = lambda path: json.loads(path.read_text())
    metric = read(directory / "metrics.json")
    provenance = read(directory / "provenance.json")
    plan = read(study / "exploratory_prefix_whitespace_plan.json")
    checks = Checks()
    checks.require(read(study / "FINAL_AUDIT.json")["status"] == "passed", "Primary audit did not pass")
    checks.require(read(study / "pipeline_completion.json")["status"] == "complete", "Pipeline incomplete")
    checks.require(plan["post_hoc"] and metric["post_hoc"], "Exploratory status missing")
    for name in ("new_model_fits", "optimizer_updates"):
        checks.require(metric[name] == provenance[name] == 0, f"Unexpected {name}")
    checks.require(provenance["diagnostic_plan_sha256"] == sha256(study / "exploratory_prefix_whitespace_plan.json"), "Diagnostic plan changed")
    checks.require(provenance["primary_plan_sha256"] == sha256(study / "plan.json"), "Primary plan changed")
    checks.require(provenance["checkpoint_sha256"] == sha256(fit / "model.pt"), "Checkpoint changed")
    for relative, digest in plan["source_sha256"].items():
        checks.require(sha256(ROOT / relative) == digest, f"Source changed: {relative}")
    for name, digest in read(directory / "validation_receipt.json")["files_sha256"].items():
        checks.require(sha256(directory / name) == digest, f"Evidence changed: {name}")
    checks.require(read(directory / "fallback_receipt.json")["captured_fallback_warning_count"] == 0, "Python fallback warning")
    checks.require(read(directory / "warnings.json") == [], "Unexpected Python warnings")
    receipts = read(study / "postprocess_execution_receipts.json")
    receipt = [r for r in receipts if r["stage"] == "declared_prefix_whitespace_inference"][-1]
    checks.require(receipt["return_code"] == 0, "Diagnostic process failed")
    for part in ("stdout", "stderr"):
        path = ROOT / receipt[f"{part}_path"]
        checks.require(sha256(path) == receipt[f"{part}_sha256"], "Native log changed")
        checks.require(not re.search(r"not currently supported on[^\n]*mps|will fall back[^\n]*cpu|mps[^\n]*(?:fallback|fall back)[^\n]*cpu", path.read_text(), re.I), "Native fallback warning")
    _, held = load_partitions()
    eligible = [r for r in held if len(r["text"].split()) >= 150]
    ids = [r["id"] for r in eligible]
    rows = read_csv(directory / "predictions.csv")
    mapping = {(r["condition"], r["id"]): r for r in rows}
    checks.require(len(mapping) == len(rows) == 2 * len(ids), "Duplicate or missing predictions")
    checks.require(set(mapping) == {(c, i) for c in ("joined", "raw") for i in ids}, "Unexpected prediction IDs")
    cached = {(r["condition"], r["id"]): r for r in read_csv(fit / "predictions.csv")}
    labels = [int(r["label"]) for r in eligible]
    scores = {c: [float(mapping[(c, i)]["logit_margin"]) for i in ids] for c in ("joined", "raw")}
    changed = 0
    for original in eligible:
        words = original["text"].split()[:100]
        end = list(re.finditer(r"\S+", original["text"]))[99].end()
        raw, joined = original["text"][:end], " ".join(words)
        checks.require(raw.split() == joined.split() == words, "Word sequences differ")
        changed += raw != joined
        digest = hashlib.sha256("\n".join(words).encode()).hexdigest()
        for condition in ("joined", "raw"):
            row = mapping[(condition, original["id"])]
            checks.require(int(row["label"]) == int(original["label"]), "Label mismatch")
            checks.require(row["word_sequence_sha256"] == digest, "Word digest differs")
            margin, probability = float(row["logit_margin"]), float(row["p_ai"])
            expected = 1 / (1 + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1 + math.exp(margin))
            checks.close(probability, expected, "Probability orientation differs", 1e-12)
            checks.require(int(row["tokens_kept"]) == min(int(row["token_count_full"]), 512), "Token coverage mismatch")
            if condition == "joined":
                old = cached[("prefix_100", original["id"])]
                checks.close(margin, float(old["logit_margin"]), "Joined logit replay differs", 1e-6)
                checks.close(probability, float(old["p_ai"]), "Joined probability replay differs", 1e-6)
    checks.require(changed == metric["original_whitespace_changed_n"], "Changed-whitespace count differs")
    checks.require(len(ids) == metric["n"], "Eligible size differs")
    for condition in ("joined", "raw"):
        checks.close(auc(labels, scores[condition]), metric["conditions"][condition]["auroc_logit_margin"], "Independent AUC differs", 1e-12)
    rng = random.Random(plan["seed"])
    groups = [[i for i, label in enumerate(labels) if label == target] for target in (0, 1)]
    differences = []
    for _ in range(plan["bootstrap_repetitions"]):
        draw = [group[rng.randrange(len(group))] for group in groups for _ in group]
        y = [labels[i] for i in draw]
        differences.append(auc(y, [scores["raw"][i] for i in draw]) - auc(y, [scores["joined"][i] for i in draw]))
    bootstrap = read(directory / "paired_bootstrap.json")
    for a, b in zip(differences, bootstrap["replicate_deltas"], strict=True):
        checks.close(a, b, "Independent bootstrap replicate differs", 1e-12)
    interval = [quantile(differences, p) for p in (.025, .975)]
    for actual, expected in zip(interval, metric["paired_95_interval"], strict=True):
        checks.close(actual, expected, "Independent interval differs", 1e-12)
    clean = [float(cached[("clean", i)]["logit_margin"]) for i in ids]
    result = {"status": "passed" if not checks.errors else "failed", "passed_checks": checks.passed,
              "errors": checks.errors, "new_model_fits": 0, "model_imports": 0,
              "n": len(ids), "original_clean_auc_same_eligible_ids": auc(labels, clean),
              "joined_auc": auc(labels, scores["joined"]), "raw_auc": auc(labels, scores["raw"]),
              "independent_paired_interval": interval, "verifier_source_sha256": sha256(__file__),
              "base_independent_auc_source_sha256": sha256(ROOT / "experiments/supervised/verify_roberta.py"),
              "scope": "Post-run arithmetic, IDs, equal-word intervention, replay and evidence hashes; no fresh inference or fit"}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))
    if checks.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.study_dir.resolve(), args.output.resolve())
