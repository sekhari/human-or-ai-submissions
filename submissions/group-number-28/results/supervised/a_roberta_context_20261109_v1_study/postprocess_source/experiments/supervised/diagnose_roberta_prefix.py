"""Post-hoc, zero-fit test of whitespace in an identical 100-word prefix.

The primary study is unchanged. Run only after all seven prescribed fits have
completed and the separate exploratory plan has been frozen. No optimizer,
backward pass, checkpoint selection, or test prediction is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
import traceback
import warnings


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_ai.data import load_partitions, read_csv, sha256
from experiments.supervised.train_roberta import atomic_csv, atomic_json, auc, id_digest


PREFIX_WORDS = 100
MINIMUM_WORDS = 150
PREDICTION_FIELDS = ["id", "label", "condition", "p_ai", "logit_margin", "cached_joined_p_ai",
                     "cached_joined_logit_margin", "word_sequence_sha256", "n_original_words",
                     "token_count_full", "tokens_kept", "truncated"]


def prefixes(text):
    """Preserve leading/interword whitespace through the 100th word's end."""
    words = text.split()
    if len(words) < MINIMUM_WORDS:
        return None
    matches = re.finditer(r"\S+", text)
    hundredth = None
    for number, match in enumerate(matches, start=1):
        if number == PREFIX_WORDS:
            hundredth = match
            break
    if hundredth is None:
        raise ValueError("Whitespace-token regex and str.split disagree on eligibility")
    raw = text[:hundredth.end()]
    joined = " ".join(words[:PREFIX_WORDS])
    if raw.split() != words[:PREFIX_WORDS] or joined.split() != raw.split():
        raise ValueError("Raw and joined prefixes must contain exactly the same 100 word tokens")
    return {"raw": raw, "joined": joined, "n_original_words": len(words),
            "word_sequence_sha256": id_digest(words[:PREFIX_WORDS])}


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_bootstrap(labels, joined, raw, seed=551, repetitions=1000):
    """Resample the same original IDs across conditions, within true classes."""
    groups = [[i for i, label in enumerate(labels) if label == target] for target in (0, 1)]
    if not all(groups):
        return {"status": "insufficient_classes", "requested": repetitions, "valid": 0,
                "delta_raw_minus_joined": None, "interval_95": None}
    rng = random.Random(seed)
    deltas = []
    for _ in range(repetitions):
        draw = [group[rng.randrange(len(group))] for group in groups for _ in group]
        sampled_labels = [labels[i] for i in draw]
        before = auc(sampled_labels, [joined[i] for i in draw])
        after = auc(sampled_labels, [raw[i] for i in draw])
        if before is not None and after is not None:
            deltas.append(after - before)
    return {"status": "complete" if len(deltas) == repetitions else "incomplete",
            "seed": seed, "requested": repetitions, "valid": len(deltas),
            "delta_raw_minus_joined": auc(labels, raw) - auc(labels, joined),
            "interval_95": [quantile(deltas, 0.025), quantile(deltas, 0.975)] if deltas else None,
            "method": "Paired original-ID bootstrap, class-stratified with replacement; percentile linear quantiles",
            "limitations": "Conditional on this fitted model and reused eligible originals; post-hoc, no selection/multiplicity correction",
            "replicate_deltas": deltas}


def validate_exploratory_plan(study, fit_key):
    path = study / "exploratory_prefix_whitespace_plan.json"
    plan = json.loads(path.read_text())
    required = {"post_hoc": True, "basis_fit_key": "random_0", "fit_key": "canonical_8000",
                "seed": 551, "bootstrap_repetitions": 1000,
                "minimum_eligible_original_words": MINIMUM_WORDS, "prefix_words": PREFIX_WORDS,
                "sequence_length": 512, "microbatch": 4, "dtype": "float32", "attention": "sdpa",
                "model_fits": 0}
    if fit_key != "canonical_8000" or any(plan.get(key) != value for key, value in required.items()):
        raise ValueError("The separate fixed post-hoc diagnostic plan does not match this execution")
    if plan["primary_study_plan_sha256"] != sha256(study / "plan.json"):
        raise ValueError("The primary study plan changed")
    own_name = "experiments/supervised/diagnose_roberta_prefix.py"
    if plan["source_sha256"].get(own_name) != sha256(__file__):
        raise ValueError("The diagnostic source must be frozen before execution")
    for relative, expected in plan["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"A frozen diagnostic source changed: {relative}")
    if not plan.get("single_changed_factor"):
        raise ValueError("The diagnostic must declare its single changed factor")
    return plan, path


def run(args, output):
    if Path(sys.executable).parent != Path("/opt/anaconda3/envs/python_course/bin"):
        raise RuntimeError("Use the user-selected python_course interpreter")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("Start with generic unsupported-op fallback disabled")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    started = time.perf_counter()
    study = args.study_dir.resolve()
    plan, plan_path = validate_exploratory_plan(study, args.fit_key)
    primary_plan = json.loads((study / "plan.json").read_text())
    fits = primary_plan["fits"]
    if len(fits) != 7:
        raise ValueError("Expected the complete prescribed seven-fit primary comparison")
    for definition in fits:
        completion = json.loads((study / "fits" / definition["key"] / "completion.json").read_text())
        if completion["status"] != "complete" or completion["epochs"] != 2:
            raise ValueError("All seven primary fits must finish before this post-hoc diagnostic")
    fit = study / "fits" / args.fit_key
    fit_config = json.loads((fit / "config.json").read_text())
    fit_metrics = json.loads((fit / "metrics.json").read_text())
    checkpoint = fit / "model.pt"
    checkpoint_sha = sha256(checkpoint)
    if fit_metrics["epochs"] != 2 or fit_metrics["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("The fixed final-epoch scientific checkpoint differs")
    fitting, held = load_partitions()
    fitting_ids = [row["id"] for row in fitting]
    held_by_id = {row["id"]: row for row in held}
    saved_train = read_csv(fit / "train_ids.csv")
    saved_held = read_csv(fit / "held_ids.csv")
    if [row["id"] for row in saved_train] != fitting_ids:
        raise ValueError("Checkpoint training IDs are not the canonical 8,000 originals")
    held_ids = [row["id"] for row in saved_held]
    if set(held_ids) != set(held_by_id) or len(held_ids) != len(set(held_ids)):
        raise ValueError("Expected exact canonical held original IDs")
    if fit_config["n_train"] != 8000 or fit_config["n_held"] != 2000:
        raise ValueError("The diagnostic uses only the fixed canonical 8,000-trained model")
    records = []
    for identifier in held_ids:
        original = held_by_id[identifier]
        result = prefixes(original["text"])
        if result is not None:
            records.append({"id": identifier, "label": int(original["label"]), **result})
    if not records or {row["label"] for row in records} != {0, 1}:
        raise ValueError("The eligible original set must contain both classes")
    cached_rows = [row for row in read_csv(fit / "predictions.csv") if row["condition"] == "prefix_100"]
    cached = {row["id"]: row for row in cached_rows}
    if len(cached) != len(cached_rows) or set(cached) != {row["id"] for row in records}:
        raise ValueError("Existing joined-prefix predictions must cover the identical eligible originals")
    if any(int(cached[row["id"]]["label"]) != row["label"] for row in records):
        raise ValueError("Cached and canonical eligible labels differ")
    provenance = {"diagnostic_plan_sha256": sha256(plan_path), "primary_plan_sha256": sha256(study / "plan.json"),
        "fit_config_sha256": sha256(fit / "config.json"), "fit_metrics_sha256": sha256(fit / "metrics.json"),
        "cached_predictions_sha256": sha256(fit / "predictions.csv"), "checkpoint_sha256": checkpoint_sha,
        "canonical_manifest_sha256": sha256(ROOT / "data/splits/manifest.json"),
        "source_sha256": {"experiments/supervised/diagnose_roberta_prefix.py": sha256(__file__),
                          "experiments/supervised/train_roberta.py": sha256(ROOT / "experiments/supervised/train_roberta.py"),
                          "human_ai/data.py": sha256(ROOT / "human_ai/data.py")},
        "eligible_id_sequence_sha256": id_digest([row["id"] for row in records]),
        "new_model_fits": 0, "optimizer_updates": 0, "test_inputs_used": False,
        "post_hoc": True, "checkpoint_selection": "Fixed canonical_8000 final epoch2, unchanged from primary study",
        "python": sys.executable, "packages": {name: version(name) for name in (
            "torch", "transformers", "huggingface-hub", "tokenizers", "numpy", "safetensors")}}
    atomic_json(output / "provenance.json", provenance)
    atomic_csv(output / "eligible_ids.csv", [{"id": row["id"], "label": row["label"],
        "n_original_words": row["n_original_words"], "word_sequence_sha256": row["word_sequence_sha256"]}
        for row in records], ["id", "label", "n_original_words", "word_sequence_sha256"])
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    torch.manual_seed(42)
    torch.mps.manual_seed(42)
    torch.set_num_threads(10)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(fit / "model_config", local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(config, attn_implementation="sdpa")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = model.float().to("mps").eval()
    if model.training or any(parameter.device.type != "mps" or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("The diagnostic model must be in evaluation mode, FP32, on MPS")
    tokenizer = AutoTokenizer.from_pretrained(Path(primary_plan["model"]["snapshot"]), local_files_only=True)
    predictions, values, inference_times = [], {}, {}
    joined_max_margin_error, joined_max_probability_error = None, None
    for name in ("joined", "raw"):
        # The joined condition is completed and checked against the saved primary
        # predictions before the new raw-whitespace condition is evaluated.
        texts = [row[name] for row in records]
        full_encoded = tokenizer(texts, truncation=False, padding=False, return_attention_mask=False)
        full_lengths = [len(tokens) for tokens in full_encoded["input_ids"]]
        del full_encoded
        encoded = tokenizer(texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        encoded = {key: tensor for key, tensor in encoded.items() if key in ("input_ids", "attention_mask", "token_type_ids")}
        kept = encoded["attention_mask"].sum(dim=1).tolist()
        margins, probabilities = [], []
        torch.mps.synchronize()
        condition_started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(records), 4):
                batch = {key: tensor[start:start + 4].to("mps") for key, tensor in encoded.items()}
                logits = model(**batch).logits
                if logits.device.type != "mps":
                    raise RuntimeError("Diagnostic inference logits left MPS")
                host_logits = logits.detach().cpu().to(dtype=torch.float64)
                if not torch.isfinite(host_logits).all().item():
                    raise RuntimeError("Nonfinite diagnostic logits")
                margins.extend((host_logits[:, 1] - host_logits[:, 0]).tolist())
                probabilities.extend(host_logits.softmax(-1)[:, 1].tolist())
        torch.mps.synchronize()
        inference_times[name] = time.perf_counter() - condition_started
        if name == "joined":
            joined_max_margin_error = max(abs(value - float(cached[row["id"]]["logit_margin"])) for row, value in zip(records, margins, strict=True))
            joined_max_probability_error = max(abs(value - float(cached[row["id"]]["p_ai"])) for row, value in zip(records, probabilities, strict=True))
            atomic_json(output / "joined_reproduction_receipt.json", {
                "n": len(records), "maximum_logit_margin_error": joined_max_margin_error,
                "maximum_probability_error": joined_max_probability_error, "tolerance": 1e-6,
                "passed": joined_max_margin_error <= 1e-6 and joined_max_probability_error <= 1e-6})
            if joined_max_margin_error > 1e-6 or joined_max_probability_error > 1e-6:
                raise RuntimeError("Joined-prefix reproduction exceeds the fixed 1e-6 tolerance; raw condition was not evaluated")
        values[name] = {"margins": margins, "probabilities": probabilities,
                        "full_lengths": full_lengths, "kept": kept}
        predictions.extend({"id": row["id"], "label": row["label"], "condition": name,
            "p_ai": format(probability, ".17g"), "logit_margin": format(margin, ".17g"),
            "cached_joined_p_ai": cached[row["id"]]["p_ai"],
            "cached_joined_logit_margin": cached[row["id"]]["logit_margin"],
            "word_sequence_sha256": row["word_sequence_sha256"], "n_original_words": row["n_original_words"],
            "token_count_full": full_length, "tokens_kept": int(retained), "truncated": full_length > 512}
            for row, probability, margin, full_length, retained in zip(records, probabilities, margins, full_lengths, kept, strict=True))
        atomic_csv(output / "predictions.csv", predictions, PREDICTION_FIELDS)
        print(json.dumps({"phase": "condition_complete", "condition": name, "n": len(records),
                          "inference_seconds": inference_times[name]}), flush=True)
    labels = [row["label"] for row in records]
    bootstrap = paired_bootstrap(labels, values["joined"]["margins"], values["raw"]["margins"],
                                 seed=plan["seed"], repetitions=plan["bootstrap_repetitions"])
    atomic_json(output / "paired_bootstrap.json", bootstrap)
    margin_changes = [after - before for before, after in zip(values["joined"]["margins"], values["raw"]["margins"], strict=True)]
    probability_changes = [after - before for before, after in zip(values["joined"]["probabilities"], values["raw"]["probabilities"], strict=True)]
    covered = [i for i, (joined_length, raw_length) in enumerate(zip(
        values["joined"]["full_lengths"], values["raw"]["full_lengths"], strict=True))
        if joined_length <= 512 and raw_length <= 512]
    covered_labels = [labels[i] for i in covered]
    covered_joined_auc = auc(covered_labels, [values["joined"]["margins"][i] for i in covered])
    covered_raw_auc = auc(covered_labels, [values["raw"]["margins"][i] for i in covered])
    metrics = {"status": "complete", "post_hoc": True, "fit_key": args.fit_key,
        "single_changed_factor": plan["single_changed_factor"], "new_model_fits": 0, "optimizer_updates": 0,
        "n": len(records), "n_human": labels.count(0), "n_ai": labels.count(1),
        "primary_ranking_score": "Raw class1-class0 logits after FP32->CPU->float64 transfer",
        "conditions": {name: {"auroc_logit_margin": auc(labels, result["margins"]),
            "auroc_probability": auc(labels, result["probabilities"]), "inference_seconds": inference_times[name],
            "mean_full_token_count": statistics.mean(result["full_lengths"]), "mean_tokens_kept": statistics.mean(result["kept"]),
            "n_truncated_at512": sum(value > 512 for value in result["full_lengths"])} for name, result in values.items()},
        "auroc_delta_raw_minus_joined": auc(labels, values["raw"]["margins"]) - auc(labels, values["joined"]["margins"]),
        "paired_95_interval": bootstrap["interval_95"], "bootstrap_valid_replicates": bootstrap["valid"],
        "secondary_fully_covered_in_both_variants": {"n": len(covered), "n_human": covered_labels.count(0),
            "n_ai": covered_labels.count(1), "joined_auroc_logit_margin": covered_joined_auc,
            "raw_auroc_logit_margin": covered_raw_auc,
            "delta_raw_minus_joined": covered_raw_auc - covered_joined_auc
                if covered_raw_auc is not None and covered_joined_auc is not None else None,
            "interpretation": "Descriptive paired subset with both complete prefixes retained at512; does not filter or replace the all-eligible primary contrast"},
        "joined_reproduction_max_logit_margin_error": joined_max_margin_error,
        "joined_reproduction_max_probability_error": joined_max_probability_error,
        "mean_logit_margin_change": statistics.mean(margin_changes),
        "mean_absolute_logit_margin_change": statistics.mean(abs(value) for value in margin_changes),
        "maximum_absolute_logit_margin_change": max(abs(value) for value in margin_changes),
        "mean_probability_change": statistics.mean(probability_changes),
        "mean_absolute_probability_change": statistics.mean(abs(value) for value in probability_changes),
        "maximum_absolute_probability_change": max(abs(value) for value in probability_changes),
        "classification_flip_fraction_at_probability0_5": sum((before >= .5) != (after >= .5) for before, after in zip(
            values["joined"]["probabilities"], values["raw"]["probabilities"], strict=True)) / len(records),
        "original_whitespace_changed_n": sum(row["raw"] != row["joined"] for row in records),
        "process_elapsed_seconds": time.perf_counter() - started,
        "limitations": ["Post-hoc hypothesis motivated by the first primary fold; no untouched confirmatory claim",
            "One fixed canonical model and reused eligible held originals; no training variation",
            "Not evidence about the competition hidden text distribution or causal authorship",
            "No primary recipe/checkpoint/selection rule is changed by this diagnostic",
            "CPU preprocessing/host transfers remain expected; inspect native logs for unsupported-op fallback"]}
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "validation_receipt.json", {"new_model_fits": 0, "optimizer_updates": 0,
        "test_inputs_used": False, "raw_and_joined_word_sequences_identical": True,
        "all_logits_finite": True, "model_eval_mode": True, "inference_mode": True,
        "joined_reproduction_passed": True, "joined_reproduction_tolerance": 1e-6,
        "checkpoint_sha256": checkpoint_sha, "diagnostic_plan_sha256": sha256(plan_path),
        "files_sha256": {name: sha256(output / name) for name in (
            "provenance.json", "eligible_ids.csv", "predictions.csv", "joined_reproduction_receipt.json", "paired_bootstrap.json", "metrics.json")}})
    return metrics


def self_test():
    cases = [
        " ".join(f"word{i}" for i in range(150)),
        "\n\t  " + "\n\t".join(f"word{i}，!?" for i in range(150)),
        "\u00a0\u2003" + "\u00a0\u2003".join(f"词{i}、。" for i in range(150)),
        "\n" + "\t ".join(f"word{i}“—”" for i in range(151)),
    ]
    for text in cases:
        result = prefixes(text)
        assert result is not None
        assert result["raw"].split() == result["joined"].split() == text.split()[:100]
        assert len(result["raw"].split()) == 100
        assert result["raw"] == text[:list(re.finditer(r"\S+", text))[99].end()]
        assert result["word_sequence_sha256"] == id_digest(text.split()[:100])
    assert prefixes(" ".join("word" for _ in range(149))) is None
    assert prefixes("") is None
    assert prefixes("\n\t\u00a0") is None
    labels, joined, raw = [0, 0, 1, 1], [0., 2., 1., 3.], [0., 1., 2., 3.]
    first = paired_bootstrap(labels, joined, raw, seed=551, repetitions=10)
    second = paired_bootstrap(labels, joined, raw, seed=551, repetitions=10)
    assert first == second and first["valid"] == 10
    assert first["delta_raw_minus_joined"] == 0.25
    assert quantile([0., 1.], .025) == .025
    print("Prefix/eligibility/Unicode-whitespace and paired-bootstrap checks passed; zero model imports/fits.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path)
    parser.add_argument("--fit-key", default="canonical_8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.study_dir is None or args.output is None:
        parser.error("--study-dir and --output are required unless --self-test is used")
    if args.fit_key != "canonical_8000":
        parser.error("This frozen exploratory diagnostic uses canonical_8000 only")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            metrics = run(args, output)
        except Exception as error:
            atomic_json(output / "failure.json", {"status": "failed", "error": repr(error),
                "traceback": traceback.format_exc(), "source_sha256": sha256(__file__),
                "failed_at_utc": datetime.now(timezone.utc).isoformat()})
            raise
        finally:
            notes = [{"category": note.category.__name__, "message": str(note.message)} for note in caught]
            atomic_json(output / "warnings.json", notes)
    fallback = [note for note in notes if any(term in note["message"].lower() for term in (
        "fallback", "fall back", "falling back", "not currently supported on the mps"))]
    atomic_json(output / "fallback_receipt.json", {"environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "captured_fallback_warning_count": len(fallback), "captured_fallback_warnings": fallback,
        "native_stdout_stderr_review_required": True,
        "claim": "No unsupported-op fallback observed only if native logs also have no matches; ordinary CPU work is expected"})
    atomic_json(output / "completion.json", {"status": "complete", "new_model_fits": 0,
        "optimizer_updates": 0, "post_hoc": True, "n": metrics["n"],
        "metrics_sha256": sha256(output / "metrics.json"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "complete", "output": str(output), "post_hoc": True,
                      "auroc_delta_raw_minus_joined": metrics["auroc_delta_raw_minus_joined"]}), flush=True)


if __name__ == "__main__":
    main()
