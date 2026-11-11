"""One fixed-recipe RoBERTa fit; held predictions occur after the final epoch.

Use the user-selected python_course environment. Every process starts from the
pinned generic checkpoint. No model selection, external service, or test access
is performed here. A mechanical restart can use the last completed epoch only.
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import resource
import shutil
import sys
import time
import traceback
import warnings


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.supervised.benchmark_roberta_mps import MemorySampler
from human_ai.data import load_partitions, read_csv, sha256


PREDICTION_FIELDS = ["fit_key", "scheme", "fold", "condition", "id", "label", "p_ai", "logit_margin"]
STEP_FIELDS = ["epoch", "optimizer_update_in_epoch", "optimizer_update_global", "documents",
               "microbatch_sizes_json", "original_id_sequence_sha256", "effective_tokens", "seconds",
               "mean_loss", "preclip_gradient_norm", "loss_and_gradient_norm_finite",
               "logits_on_mps", "gradient_norm_on_mps"]
QUOTE_TRANSLATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
})
DASH_TRANSLATION = str.maketrans({"\u2013": "-", "\u2014": "-"})


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def id_digest(ids):
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def epoch_order(n_originals, seed, epoch):
    order = list(range(n_originals))
    random.Random(seed + epoch).shuffle(order)
    return order


def update_groups(order, microbatch=4, accumulation=4):
    """Pure helper: retain each original and weight partial updates by size."""
    effective = microbatch * accumulation
    for start in range(0, len(order), effective):
        update = order[start:start + effective]
        microbatches = [update[pos:pos + microbatch] for pos in range(0, len(update), microbatch)]
        yield update, microbatches, [len(micro) / len(update) for micro in microbatches]


def conditions(texts):
    """Match the five primary crop_evaluation conditions without fitting."""
    words = [text.split() for text in texts]
    indices = list(range(len(texts)))
    eligible = [i for i, tokens in enumerate(words) if len(tokens) >= 150]
    return {
        "clean": (indices, list(texts)),
        "quote_and_dash": (indices, [t.translate(QUOTE_TRANSLATION).translate(DASH_TRANSLATION) for t in texts]),
        "digit_runs_to_zero": (indices, [re.sub(r"\d+", "0", t) for t in texts]),
        "whitespace_only": (indices, [" ".join(tokens) for tokens in words]),
        "prefix_100": (eligible, [" ".join(words[i][:100]) for i in eligible]),
    }


def auc(labels, scores):
    """Mann-Whitney AUROC with half credit for tied ranking scores."""
    n1 = sum(labels)
    n0 = len(labels) - n1
    if not n1 or not n0:
        return None
    ordered = sorted(zip(scores, labels))
    concordance, negatives, start = 0.0, 0, 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[start][0]:
            stop += 1
        positives = sum(label for _, label in ordered[start:stop])
        group_negatives = stop - start - positives
        concordance += positives * (negatives + 0.5 * group_negatives)
        negatives += group_negatives
        start = stop
    return concordance / (n1 * n0)


def safe_study_path(study, relative):
    path = (study / relative).resolve()
    if not path.is_relative_to(study):
        raise ValueError("Input ID files must lie within the study directory")
    return path


def read_ids(path):
    rows = read_csv(path)
    if not rows or any("id" not in row for row in rows):
        raise ValueError(f"Expected a nonempty original-ID file: {path}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate original IDs: {path}")
    return ids


def resume_source_amendment(study, key, original, current):
    """Permit only an explicitly recorded, evaluation-only source repair."""
    if original == current:
        return None
    before, after = dict(original), dict(current)
    old_source, new_source = before.pop("source_sha256"), after.pop("source_sha256")
    if before != after:
        raise ValueError("A resume amendment cannot change the plan, data, recipe or environment")
    path = study / "mechanical_amendments.json"
    amendments = json.loads(path.read_text())["amendments"]
    matches = [item for item in amendments
               if item["old_worker_sha256"] == old_source and item["new_worker_sha256"] == new_source
               and key in item["applies_to_fit_keys"]
               and item["allowed_change"] == "evaluation_cpu_transfer_cast_only"]
    if len(matches) != 1:
        raise ValueError("No unique authorized mechanical source amendment for this resume")
    return {"amendment": matches[0], "mechanical_amendments_sha256": sha256(path),
            "original_config_sha256": sha256(study / "fits" / key / "config.json"),
            "original_worker_sha256": old_source, "recovery_worker_sha256": new_source}


def archive_resume_outputs(output):
    """Preserve all prior small evidence before any resumed-process writes."""
    destination = output / "recovery_attempts" / str(time.time_ns())
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.suffix in (".csv", ".json"):
            shutil.copy2(path, destination / path.name)
            hashes[path.name] = sha256(path)
    atomic_json(destination / "preservation_receipt.json", {"preserved_files_sha256": hashes,
        "preserved_at_utc": datetime.now(timezone.utc).isoformat(),
        "large_checkpoint_preserved_in_place": "resume.pt"})
    return destination


def load_fit_inputs(study, key):
    plan_path = study / "plan.json"
    plan = json.loads(plan_path.read_text())
    definitions = [fit for fit in plan["fits"] if fit["key"] == key]
    if len(definitions) != 1:
        raise ValueError("Expected one prescribed fit definition")
    fit = definitions[0]
    train_path = safe_study_path(study, fit["train_ids_file"])
    held_path = safe_study_path(study, fit["held_ids_file"])
    train_ids, held_ids = read_ids(train_path), read_ids(held_path)
    if set(train_ids) & set(held_ids):
        raise ValueError("Training and held original IDs overlap")
    fitting, validation = load_partitions()
    fitting_by_id = {row["id"]: row for row in fitting}
    validation_by_id = {row["id"]: row for row in validation}
    if fit["scheme"] in ("random", "topic"):
        if set(train_ids) | set(held_ids) != set(fitting_by_id):
            raise ValueError("CV train/held IDs must partition the canonical 8,000 originals")
        by_id = fitting_by_id
    elif fit["scheme"] == "validation":
        if set(train_ids) != set(fitting_by_id) or set(held_ids) != set(validation_by_id):
            raise ValueError("Canonical fit must retain the exact 8,000/2,000 boundary")
        by_id = {**fitting_by_id, **validation_by_id}
    else:
        raise ValueError("Only frozen random/topic CV or canonical validation fits are allowed")
    train = [by_id[identifier] for identifier in train_ids]
    held = [by_id[identifier] for identifier in held_ids]
    for records in (train, held):
        if {row["label"] for row in records} != {"0", "1"}:
            raise ValueError("Each fitting/held set must contain both classes")
    provenance = {"plan_sha256": sha256(plan_path), "train_input_ids_sha256": sha256(train_path),
                  "held_input_ids_sha256": sha256(held_path),
                  "canonical_split_manifest_sha256": sha256(ROOT / "data/splits/manifest.json")}
    return plan, fit, train, held, provenance


def tokenize(tokenizer, records, length, torch):
    texts = [row["text"] for row in records]
    started = time.perf_counter()
    full = tokenizer(texts, truncation=False, padding=False, return_attention_mask=False)
    lengths = [len(tokens) for tokens in full["input_ids"]]
    del full
    encoded = tokenizer(texts, truncation=True, max_length=length, padding="max_length", return_tensors="pt")
    encoded = {name: values for name, values in encoded.items() if name in ("input_ids", "attention_mask", "token_type_ids")}
    kept = encoded["attention_mask"].sum(dim=1).tolist()
    metadata = [{"id": row["id"], "label": int(row["label"]), "n_words": len(row["text"].split()),
                 "token_count_full": full_length, "tokens_kept": int(retained), "truncated": full_length > length}
                for row, full_length, retained in zip(records, lengths, kept, strict=True)]
    return encoded, metadata, time.perf_counter() - started


def cpu_tree(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item, torch) for item in value)
    return value


def atomic_torch_save(torch, value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run(args, output):
    if Path(sys.executable).parent != Path("/opt/anaconda3/envs/python_course/bin"):
        raise RuntimeError("Use the user-selected python_course interpreter")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("Start a fresh process with unsupported-op fallback disabled")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import psutil
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    started = time.perf_counter()
    study = args.study_dir.resolve()
    plan, fit, train, held, provenance = load_fit_inputs(study, args.fit_key)
    execution_provenance = json.loads((study / "provenance.json").read_text())
    if execution_provenance["plan_sha256"] != provenance["plan_sha256"]:
        raise ValueError("The scientific plan changed after execution-source freezing")
    for relative, expected in execution_provenance["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen executing source changed: {relative}")
    if execution_provenance["source_sha256"].get("experiments/supervised/train_roberta.py") != sha256(__file__):
        raise ValueError("The worker must be included in frozen executing sources")
    for relative, expected in plan.get("input_files_sha256", {}).items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen reference input changed: {relative}")
    for relative, expected in plan.get("prepared_files_sha256", {}).items():
        if sha256(safe_study_path(study, relative)) != expected:
            raise ValueError(f"Prepared study input changed: {relative}")
    recipe = plan["recipe"]
    required = {"epochs": 2, "microbatch": 4, "gradient_accumulation": 4, "max_length": 512,
                "learning_rate": 2e-5, "weight_decay": 0.01, "max_grad_norm": 1.0,
                "seed": 42, "attention": "sdpa", "dtype": "float32"}
    if any(recipe.get(key) != value for key, value in required.items()):
        raise ValueError("The worker requires the prescribed fixed scientific recipe")
    snapshot = Path(plan["model"]["snapshot"]).resolve()
    if snapshot.name != plan["model"]["revision"] or sha256(snapshot / "model.safetensors") != plan["model"]["weights_sha256"]:
        raise ValueError("Pinned generic model checkpoint differs")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    seed, batch_size = recipe["seed"], recipe["microbatch"]
    accumulation, length = recipe["gradient_accumulation"], recipe["max_length"]
    effective_batch = batch_size * accumulation
    random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.set_num_threads(recipe.get("cpu_threads", 10))
    config = {
        **provenance, "source_sha256": sha256(__file__), "fit": fit, "recipe": recipe,
        "model": plan["model"], "n_train": len(train), "n_held": len(held),
        "training_id_order_sha256": id_digest([row["id"] for row in train]),
        "held_id_order_sha256": id_digest([row["id"] for row in held]),
        "python": sys.executable, "python_version": platform.python_version(),
        "macOS": platform.mac_ver()[0], "packages": {name: version(name) for name in (
            "torch", "transformers", "huggingface-hub", "tokenizers", "numpy", "psutil", "safetensors")},
        "torch_cpu_threads": torch.get_num_threads(), "mps_built": True, "mps_available": True,
        "mps_recommended_max_memory_bytes": int(torch.mps.recommended_max_memory()),
        "fallback_environment_before_torch_import": "0", "scheduler": "constant; no warmup",
        "epoch_shuffle": "Python random.Random(seed42+epoch).shuffle, one-based epoch; independent of torch dropout RNG",
        "partial_update_loss": "Each microbatch mean loss multiplied by microbatch_n / actual_effective_update_n",
        "checkpoint_selection": "Final prescribed epoch2 only; held model inputs/scoring after final training update",
        "test_access": False, "train_from_device_probe_weights": False,
    }
    config_path = output / "config.json"
    recovery_lineage = None
    recovery_archive = None
    if args.resume:
        if not config_path.is_file():
            raise ValueError("Resume requires the original immutable configuration")
        recovery_lineage = resume_source_amendment(study, args.fit_key, json.loads(config_path.read_text()), config)
        if (output / "completion.json").is_file():
            raise ValueError("A completed fit cannot be resumed")
        recovery_archive = archive_resume_outputs(output)
    else:
        atomic_json(config_path, config)
        atomic_csv(output / "train_ids.csv", [{"id": row["id"], "label": row["label"]} for row in train], ["id", "label"])
        atomic_csv(output / "held_ids.csv", [{"id": row["id"], "label": row["label"]} for row in held], ["id", "label"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    train_encoded, train_metadata, train_tokenization_seconds = tokenize(tokenizer, train, length, torch)
    if args.resume:
        previous_metadata = read_csv(output / "train_metadata.csv")
        normalized_metadata = [{key: str(value) for key, value in row.items()} for row in train_metadata]
        if previous_metadata != normalized_metadata:
            raise ValueError("Re-tokenized training metadata differs from the original fit")
    else:
        atomic_csv(output / "train_metadata.csv", train_metadata, list(train_metadata[0]))
    train_encoded["labels"] = torch.tensor([int(row["label"]) for row in train], dtype=torch.long)
    valid_counts = train_encoded["attention_mask"].sum(dim=1).tolist()
    model_load_started = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True, use_safetensors=True, num_labels=2,
        dtype=torch.float32, attn_implementation=recipe["attention"],
    ).to("mps")
    torch.mps.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    if any(p.device.type != "mps" or p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Model parameters must be FP32 on MPS")
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"],
                                 foreach=False, fused=False)
    named = dict(model.named_parameters())
    monitor_names = ("roberta.encoder.layer.0.attention.self.query.weight", "classifier.out_proj.weight")
    initial_parameters = {name: named[name].detach().reshape(-1)[:256].cpu().clone() for name in monitor_names}
    start_epoch, epoch_receipts, steps = 1, [], []
    resume_path = output / "resume.pt"
    if args.resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        if state["config_sha256"] != sha256(config_path):
            raise ValueError("Resume checkpoint belongs to a different frozen configuration")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng_state"])
        torch.mps.set_rng_state(state["mps_rng_state"])
        random.setstate(state["python_rng_state"])
        initial_parameters = state["initial_parameters"]
        epoch_receipts = state["epoch_receipts"]
        start_epoch = int(state["completed_epoch"]) + 1
        original_epoch_receipts = json.loads((output / "epoch_receipts.json").read_text())
        if len(original_epoch_receipts) < len(epoch_receipts):
            raise ValueError("Original completed-epoch receipts are incomplete")
        for saved, recorded in zip(epoch_receipts, original_epoch_receipts, strict=False):
            if any(recorded.get(key) != value for key, value in saved.items()):
                raise ValueError("Checkpoint and original epoch receipts differ")
        epoch_receipts = original_epoch_receipts[:start_epoch - 1]
        previous_steps_path = output / "training_steps.csv"
        if previous_steps_path.is_file():
            prior = read_csv(previous_steps_path)
            steps = [row for row in prior if int(row["epoch"]) < start_epoch]
        if len(steps) != len(epoch_receipts) * math.ceil(len(train) / effective_batch):
            raise ValueError("Resume completed-epoch step records are incomplete")
        atomic_json(output / f"resume_receipt_{time.time_ns()}.json", {
            "completed_epoch": start_epoch - 1, "checkpoint_sha256": sha256(resume_path),
            "config_sha256": sha256(config_path), "resume_at_utc": datetime.now(timezone.utc).isoformat(),
            "current_worker_sha256": sha256(__file__), "source_amendment": recovery_lineage,
            "preserved_evidence_directory": str(recovery_archive.relative_to(output)),
        })
        del state
        gc.collect()
    step_path = output / "training_steps.csv"
    has_training_remaining = start_epoch <= recipe["epochs"]
    if has_training_remaining:
        atomic_csv(step_path, steps, STEP_FIELDS)
    sampler = MemorySampler(torch, psutil, interval=0.5) if has_training_remaining else None
    if sampler is not None:
        sampler.start()
    training_started = time.perf_counter()
    try:
        with step_path.open("a", newline="", encoding="utf-8") as step_stream:
            writer = csv.DictWriter(step_stream, fieldnames=STEP_FIELDS, lineterminator="\n")
            for epoch in range(start_epoch, recipe["epochs"] + 1):
                sampler.phase = f"training_epoch_{epoch}"
                model.train()
                order = epoch_order(len(train), seed, epoch)
                epoch_ids = [train[i]["id"] for i in order]
                epoch_order_path = output / f"epoch_{epoch}_order.csv"
                atomic_csv(epoch_order_path, [{"id": identifier} for identifier in epoch_ids], ["id"])
                epoch_started = time.perf_counter()
                this_epoch_steps = []
                for update_index, (update_indices, microbatches, loss_weights) in enumerate(
                        update_groups(order, batch_size, accumulation), start=1):
                    torch.mps.synchronize()
                    update_started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    micro_losses, micro_sizes = [], []
                    for indices, loss_weight in zip(microbatches, loss_weights, strict=True):
                        index_tensor = torch.tensor(indices, dtype=torch.long)
                        batch = {name: values.index_select(0, index_tensor).to("mps") for name, values in train_encoded.items()}
                        result = model(**batch)
                        micro_n = len(indices)
                        loss = result.loss * loss_weight
                        loss.backward()
                        micro_losses.append(result.loss.detach())
                        micro_sizes.append(micro_n)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["max_grad_norm"],
                                                         error_if_nonfinite=False, foreach=False)
                    optimizer.step()
                    torch.mps.synchronize()
                    seconds = time.perf_counter() - update_started
                    scalar_losses = [float(value.item()) for value in micro_losses]
                    scalar_norm = float(norm.item())
                    finite = all(math.isfinite(value) for value in scalar_losses) and math.isfinite(scalar_norm)
                    if not finite or result.logits.device.type != "mps" or norm.device.type != "mps":
                        raise RuntimeError("Training numeric/device checks failed")
                    row = {
                        "epoch": epoch, "optimizer_update_in_epoch": update_index,
                        "optimizer_update_global": (epoch - 1) * math.ceil(len(train) / effective_batch) + update_index,
                        "documents": len(update_indices), "microbatch_sizes_json": json.dumps(micro_sizes),
                        "original_id_sequence_sha256": id_digest([train[i]["id"] for i in update_indices]),
                        "effective_tokens": sum(valid_counts[i] for i in update_indices), "seconds": seconds,
                        "mean_loss": sum(value * size for value, size in zip(scalar_losses, micro_sizes)) / len(update_indices),
                        "preclip_gradient_norm": scalar_norm, "loss_and_gradient_norm_finite": finite,
                        "logits_on_mps": True, "gradient_norm_on_mps": True,
                    }
                    writer.writerow(row)
                    steps.append(row)
                    this_epoch_steps.append(row)
                    if update_index == 1 or update_index % 25 == 0 or update_index == math.ceil(len(order) / effective_batch):
                        step_stream.flush()
                        sampler.sample()
                        recent = this_epoch_steps[-25:]
                        atomic_json(output / "progress.json", {"fit_key": args.fit_key, "phase": "training", "epoch": epoch,
                            "epochs": recipe["epochs"], "updates_in_epoch": update_index,
                            "expected_updates_per_epoch": math.ceil(len(train) / effective_batch),
                            "last_loss": row["mean_loss"], "recent_mean_update_seconds": sum(r["seconds"] for r in recent) / len(recent),
                            "elapsed_this_process_seconds": time.perf_counter() - started})
                        remaining_updates = recipe["epochs"] * math.ceil(len(train) / effective_batch) - row["optimizer_update_global"]
                        atomic_json(output / "status.json", {"fit_key": args.fit_key, "phase": "training", "epoch": epoch,
                            "update_global": row["optimizer_update_global"], "examples": (epoch - 1) * len(train) + min(update_index * effective_batch, len(train)),
                            "loss": row["mean_loss"], "recent_mean_update_seconds": sum(r["seconds"] for r in recent) / len(recent),
                            "training_only_eta_seconds": remaining_updates * sum(r["seconds"] for r in recent) / len(recent),
                            "elapsed_this_process_seconds": time.perf_counter() - started})
                        print(json.dumps({"fit_key": args.fit_key, "phase": "training", "epoch": epoch,
                            "update": update_index, "total_updates_in_epoch": math.ceil(len(train) / effective_batch),
                            "recent_loss": sum(r["mean_loss"] * r["documents"] for r in recent) / sum(r["documents"] for r in recent),
                            "recent_documents_per_second": sum(r["documents"] for r in recent) / sum(r["seconds"] for r in recent)}), flush=True)
                step_stream.flush()
                receipt = {"epoch": epoch, "originals_seen": sum(row["documents"] for row in this_epoch_steps),
                    "unique_originals_seen": len(set(epoch_ids)), "epoch_permutation_sha256": id_digest(epoch_ids),
                    "epoch_order_file": epoch_order_path.name, "epoch_order_file_sha256": sha256(epoch_order_path),
                    "optimizer_updates": len(this_epoch_steps), "mean_loss": sum(row["mean_loss"] * row["documents"] for row in this_epoch_steps) / len(train),
                    "training_update_seconds": sum(row["seconds"] for row in this_epoch_steps),
                    "epoch_loop_wall_seconds": time.perf_counter() - epoch_started,
                    "held_model_inputs_used": False, "held_metrics_evaluated": False}
                epoch_receipts.append(receipt)
                sampler.phase = "epoch_checkpoint"
                checkpoint_started = time.perf_counter()
                atomic_torch_save(torch, {
                    "config_sha256": sha256(config_path), "completed_epoch": epoch,
                    "model": cpu_tree(model.state_dict(), torch), "optimizer": cpu_tree(optimizer.state_dict(), torch),
                    "cpu_rng_state": torch.get_rng_state(), "mps_rng_state": torch.mps.get_rng_state(),
                    "python_rng_state": random.getstate(), "initial_parameters": initial_parameters,
                    "epoch_receipts": epoch_receipts,
                }, resume_path)
                receipt["epoch_checkpoint_seconds"] = time.perf_counter() - checkpoint_started
                atomic_json(output / "epoch_receipts.json", epoch_receipts)
                print(json.dumps({"fit_key": args.fit_key, "phase": "epoch_complete", **receipt}), flush=True)
    finally:
        if sampler is not None:
            sampler.stop()
            atomic_csv(output / "memory_samples.csv", sampler.rows, list(sampler.rows[0]) if sampler.rows else ["elapsed_seconds", "phase"])
            atomic_json(output / "memory_summary.json", sampler.summary())
    training_process_wall_seconds = time.perf_counter() - training_started
    if len(epoch_receipts) != recipe["epochs"]:
        raise RuntimeError("The prescribed epochs did not complete")
    all_gradients_mps = all(p.grad is None or p.grad.device.type == "mps" for p in model.parameters())
    state_devices = {}
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                key = f"{name}:{value.device.type}:{value.dtype}"
                state_devices[key] = state_devices.get(key, 0) + 1
                if name != "step" and value.device.type != "mps":
                    raise RuntimeError("Optimizer floating moment tensors left MPS")
    deltas = {name: float((named[name].detach().reshape(-1)[:256].cpu() - old).abs().max())
              for name, old in initial_parameters.items()}
    if not all_gradients_mps or not all(value > 0 for value in deltas.values()):
        raise RuntimeError("Parameter update/device verification failed")
    model.eval()
    evaluation_started = time.perf_counter()
    atomic_json(output / "progress.json", {"fit_key": args.fit_key, "phase": "evaluation_after_final_epoch", "completed_epochs": 2})
    # First held model/tokenizer inputs are constructed only here, after epoch2.
    held_encoded, held_metadata, held_tokenization_seconds = tokenize(tokenizer, held, length, torch)
    atomic_csv(output / "held_metadata.csv", held_metadata, list(held_metadata[0]))
    held_conditions = conditions([row["text"] for row in held])
    prediction_rows, condition_metrics = [], {}
    reload_inputs, before_reload = None, None
    for condition, (indices, texts) in held_conditions.items():
        encoded = held_encoded if condition == "clean" else tokenizer(
            texts, truncation=True, max_length=length, padding="max_length", return_tensors="pt")
        encoded = {name: values for name, values in encoded.items() if name in ("input_ids", "attention_mask", "token_type_ids")}
        margins, probabilities = [], []
        condition_started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                batch = {name: values[start:start + batch_size].to("mps") for name, values in encoded.items()}
                logits = model(**batch).logits
                if logits.device.type != "mps":
                    raise RuntimeError("Evaluation logits left MPS")
                host_logits = logits.detach().cpu().to(dtype=torch.float64)
                if not torch.isfinite(host_logits).all().item():
                    raise RuntimeError("Nonfinite evaluation logits")
                margins.extend((host_logits[:, 1] - host_logits[:, 0]).tolist())
                probabilities.extend(host_logits.softmax(-1)[:, 1].tolist())
                if condition == "clean" and start == 0:
                    reload_inputs = {name: values[:min(2, len(indices))].to("mps") for name, values in encoded.items()}
                    before_reload = model(**reload_inputs).logits.detach().cpu().to(dtype=torch.float64)
        labels = [int(held[i]["label"]) for i in indices]
        condition_metrics[condition] = {"n": len(indices), "n_human": labels.count(0), "n_ai": labels.count(1),
            "auroc_logit_margin": auc(labels, margins), "auroc_probability": auc(labels, probabilities),
            "inference_seconds": time.perf_counter() - condition_started,
            "ranking_score": "class1-class0 logits, cast from FP32 to float64 before subtraction"}
        prediction_rows.extend({"fit_key": args.fit_key, "scheme": fit["scheme"], "fold": fit["fold"],
            "condition": condition, "id": held[i]["id"], "label": held[i]["label"],
            "p_ai": format(float(probability), ".17g"), "logit_margin": format(float(margin), ".17g")}
            for i, probability, margin in zip(indices, probabilities, margins, strict=True))
        print(json.dumps({"fit_key": args.fit_key, "phase": "evaluation_complete", "condition": condition,
                          **condition_metrics[condition]}), flush=True)
    atomic_csv(output / "predictions.csv", prediction_rows, PREDICTION_FIELDS)
    if fit["scheme"] == "validation":
        atomic_csv(output / "validation_predictions.csv", [{key: row[key] for key in ("id", "label", "p_ai")}
                   for row in prediction_rows if row["condition"] == "clean"], ["id", "label", "p_ai"])
    evaluation_seconds = time.perf_counter() - evaluation_started
    checkpoint_started = time.perf_counter()
    model_config = model.config
    model_config.save_pretrained(output / "model_config")
    checkpoint = output / "model.pt"
    atomic_torch_save(torch, cpu_tree(model.state_dict(), torch), checkpoint)
    del model, named, optimizer
    gc.collect()
    torch.mps.empty_cache()
    restored = AutoModelForSequenceClassification.from_config(model_config, attn_implementation=recipe["attention"])
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    restored = restored.float().to("mps").eval()
    with torch.inference_mode():
        after_reload = restored(**reload_inputs).logits.detach().cpu().to(dtype=torch.float64)
    reload_error = float((after_reload - before_reload).abs().max())
    if not torch.isfinite(after_reload).all().item() or reload_error > 1e-6:
        raise RuntimeError(f"Saved-model reload logit discrepancy: {reload_error}")
    checkpoint_reload_seconds = time.perf_counter() - checkpoint_started
    metrics = {"status": "complete", "fit_key": args.fit_key, "scheme": fit["scheme"], "fold": fit["fold"],
        "n_train": len(train), "n_held": len(held), "epochs": recipe["epochs"],
        "optimizer_updates": sum(int(row["optimizer_updates"]) for row in epoch_receipts),
        "training_documents_processed": sum(int(row["originals_seen"]) for row in epoch_receipts),
        "training_update_seconds": sum(float(row["training_update_seconds"]) for row in epoch_receipts),
        "training_epoch_loop_seconds": sum(float(row["epoch_loop_wall_seconds"]) for row in epoch_receipts),
        "training_process_wall_seconds": training_process_wall_seconds,
        "training_process_wall_seconds_scope": "Training/checkpoint stage in this process; a resumed evaluation-only process performs zero training updates",
        "training_updates_in_this_process": sum(int(row["optimizer_updates"]) for row in epoch_receipts if int(row["epoch"]) >= start_epoch),
        "completed_epoch_loop_and_checkpoint_seconds": sum(float(row["epoch_loop_wall_seconds"]) + float(row.get("epoch_checkpoint_seconds", 0)) for row in epoch_receipts),
        "recovery_process": bool(args.resume), "recovery_source_amendment": recovery_lineage,
        "train_tokenization_seconds": train_tokenization_seconds,
        "held_tokenization_seconds": held_tokenization_seconds, "model_load_seconds": model_load_seconds,
        "evaluation_seconds": evaluation_seconds, "checkpoint_reload_seconds": checkpoint_reload_seconds,
        "process_elapsed_seconds": time.perf_counter() - started,
        "condition_metrics": condition_metrics, "all_gradients_mps": all_gradients_mps,
        "optimizer_state_devices": state_devices, "monitored_parameter_max_changes": deltas,
        "reload_max_logit_error": reload_error, "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "process_lifetime_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "limitations": ["One seed/recipe; overlapping CV fits are not independent replications",
            "Prior diagnostic folds reused; no untouched hidden-domain estimate",
            "Memory is sampled, not an exact peak; unified-memory measures overlap",
            "CPU host/bookkeeping work expected; inspect native logs in addition to Python warnings",
            "No probability calibration/ensemble fitting or test prediction is performed"],
    }
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "validation_receipt.json", {"forward_backward_optimizer_executed": True,
        "all_numeric_checks_passed": True, "all_gradients_mps": all_gradients_mps,
        "encoder_and_classifier_parameters_updated": True, "completed_epochs_before_held_inference": 2,
        "held_early_stopping_or_epoch_selection": False, "reload_max_logit_error": reload_error,
        "reload_tolerance": 1e-6, "reload_check_training_or_held_examples": "first two held clean inputs, after training only",
        "actual_execution_worker_sha256": sha256(__file__), "recovery_source_amendment": recovery_lineage,
        "files_sha256": {name: sha256(output / name) for name in (
            "config.json", "train_ids.csv", "held_ids.csv", "train_metadata.csv", "held_metadata.csv",
            "training_steps.csv", "epoch_receipts.json", "predictions.csv", "metrics.json")}})
    # Remove only the superseded local restart checkpoint after a complete final
    # checkpoint/reload check; final model.pt remains as the scientific artifact.
    resume_path.unlink(missing_ok=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--fit-key", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume only the last complete epoch with identical frozen inputs")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.fit_key):
        raise ValueError("Unsafe fit key")
    output = args.study_dir.resolve() / "fits" / args.fit_key
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError("No fit exists to resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
    caught = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            metrics = run(args, output)
        except Exception as error:
            failure = {"status": "failed", "error": repr(error),
                "traceback": traceback.format_exc(), "source_sha256": sha256(__file__),
                "failed_at_utc": datetime.now(timezone.utc).isoformat()}
            atomic_json(output / f"failure_{time.time_ns()}.json", failure)
            atomic_json(output / "failure.json", failure)
            raise
        finally:
            notes = [{"category": note.category.__name__, "message": str(note.message)} for note in caught]
            atomic_json(output / "warnings.json", notes)
    fallback = [note for note in notes if any(term in note["message"].lower() for term in (
        "fallback", "fall back", "falling back", "not currently supported on the mps"))]
    atomic_json(output / "fallback_receipt.json", {"environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "captured_fallback_warning_count": len(fallback), "captured_fallback_warnings": fallback,
        "claim": "No unsupported-op fallback observed only if native logs also have no matches; ordinary CPU work is expected"})
    atomic_json(output / "completion.json", {"status": "complete", "fit_key": args.fit_key,
        "epochs": metrics["epochs"], "optimizer_updates": metrics["optimizer_updates"],
        "metrics_sha256": sha256(output / "metrics.json"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    atomic_json(output / "status.json", {"fit_key": args.fit_key, "phase": "complete", "epochs": metrics["epochs"],
        "optimizer_updates": metrics["optimizer_updates"], "metrics_sha256": sha256(output / "metrics.json")})
    print(json.dumps({"fit_key": args.fit_key, "status": "complete", "metrics": str(output / "metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
