"""Refit the selected raw RoBERTa recipe on all supplied labels and export.

Uses python_course, never warms from a development model, and preserves raw
whitespace. There is no held-out score after fitting all 10,000 labels.
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback
import warnings

from human_ai.data import ROOT, load_partitions, raw_path, read_csv, sha256
from human_ai.check_predictions import check
from .benchmark_roberta_mps import MemorySampler
from .train_roberta import (atomic_csv, atomic_json, atomic_torch_save,
                           cpu_tree, epoch_order, id_digest, tokenize, update_groups)

STEP_FIELDS = ["epoch", "update_in_epoch", "update_global", "documents",
               "original_id_sequence_sha256", "microbatch_sizes_json", "loss",
               "gradient_norm", "seconds", "numeric_finite", "logits_on_mps"]


def prepare(source, run, output):
    if run.exists() or output.exists() or output.with_suffix(".metadata.json").exists():
        raise FileExistsError("Preserve existing artifacts; choose a fresh run/output")
    parent = json.loads((source / "plan.json").read_text())
    audit = json.loads((source / "FINAL_AUDIT.json").read_text())
    if audit["status"] != "passed" or audit["audit_scope"] != "all_prescribed_fits":
        raise ValueError("The selected development study must pass its complete audit")
    fitting, validation = load_partitions()
    records = fitting + validation
    if len(records) != 10000 or len({r["id"] for r in records}) != 10000:
        raise ValueError("Use all exact supplied labeled originals")
    recipe = parent["recipe"]
    required = {"epochs": 2, "microbatch": 4, "gradient_accumulation": 4,
                "max_length": 512, "learning_rate": 2e-5, "weight_decay": .01,
                "max_grad_norm": 1., "seed": 42, "attention": "sdpa", "dtype": "float32"}
    if any(recipe[k] != v for k, v in required.items()):
        raise ValueError("The selected recipe differs from the evaluated one")
    run.mkdir(parents=True)
    sources = ["experiments/supervised/export_roberta.py", "experiments/supervised/train_roberta.py",
               "experiments/supervised/benchmark_roberta_mps.py", "human_ai/data.py",
               "human_ai/check_predictions.py"]
    hashes = {p: sha256(ROOT / p) for p in sources}
    for name in sources:
        dest = run / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    atomic_csv(run / "train_ids.csv", [{"id": r["id"], "label": r["label"]} for r in records], ["id", "label"])
    plan = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "User-requested local submission file from the strongest evaluated clean-ranking candidate",
            "source_study": str(source), "source_plan_sha256": sha256(source / "plan.json"),
            "source_audit_sha256": sha256(source / "FINAL_AUDIT.json"),
            "source_metrics_sha256": sha256(source / "analysis/aggregate_metrics.json"),
            "model": parent["model"], "recipe": recipe, "input_preprocessing": "Raw text; preserve original whitespace",
            "training_n": 10000, "training_order": "Canonical 8000 followed by canonical 2000, then the same seeded epoch shuffle",
            "training_ids_sha256": sha256(run / "train_ids.csv"),
            "train_source_sha256": sha256(raw_path("train.csv")),
            "expected_updates_per_epoch": 625, "expected_total_updates": 1250,
            "selection_basis": "Highest clean random/topic and canonical development AUROCs among evaluated candidates; formatting sensitivity remains a known tradeoff",
            "held_out_score": None, "leaderboard_score": None,
            "warm_start_from_development_checkpoint": False, "new_search": False,
            "test_fitting": False, "automatic_upload": False, "output": str(output),
            "source_sha256": hashes}
    atomic_json(run / "plan.json", plan)
    return plan, records


def predict(torch, model, encoded, batch_size=4):
    probabilities, margins = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded["input_ids"]), batch_size):
            batch = {k: v[start:start + batch_size].to("mps") for k, v in encoded.items()}
            logits = model(**batch).logits
            if logits.device.type != "mps":
                raise RuntimeError("Prediction logits left MPS")
            host = logits.detach().cpu().to(dtype=torch.float64)
            if not torch.isfinite(host).all().item():
                raise RuntimeError("Nonfinite test logits")
            probabilities.extend(host.softmax(-1)[:, 1].tolist())
            margins.extend((host[:, 1] - host[:, 0]).tolist())
    return probabilities, margins


def run(args):
    if Path(sys.executable).parent != Path("/opt/anaconda3/envs/python_course/bin"):
        raise RuntimeError("Use the selected python_course interpreter")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("Disable generic MPS fallback in a fresh process")
    os.environ.update(PYTORCH_ENABLE_MPS_FALLBACK="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    source, output = args.source_study.resolve(), args.output.resolve()
    directory = ROOT / "results/supervised" / args.run_id
    if args.resume:
        plan = json.loads((directory / "plan.json").read_text())
        fitting, validation = load_partitions()
        records = fitting + validation
        if plan["source_study"] != str(source) or plan["output"] != str(output) or (directory / "completion.json").exists():
            raise ValueError("Invalid or already complete resume")
    else:
        plan, records = prepare(source, directory, output)
    for name, digest in plan["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen executing source changed: {name}")
    if sha256(raw_path("train.csv")) != plan["train_source_sha256"] or sha256(directory / "train_ids.csv") != plan["training_ids_sha256"]:
        raise ValueError("Training inputs changed")
    import psutil
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    started = time.perf_counter()
    recipe = plan["recipe"]
    random.seed(42)
    torch.manual_seed(42)
    torch.mps.manual_seed(42)
    torch.set_num_threads(recipe["cpu_threads"])
    snapshot = Path(plan["model"]["snapshot"])
    if sha256(snapshot / "model.safetensors") != plan["model"]["weights_sha256"]:
        raise ValueError("Generic checkpoint changed")
    config = {"plan_sha256": sha256(directory / "plan.json"), "python": sys.executable,
              "python_version": platform.python_version(), "macOS": platform.mac_ver()[0],
              "packages": {name: version(name) for name in ("torch", "transformers", "huggingface-hub", "tokenizers", "numpy", "psutil", "safetensors")},
              "n_train": len(records), "class_counts": {str(i): sum(int(r["label"]) == i for r in records) for i in (0, 1)},
              "training_id_order_sha256": id_digest([r["id"] for r in records]),
              "held_out_evaluation": None, "test_used_for_fitting": False}
    if args.resume:
        if json.loads((directory / "config.json").read_text()) != config:
            raise ValueError("Resume configuration/environment differs")
    else:
        atomic_json(directory / "config.json", config)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoded, metadata, token_seconds = tokenize(tokenizer, records, 512, torch)
    atomic_csv(directory / "train_metadata.csv", metadata, list(metadata[0]))
    encoded["labels"] = torch.tensor([int(r["label"]) for r in records], dtype=torch.long)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True,
        use_safetensors=True, num_labels=2, dtype=torch.float32, attn_implementation="sdpa").to("mps")
    if any(p.dtype != torch.float32 or p.device.type != "mps" for p in model.parameters()):
        raise RuntimeError("Model must be FP32 on MPS")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01, foreach=False, fused=False)
    names = dict(model.named_parameters())
    monitors = ["roberta.encoder.layer.0.attention.self.query.weight", "classifier.out_proj.weight"]
    before = {name: names[name].detach().reshape(-1)[:256].cpu().clone() for name in monitors}
    start_epoch, epochs = 1, []
    resume_path = directory / "resume.pt"
    if args.resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        if state["config_sha256"] != sha256(directory / "config.json"):
            raise ValueError("Checkpoint configuration differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.mps.set_rng_state(state["mps_rng"])
        before, epochs = state["before"], state["epochs"]
        start_epoch = state["completed_epoch"] + 1
        del state
        gc.collect()
        old = directory / f"recovery_{time.time_ns()}"
        old.mkdir()
        for p in directory.glob("*.csv"):
            shutil.copy2(p, old / p.name)
        prior = [r for r in read_csv(directory / "training_steps.csv") if int(r["epoch"]) < start_epoch]
        atomic_csv(directory / "training_steps.csv", prior, STEP_FIELDS)
    else:
        atomic_csv(directory / "training_steps.csv", [], STEP_FIELDS)
    sampler = MemorySampler(torch, psutil, interval=.5)
    sampler.start()
    training_start = time.perf_counter()
    try:
        with (directory / "training_steps.csv").open("a", newline="") as stream:
            writer = csv.DictWriter(stream, STEP_FIELDS, lineterminator="\n")
            for epoch in range(start_epoch, 3):
                sampler.phase = f"training_epoch_{epoch}"
                model.train()
                order = epoch_order(10000, 42, epoch)
                ids = [records[i]["id"] for i in order]
                atomic_csv(directory / f"epoch_{epoch}_order.csv", [{"id": i} for i in ids], ["id"])
                epoch_started = time.perf_counter()
                rows = []
                for update, (indices, micros, weights) in enumerate(update_groups(order, 4, 4), start=1):
                    torch.mps.synchronize()
                    clock = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for micro, weight in zip(micros, weights, strict=True):
                        index = torch.tensor(micro, dtype=torch.long)
                        batch = {k: v.index_select(0, index).to("mps") for k, v in encoded.items()}
                        result = model(**batch)
                        (result.loss * weight).backward()
                        losses.append(result.loss.detach())
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=False, error_if_nonfinite=True)
                    optimizer.step()
                    torch.mps.synchronize()
                    seconds = time.perf_counter() - clock
                    loss = sum(float(v.item()) * w for v, w in zip(losses, weights))
                    gradient = float(norm.item())
                    if not math.isfinite(loss) or not math.isfinite(gradient) or result.logits.device.type != "mps":
                        raise RuntimeError("Numeric/device check failed")
                    row = {"epoch": epoch, "update_in_epoch": update, "update_global": (epoch-1)*625+update,
                           "documents": len(indices), "original_id_sequence_sha256": id_digest([records[i]["id"] for i in indices]),
                           "microbatch_sizes_json": json.dumps([len(m) for m in micros]), "loss": loss,
                           "gradient_norm": gradient, "seconds": seconds, "numeric_finite": True, "logits_on_mps": True}
                    writer.writerow(row)
                    rows.append(row)
                    if update == 1 or update % 25 == 0:
                        stream.flush()
                        recent = rows[-25:]
                        progress = {"epoch": epoch, "update": update, "updates_per_epoch": 625,
                                    "recent_loss": sum(r["loss"] for r in recent)/len(recent),
                                    "documents_per_second": sum(r["documents"] for r in recent)/sum(r["seconds"] for r in recent)}
                        atomic_json(directory / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
                stream.flush()
                entry = {"epoch": epoch, "originals": sum(r["documents"] for r in rows),
                         "unique_originals": len(set(ids)), "order_sha256": id_digest(ids),
                         "updates": len(rows), "loop_seconds": time.perf_counter()-epoch_started,
                         "mean_loss": sum(r["loss"]*r["documents"] for r in rows)/10000,
                         "test_used_for_fitting": False}
                epochs.append(entry)
                sampler.phase = "epoch_checkpoint"
                atomic_torch_save(torch, {"model": cpu_tree(model.state_dict(), torch),
                    "optimizer": cpu_tree(optimizer.state_dict(), torch), "completed_epoch": epoch,
                    "config_sha256": sha256(directory / "config.json"), "epochs": epochs, "before": before,
                    "cpu_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state()}, resume_path)
                atomic_json(directory / "epochs.json", epochs)
                print(json.dumps({"phase": "epoch_complete", **entry}), flush=True)
    finally:
        sampler.stop()
        atomic_csv(directory / "memory_samples.csv", sampler.rows, list(sampler.rows[0]))
        atomic_json(directory / "memory_summary.json", sampler.summary())
    train_seconds = time.perf_counter()-training_start
    if len(epochs) != 2 or sum(e["updates"] for e in epochs) != 1250:
        raise RuntimeError("Incomplete full-label training")
    changes = {name: float((names[name].detach().reshape(-1)[:256].cpu()-before[name]).abs().max()) for name in monitors}
    if not all(v > 0 for v in changes.values()) or any(p.grad is not None and p.grad.device.type != "mps" for p in model.parameters()):
        raise RuntimeError("Full-model update check failed")
    devices = {}
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                label = f"{key}:{value.device.type}"
                devices[label] = devices.get(label, 0)+1
                if key != "step" and value.device.type != "mps":
                    raise RuntimeError("Floating optimizer moments left MPS")
    atomic_json(directory / "progress.json", {"phase": "test_prediction_after_final_epoch", "completed_epochs": 2})
    test = read_csv(raw_path("test.csv"))
    test_ids = [r["id"] for r in test]
    if len(test_ids) != 3000 or len(set(test_ids)) != 3000 or set(test_ids) & {r["id"] for r in records}:
        raise ValueError("Unexpected test ID coverage or overlap")
    test_encoded = tokenizer([r["text"] for r in test], truncation=True, max_length=512,
                             padding="max_length", return_tensors="pt")
    test_encoded = {k: v for k, v in test_encoded.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    prediction_start = time.perf_counter()
    probabilities, margins = predict(torch, model, test_encoded)
    model.config.save_pretrained(directory / "model_config")
    checkpoint = directory / "model.pt"
    atomic_torch_save(torch, cpu_tree(model.state_dict(), torch), checkpoint)
    model_config = model.config
    del model, optimizer, names
    gc.collect()
    torch.mps.empty_cache()
    restored = AutoModelForSequenceClassification.from_config(model_config, attn_implementation="sdpa")
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    restored = restored.float().to("mps").eval()
    reloaded_probabilities, reloaded_margins = predict(torch, restored, test_encoded)
    probability_error = max(abs(a-b) for a,b in zip(probabilities, reloaded_probabilities, strict=True))
    margin_error = max(abs(a-b) for a,b in zip(margins, reloaded_margins, strict=True))
    if max(probability_error, margin_error) > 1e-6 or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise RuntimeError("Reload/probability validation failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(output, [{"id": i, "label": format(p, ".17g")} for i,p in zip(test_ids, probabilities, strict=True)], ["id", "label"])
    atomic_csv(directory / "test_predictions.csv", [{"id": i, "p_ai": format(p, ".17g"), "logit_margin": format(m, ".17g")}
               for i,p,m in zip(test_ids, probabilities, margins, strict=True)], ["id", "p_ai", "logit_margin"])
    n = check(output, "submission")
    metrics = {"status": "complete", "n_train": 10000, "n_test": n, "epochs": 2, "optimizer_updates": 1250,
               "held_out_evaluation": None, "training_process_seconds": train_seconds,
               "prediction_save_reload_seconds": time.perf_counter()-prediction_start,
               "process_elapsed_seconds": time.perf_counter()-started, "train_tokenization_seconds": token_seconds,
               "parameter_max_changes": changes, "optimizer_state_devices": devices,
               "reload_probability_max_error_all_test": probability_error, "reload_margin_max_error_all_test": margin_error,
               "checkpoint_sha256": sha256(checkpoint), "csv_sha256": sha256(output),
               "test_source_sha256": sha256(raw_path("test.csv")), "raw_whitespace_preserved": True,
               "test_used_for_fitting": False, "class1_is_ai": True}
    atomic_json(directory / "metrics.json", metrics)
    atomic_json(output.with_suffix(".metadata.json"), {"run_id": args.run_id, "source_study": str(source),
        "n_train": 10000, "n_test": n, "full_refit": True, "held_out_score": None, "public_score": None,
        "csv_sha256": sha256(output), "checkpoint_sha256": sha256(checkpoint),
        "reload_probability_max_error": probability_error, "raw_whitespace_preserved": True,
        "uploaded_to_kaggle": False, "selection_basis": plan["selection_basis"],
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    return directory, output, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("Unsafe run ID")
    directory = ROOT / "results/supervised" / args.run_id
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            directory, output, metrics = run(args)
        except Exception as error:
            if directory.exists():
                atomic_json(directory / f"failure_{time.time_ns()}.json", {"error": repr(error), "traceback": traceback.format_exc()})
            raise
        finally:
            if directory.exists():
                notes = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
                atomic_json(directory / "warnings.json", notes)
    fallback = [w for w in notes if any(x in w["message"].lower() for x in ("fallback", "fall back", "falling back"))]
    atomic_json(directory / "fallback_receipt.json", {"generic_fallback_environment": "0", "python_fallback_warnings": fallback,
                "native_logs_also_require_review": True})
    atomic_json(directory / "completion.json", {"status": "complete", "csv": str(output),
                "metrics_sha256": sha256(directory / "metrics.json"), "csv_sha256": sha256(output),
                "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    (directory / "resume.pt").unlink(missing_ok=True)
    print(json.dumps({"status": "complete", "submission": str(output), "n_test": metrics["n_test"]}), flush=True)


if __name__ == "__main__":
    main()
