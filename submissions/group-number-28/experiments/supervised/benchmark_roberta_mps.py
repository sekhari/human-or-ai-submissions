"""Measure a complete RoBERTa classification training update on Apple MPS.

Run each configuration in a fresh python_course process. Uses only a small,
deterministic selection of canonical fitting originals, never held/test scores.
The checkpoint is a device-test artifact, not a competition candidate.
"""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import threading
import time
import traceback
import warnings


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    import csv
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class MemorySampler:
    """Observed allocator/RSS samples; this is not an exact peak profiler."""

    def __init__(self, torch, psutil, interval=0.1):
        self.torch, self.psutil, self.interval = torch, psutil, interval
        self.process = psutil.Process()
        self.started = time.perf_counter()
        self.phase = "initial"
        self.rows, self.errors = [], []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        try:
            self.rows.append({
                "elapsed_seconds": time.perf_counter() - self.started,
                "phase": self.phase,
                "mps_current_allocated_bytes": int(self.torch.mps.current_allocated_memory()),
                "mps_driver_allocated_bytes": int(self.torch.mps.driver_allocated_memory()),
                "process_rss_bytes": int(self.process.memory_info().rss),
                "system_available_bytes": int(self.psutil.virtual_memory().available),
                "system_swap_used_bytes": int(self.psutil.swap_memory().used),
            })
        except Exception as error:
            self.errors.append(repr(error))

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.sample()

    def summary(self):
        names = ("mps_current_allocated_bytes", "mps_driver_allocated_bytes", "process_rss_bytes",
                 "system_available_bytes", "system_swap_used_bytes")
        result = {"sample_interval_seconds": self.interval, "n_samples": len(self.rows),
                  "sample_errors": self.errors, "peak_kind": "Observed samples, not guaranteed exact peaks",
                  "unified_memory_note": "RSS and MPS driver allocation overlap; do not sum them."}
        for phase in ("all_training", "timed"):
            rows = self.rows if phase == "all_training" else [r for r in self.rows if r["phase"] == "timed"]
            result[phase] = {name: {"minimum": min(r[name] for r in rows),
                                     "maximum": max(r[name] for r in rows)} for name in names} if rows else None
        return result


def run(args, output):
    # This is deliberately before importing torch: unsupported-op fallback must
    # fail visibly rather than silently turn the benchmark into CPU execution.
    existing = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if existing not in (None, "0"):
        raise RuntimeError("Start with PYTORCH_ENABLE_MPS_FALLBACK=0, not enabled fallback")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import psutil
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from human_ai.data import load_partitions, SPLIT_DIR

    snapshot = args.snapshot.resolve()
    weights = snapshot / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError("The pinned local safetensors checkpoint is required")
    if args.batch_size < 1 or args.grad_accum < 1 or args.steps != 20 or args.warmup != 5:
        raise ValueError("Positive batches/accumulation and exactly5 warmup+20 timed updates required")
    if not 4 <= args.sequence_length <= 512:
        raise ValueError("Sequence length must lie in [4,512]")
    random.seed(42)
    torch.manual_seed(42)
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    plan = {
        "benchmark_only": True, "model": "FacebookAI/roberta-base", "snapshot": str(snapshot),
        "snapshot_revision": snapshot.name, "weights_sha256": digest(weights),
        "weights_bytes": weights.stat().st_size, "source_sha256": digest(__file__),
        "batch_size": args.batch_size, "sequence_length": args.sequence_length,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "warmup_optimizer_updates": args.warmup, "timed_optimizer_updates": args.steps,
        "seed": 42, "dtype": "float32", "attention_implementation": args.attention,
        "optimizer": {"name": "AdamW", "learning_rate": 2e-5, "weight_decay": 0.01,
                      "foreach": False, "fused": False},
        "gradient_clip_norm": 1.0, "fallback_environment_before_torch_import": "0",
        "timing": "Synchronize before and after each complete optimizer update; phase times are CPU enqueue durations, not GPU phase timings. Numeric scalar checks occur after timed synchronization.",
        "inputs": "Pretokenized real canonical fitting originals; CPU batch slicing and device transfer are inside update timing, tokenizer execution outside.",
        "packages": {name: version(name) for name in ("torch", "transformers", "huggingface-hub", "tokenizers", "numpy", "psutil", "safetensors")},
        "python": sys.executable, "python_version": platform.python_version(), "macOS": platform.mac_ver()[0],
        "torch_cpu_threads": torch.get_num_threads(), "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "validation_or_test_inputs_used": False, "task_auroc_evaluated": False,
    }
    save_json(output / "plan.json", plan)
    if not plan["mps_built"] or not plan["mps_available"]:
        raise RuntimeError("MPS backend is unavailable in this process")
    plan["mps_recommended_max_memory_bytes"] = int(torch.mps.recommended_max_memory())
    fitting, _ = load_partitions()
    order = list(range(len(fitting)))
    random.Random(42).shuffle(order)
    n = min(len(fitting), max(128, args.batch_size * args.grad_accum * (args.warmup + args.steps)))
    records = [fitting[i] for i in order[:n]]
    write_csv(output / "benchmark_original_ids.csv", [{"id": row["id"], "label": row["label"]} for row in records])
    plan.update(n_unique_benchmark_originals=n,
                canonical_split_manifest_sha256=digest(SPLIT_DIR / "manifest.json"),
                selected_ids_sha256=digest(output / "benchmark_original_ids.csv"))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    token_started = time.perf_counter()
    encoded = tokenizer([r["text"] for r in records], truncation=True, max_length=args.sequence_length,
                        padding="max_length", return_tensors="pt")
    tokenization_seconds = time.perf_counter() - token_started
    encoded = {name: values for name, values in encoded.items()
               if name in ("input_ids", "attention_mask", "token_type_ids")}
    encoded["labels"] = torch.tensor([int(r["label"]) for r in records], dtype=torch.long)
    valid_counts = encoded["attention_mask"].sum(dim=1).tolist()
    plan.update(tokenization_seconds=tokenization_seconds,
                selected_effective_tokens=int(sum(valid_counts)),
                effective_tokens_definition="attention_mask sum, including special tokens; excludes padding",
                selected_padded_tokens=n * args.sequence_length)
    save_json(output / "plan.json", plan)

    load_started = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True, use_safetensors=True, num_labels=2,
        dtype=torch.float32, attn_implementation=args.attention,
    ).to("mps")
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_started
    if any(p.device.type != "mps" or p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("All model parameters must be float32 on MPS")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, foreach=False, fused=False)
    named = dict(model.named_parameters())
    encoder_name = next(name for name in named if "encoder.layer.0" in name and name.endswith("query.weight"))
    head_name = next(name for name in named if name.endswith("classifier.out_proj.weight"))
    monitored = {name: named[name].detach().reshape(-1)[:256].cpu().clone() for name in (encoder_name, head_name)}
    eval_inputs = {k: v[:min(2, n)].to("mps") for k, v in encoded.items() if k != "labels"}
    model.eval()
    with torch.no_grad():
        initial_probabilities = model(**eval_inputs).logits.softmax(-1).cpu()
    if not torch.isfinite(initial_probabilities).all().item():
        raise RuntimeError("Initial evaluation probabilities are nonfinite")
    initial_probabilities = initial_probabilities.tolist()
    model.train()
    sampler = MemorySampler(torch, psutil)
    sampler.start()
    update_rows, cursor = [], 0
    process = psutil.Process()
    timed_cpu_start = None
    timed_external_start = None

    def update(number, phase):
        nonlocal cursor
        sampler.phase = phase
        torch.mps.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        zero_end = time.perf_counter()
        host_transfer, host_forward, host_backward = 0.0, 0.0, 0.0
        losses, tokens, processed_ids, token_counts = [], 0, [], []
        for _ in range(args.grad_accum):
            t = time.perf_counter()
            indices = [(cursor + j) % n for j in range(args.batch_size)]
            cursor += args.batch_size
            index = torch.tensor(indices, dtype=torch.long)
            batch = {k: v.index_select(0, index).to("mps") for k, v in encoded.items()}
            tokens += sum(valid_counts[i] for i in indices)
            processed_ids.extend(records[i]["id"] for i in indices)
            token_counts.extend(valid_counts[i] for i in indices)
            host_transfer += time.perf_counter() - t
            t = time.perf_counter()
            result = model(**batch)
            loss = result.loss / args.grad_accum
            losses.append(result.loss.detach())
            host_forward += time.perf_counter() - t
            t = time.perf_counter()
            loss.backward()
            host_backward += time.perf_counter() - t
        t = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False, foreach=False)
        host_clip = time.perf_counter() - t
        t = time.perf_counter()
        optimizer.step()
        host_optimizer = time.perf_counter() - t
        torch.mps.synchronize()
        seconds = time.perf_counter() - started
        # These scalar transfers/checks are deliberately outside the update timer.
        scalar_losses = [float(value.item()) for value in losses]
        scalar_norm = float(norm.item())
        finite = all(math.isfinite(value) for value in scalar_losses) and math.isfinite(scalar_norm)
        if not finite:
            raise RuntimeError("Nonfinite training loss or gradient norm")
        if norm.device.type != "mps" or result.logits.device.type != "mps":
            raise RuntimeError("Training compute unexpectedly left MPS")
        sampler.sample()
        row = {"phase": phase, "optimizer_update": number, "seconds": seconds,
               "documents": args.batch_size * args.grad_accum, "effective_tokens": tokens,
               "padded_tokens": args.batch_size * args.grad_accum * args.sequence_length,
               "mean_loss": statistics.mean(scalar_losses), "preclip_gradient_norm": scalar_norm,
               "loss_and_gradient_norm_finite": finite,
               "original_id_sequence_sha256": hashlib.sha256("\n".join(processed_ids).encode("utf-8")).hexdigest(),
               "effective_token_counts_json": json.dumps(token_counts),
               "host_zero_grad_enqueue_seconds": zero_end - started,
               "host_transfer_enqueue_seconds": host_transfer,
               "host_forward_enqueue_seconds": host_forward,
               "host_backward_enqueue_seconds": host_backward,
               "host_clip_enqueue_seconds": host_clip,
               "host_optimizer_enqueue_seconds": host_optimizer}
        print(json.dumps({"phase": phase, "update": number, "seconds": round(seconds, 4),
                          "loss": round(row["mean_loss"], 6), "grad_norm": round(scalar_norm, 4)}), flush=True)
        return row

    try:
        for number in range(1, args.warmup + 1):
            update_rows.append(update(number, "warmup"))
        torch.mps.synchronize()
        timed_cpu_start = process.cpu_times()
        timed_external_start = time.perf_counter()
        for number in range(1, args.steps + 1):
            update_rows.append(update(number, "timed"))
        timed_external_seconds = time.perf_counter() - timed_external_start
        timed_cpu_end = process.cpu_times()
    finally:
        sampler.stop()
        if update_rows:
            write_csv(output / "steps.csv", update_rows)
        if sampler.rows:
            write_csv(output / "memory_samples.csv", sampler.rows)
        save_json(output / "memory_summary.json", sampler.summary())

    gradients_mps = all(p.grad is None or p.grad.device.type == "mps" for p in model.parameters())
    state_devices, state_bytes = {}, 0
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                key = f"{name}:{value.device.type}:{value.dtype}"
                state_devices[key] = state_devices.get(key, 0) + 1
                state_bytes += value.numel() * value.element_size()
                if name != "step" and value.device.type != "mps":
                    raise RuntimeError("Floating optimizer moments must remain on MPS")
    deltas = {name: float((named[name].detach().reshape(-1)[:256].cpu() - old).abs().max())
              for name, old in monitored.items()}
    if not gradients_mps or not all(value > 0 for value in deltas.values()):
        raise RuntimeError("Encoder/classifier update verification failed")
    model.eval()
    with torch.no_grad():
        final_probabilities = model(**eval_inputs).logits.softmax(-1).cpu()
    if not torch.isfinite(final_probabilities).all().item():
        raise RuntimeError("Final evaluation probabilities are nonfinite")
    model_config = model.config
    model_config.save_pretrained(output / "reload_config")
    checkpoint = output / "model.pt"
    cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(cpu_state, checkpoint)
    parameter_count = sum(p.numel() for p in model.parameters())
    del model, optimizer, named
    gc.collect()
    torch.mps.empty_cache()
    restored = AutoModelForSequenceClassification.from_config(model_config, attn_implementation=args.attention)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    del cpu_state
    restored = restored.float().to("mps").eval()
    with torch.no_grad():
        reloaded_probabilities = restored(**eval_inputs).logits.softmax(-1).cpu()
    if not torch.isfinite(reloaded_probabilities).all().item():
        raise RuntimeError("Reloaded evaluation probabilities are nonfinite")
    reload_error = float((final_probabilities - reloaded_probabilities).abs().max())
    if reload_error > 1e-6:
        raise RuntimeError(f"Saved-model reload error exceeds tolerance: {reload_error}")
    timed = [row for row in update_rows if row["phase"] == "timed"]
    seconds = sum(row["seconds"] for row in timed)
    docs = sum(row["documents"] for row in timed)
    metrics = {
        "status": "completed", "benchmark_only": True, "parameter_count": parameter_count,
        "model_load_and_to_mps_seconds": load_seconds,
        "timed_optimizer_updates": len(timed), "timed_microsteps": len(timed) * args.grad_accum,
        "timed_update_seconds_sum": seconds, "timed_loop_wall_seconds": timed_external_seconds,
        "mean_update_seconds": statistics.mean(row["seconds"] for row in timed),
        "median_update_seconds": statistics.median(row["seconds"] for row in timed),
        "minimum_update_seconds": min(row["seconds"] for row in timed),
        "maximum_update_seconds": max(row["seconds"] for row in timed),
        "documents_processed": docs, "documents_per_second": docs / seconds,
        "effective_tokens_per_second": sum(row["effective_tokens"] for row in timed) / seconds,
        "padded_tokens_per_second": sum(row["padded_tokens"] for row in timed) / seconds,
        "end_to_end_loop_documents_per_second": docs / timed_external_seconds,
        "timed_process_cpu_seconds": (timed_cpu_end.user + timed_cpu_end.system - timed_cpu_start.user - timed_cpu_start.system),
        "memory": sampler.summary(), "optimizer_tensor_state_bytes": state_bytes,
        "optimizer_state_devices": state_devices, "monitored_parameter_max_updates": deltas,
        "all_model_gradients_mps": gradients_mps,
        "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_sha256": digest(checkpoint),
        "reload_max_probability_error": reload_error,
        "process_lifetime_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
        "limitations": [
            "Observed sampled memory is not an exact peak; allocator/RSS measures overlap in unified memory.",
            "Checkpoint I/O/reload and tokenization are excluded from update timing.",
            "Per-phase times are host enqueue durations; GPU execution is asynchronous.",
            "Diagnostics/logging/memory sampling add overhead; this is not a full-epoch benchmark.",
            "CPU optimizer step scalars and host input/bookkeeping work are expected and do not imply unsupported-op fallback.",
            "No validation/test score or competition training result is produced.",
        ],
    }
    save_json(output / "metrics.json", metrics)
    save_json(output / "validation_receipt.json", {
        "forward_backward_optimizer_executed": True, "all_numeric_checks_passed": True,
        "encoder_and_classifier_parameters_updated": True, "gradients_on_mps": gradients_mps,
        "initial_eval_probabilities": initial_probabilities,
        "final_eval_probabilities": final_probabilities.tolist(),
        "reloaded_eval_probabilities": reloaded_probabilities.tolist(),
        "reload_max_error": reload_error, "reload_tolerance": 1e-6,
        "fallback_policy": "Unsupported-op fallback disabled before torch import; success is evidence for this exercised path, not proof that all CPU work is absent.",
        "files_sha256": {p.name: digest(p) for p in (output / "plan.json", output / "steps.csv", output / "memory_samples.csv", output / "metrics.json")},
    })
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--cpu-threads", type=int, default=None)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            metrics = run(args, output)
        except Exception as error:
            save_json(output / "failure.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
                                                "source_sha256": digest(__file__)})
            raise
        finally:
            notes = [{"category": note.category.__name__, "message": str(note.message)} for note in caught]
            save_json(output / "warnings.json", notes)
    fallback = [note for note in notes if any(word in note["message"].lower() for word in ("fallback", "fall back", "falling back", "not currently supported on the mps"))]
    save_json(output / "fallback_receipt.json", {"environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
              "captured_fallback_warning_count": len(fallback), "captured_fallback_warnings": fallback,
              "unsupported_op_fallback_enabled": False,
              "stderr_must_also_be_reviewed": True,
              "claim": "Captured Python warnings and native stderr must both be reviewed. Unsupported-op fallback was disabled; CPU host/optimizer bookkeeping is not asserted absent."})
    print(json.dumps({"status": "completed", "output": str(output),
                      "documents_per_second": metrics["documents_per_second"],
                      "mean_update_seconds": metrics["mean_update_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
