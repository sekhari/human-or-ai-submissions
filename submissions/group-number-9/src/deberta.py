"""DeBERTa-v3 sequence classification with explicit, reproducible training stages."""
import contextlib
import gc
import hashlib
import json
import math
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_ID = "microsoft/deberta-v3-base"
REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
ROOT = Path(__file__).resolve().parents[1]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("The Mac GPU is unavailable to this process; run with GPU access or use --device cpu/cuda.")
    if name == "mps":
        # Leave room for macOS/applications; fail explicitly rather than paging
        # the entire workstation if a backend cache grows unexpectedly.
        torch.mps.set_per_process_memory_fraction(0.65)
    return torch.device(name)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def autocast(device, precision):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else contextlib.nullcontext()


def precision_for(device, precision="auto"):
    if precision == "auto":
        if device.type == "cuda":
            return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        return "bf16" if device.type == "mps" else "fp32"
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This NVIDIA GPU does not support BF16. Use --precision fp16 (for example on a T4).")
    return precision


def optimizer_update(model, optimizer, scheduler, scaler):
    # Clipping scaled gradients would silently distort updates. Unscale first.
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                         error_if_nonfinite=not scaler.is_enabled())
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    applied = scaler.get_scale() >= old_scale
    if applied:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return norm, applied


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, use_fast=False,
        cache_dir=ROOT / ".cache/huggingface", local_files_only=True,
    )


def load_model(seed, checkpoint=None):
    seed_all(seed)
    if checkpoint is None:
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=REVISION, cache_dir=ROOT / ".cache/huggingface",
            local_files_only=True, num_labels=2, problem_type="single_label_classification",
            id2label={0: "human", 1: "AI"}, label2id={"human": 0, "AI": 1},
        )
    return AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.encoded = tokenizer(frame.text.tolist(), truncation=True, max_length=max_length,
                                 padding=False, return_attention_mask=True)
        self.labels = frame.label.astype(int).tolist() if "label" in frame and frame.label.notna().all() else None
        self.lengths = [len(x) for x in self.encoded["input_ids"]]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.encoded.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


class LengthBatches(Sampler):
    """Shuffle length-bucketed batches to avoid padding every short passage to 512."""
    def __init__(self, lengths, batch_size, seed):
        self.lengths, self.batch_size, self.seed, self.epoch = lengths, batch_size, seed, 0

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths)).tolist()
        pool = self.batch_size * 50
        ordered = []
        for start in range(0, len(order), pool):
            ordered.extend(sorted(order[start:start + pool], key=lambda i: self.lengths[i]))
        batches = [ordered[i:i + self.batch_size] for i in range(0, len(ordered), self.batch_size)]
        rng.shuffle(batches)
        return iter(batches)


class PadBatch:
    def __init__(self, tokenizer, multiple=8):
        self.tokenizer = tokenizer
        self.multiple = multiple

    def __call__(self, examples):
        return self.tokenizer.pad(examples, padding=True, pad_to_multiple_of=self.multiple, return_tensors="pt")


def predict(model, frame, tokenizer, device, max_length=512, batch_size=16, precision="bf16"):
    data = TextDataset(frame, tokenizer, max_length)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        collate_fn=PadBatch(tokenizer, 128 if device.type == "mps" else 8))
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device, precision):
                logits = model(**batch).logits
            probabilities.extend(logits.float().softmax(-1)[:, 1].cpu().numpy())
    result = np.asarray(probabilities, dtype=float)
    if not np.isfinite(result).all() or not ((0 <= result) & (result <= 1)).all():
        raise RuntimeError("Nonfinite or invalid prediction probabilities.")
    return result


def train_model(train, inner, config, device, checkpoint_root, on_epoch, on_progress):
    """Train a fresh pretrained model; only `inner` may choose a checkpoint."""
    tokenizer = load_tokenizer()
    data = TextDataset(train, tokenizer, config["max_length"])
    batches = LengthBatches(data.lengths, config["batch_size"], config["seed"])
    loader = DataLoader(data, batch_sampler=batches,
                        collate_fn=PadBatch(tokenizer, 128 if device.type == "mps" else 8))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    resume_pointer = checkpoint_root / "resume_checkpoint.json"
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(train[["id", "text", "label"]], index=False).to_numpy().tobytes()).hexdigest()
    resume_dir = None
    if config.get("resume") and resume_pointer.exists():
        resume_dir = checkpoint_root / json.loads(resume_pointer.read_text())["directory"]
    model = load_model(config["seed"], resume_dir).to(device)
    groups = []
    # New pooler/classifier weights learn faster than the pretrained language encoder.
    for head in (False, True):
        for decay in (False, True):
            params = [p for name, p in model.named_parameters()
                      if p.requires_grad
                      and (not name.startswith("deberta.")) == head
                      and (not (name.endswith("bias") or "LayerNorm" in name)) == decay]
            groups.append({"params": params, "lr": config["head_lr"] if head else config["encoder_lr"],
                           "weight_decay": config["weight_decay"] if decay else 0.0})
    optimizer = torch.optim.AdamW(groups, eps=1e-8, fused=True if device.type == "mps" else None)
    scaler = torch.amp.GradScaler(device.type, enabled=config["precision"] == "fp16")
    updates_per_epoch = math.ceil(len(loader) / config["gradient_accumulation"])
    total_updates = updates_per_epoch * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_updates * config["warmup_ratio"]), total_updates)
    best_score, best_epoch, updates, skipped_updates = -float("inf"), 0, 0, 0
    started = time.perf_counter()
    history = []
    start_epoch = 1
    elapsed_before = 0.0
    if resume_dir is not None:
        # Only load state generated by this experiment, never an external pickle.
        state = torch.load(resume_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
        if state["configuration"] != config or state["training_hash"] != data_hash:
            raise ValueError("Resume configuration or training examples changed; use a fresh checkpoint directory.")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        best_score, best_epoch = state["best_score"], state["best_epoch"]
        updates, skipped_updates = state["updates"], state["skipped_updates"]
        start_epoch = state["completed_epoch"] + 1
        elapsed_before = state["elapsed_s"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(state["device_rng"])
        elif device.type == "cuda":
            torch.cuda.set_rng_state_all(state["device_rng"])
        del state
        print(f"Resuming after completed epoch {start_epoch - 1}, optimizer/scheduler/RNG restored.", flush=True)
    for epoch in range(start_epoch, config["epochs"] + 1):
        batches.epoch = epoch - 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = 0.0, 0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            count = len(batch["labels"])
            # Correct normalization even for a partial final accumulation window.
            window_start = ((step - 1) // config["gradient_accumulation"]) * config["gradient_accumulation"]
            window_size = min(config["gradient_accumulation"], len(loader) - window_start)
            with autocast(device, config["precision"]):
                output = model(**batch)
                raw_loss = output.loss
                loss = raw_loss / window_size
            if not torch.isfinite(raw_loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}, batch {step}.")
            scaler.scale(loss).backward()
            loss_sum += raw_loss.item() * count
            seen += count
            if step % config["gradient_accumulation"] == 0 or step == len(loader):
                norm, applied = optimizer_update(model, optimizer, scheduler, scaler)
                updates += int(applied)
                skipped_updates += int(not applied)
            if step % 25 == 0 or step == len(loader):
                sync(device)
                if device.type == "mps":
                    torch.mps.empty_cache()
                on_progress({"epoch": epoch, "batch": step, "batches_in_epoch": len(loader),
                             "optimizer_updates": updates, "mean_training_loss": loss_sum / seen,
                             "skipped_overflow_updates": skipped_updates,
                             "elapsed_s": elapsed_before + time.perf_counter() - started,
                             "gradient_norm": float(norm) if torch.isfinite(norm) else None,
                             "encoder_lr": scheduler.get_last_lr()[0],
                             "mps_allocated_gb": torch.mps.current_allocated_memory() / 1e9 if device.type == "mps" else None,
                             "mps_driver_gb": torch.mps.driver_allocated_memory() / 1e9 if device.type == "mps" else None})
        row = {"epoch": epoch, "training_loss": loss_sum / seen,
               "elapsed_s": elapsed_before + time.perf_counter() - started, "training_rows": len(train)}
        if inner is not None:
            p = predict(model, inner, tokenizer, device, config["max_length"],
                        config["eval_batch_size"], config["precision"])
            row["inner_auroc"] = roc_auc_score(inner.label.astype(int), p)
            if row["inner_auroc"] > best_score:
                best_score, best_epoch = row["inner_auroc"], epoch
                model.save_pretrained(checkpoint_root / "best", safe_serialization=True)
                tokenizer.save_pretrained(checkpoint_root / "best")
        elif epoch == config["epochs"]:
            best_epoch = epoch
            model.save_pretrained(checkpoint_root / "final", safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_root / "final")
        history.append(row)
        on_epoch(history)
        print(f"Epoch {epoch} complete: {row}", flush=True)
        if config.get("resume"):
            directory = Path(tempfile.mkdtemp(prefix="resume_", dir=checkpoint_root))
            model.save_pretrained(directory, safe_serialization=True)
            tokenizer.save_pretrained(directory)
            torch.save({"configuration": config, "training_hash": data_hash,
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "history": history,
                        "best_score": best_score, "best_epoch": best_epoch,
                        "updates": updates, "skipped_updates": skipped_updates,
                        "completed_epoch": epoch,
                        "elapsed_s": elapsed_before + time.perf_counter() - started,
                        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "device_rng": torch.mps.get_rng_state() if device.type == "mps" else
                                      torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                       directory / "trainer_state.pt")
            previous = resume_dir
            temporary_pointer = resume_pointer.with_suffix(".tmp")
            temporary_pointer.write_text(json.dumps({"directory": directory.name, "completed_epoch": epoch}) + "\n")
            temporary_pointer.replace(resume_pointer)
            resume_dir = directory
            if previous is not None and previous != directory and previous.parent == checkpoint_root:
                shutil.rmtree(previous)
        if config.get("one_epoch_per_process"):
            break
    del model, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, history
