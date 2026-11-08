"""Controlled RoBERTa supervised, soft-pseudo-label, and MLM training."""
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from src.deberta import LengthBatches, PadBatch, TextDataset, seed_all

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "FacebookAI/roberta-base"
REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"


def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
        cache_dir=ROOT / ".cache/huggingface", local_files_only=True)


def load_model(task="classification", checkpoint=None, adapted=None, seed=42):
    seed_all(seed)
    factory = AutoModelForMaskedLM if task == "mlm" else AutoModelForSequenceClassification
    options = {"local_files_only": True, "attn_implementation": "sdpa"}
    if checkpoint is None:
        options.update(revision=REVISION, cache_dir=ROOT / ".cache/huggingface")
    if task != "mlm":
        options.update(num_labels=2)
    model = factory.from_pretrained(checkpoint or MODEL_ID, **options)
    if adapted is not None:
        if checkpoint is not None or task == "mlm":
            raise ValueError("Adaptation applies only to a fresh classifier.")
        mlm = AutoModelForMaskedLM.from_pretrained(adapted, local_files_only=True)
        # All classifier arms retain the same seeded initial classification head.
        model.roberta.load_state_dict(mlm.roberta.state_dict(), strict=True)
        del mlm
    seed_all(seed)
    return model


class SoftDataset(TextDataset):
    def __init__(self, frame, tok, max_length=512):
        super().__init__(frame[["text"]], tok, max_length)
        self.targets = frame.soft_target.to_numpy(dtype=np.float32)
        if not np.isfinite(self.targets).all() or ((self.targets < 0) | (self.targets > 1)).any():
            raise ValueError("Invalid probability targets.")

    def __getitem__(self, index):
        return {**super().__getitem__(index), "target": float(self.targets[index])}


class SoftPad:
    def __init__(self, tok, multiple=128):
        self.pad = PadBatch(tok, multiple)

    def __call__(self, examples):
        batch = self.pad([{k: v for k, v in row.items() if k != "target"} for row in examples])
        batch["target"] = torch.tensor([row["target"] for row in examples], dtype=torch.float32)
        return batch


class MaskPad:
    """15% dynamic masking; special/padding tokens never become prediction targets."""
    def __init__(self, tok, multiple=128):
        self.tok, self.pad = tok, PadBatch(tok, multiple)

    def __call__(self, examples):
        batch = self.pad(examples)
        ids = batch["input_ids"].clone()
        protected = torch.tensor([self.tok.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True)
                                  for row in ids], dtype=torch.bool)
        protected |= batch["attention_mask"].eq(0)
        eligible = ~protected
        selected = torch.rand(ids.shape).lt(.15) & eligible
        if not selected.any():
            locations = eligible.nonzero()
            if not len(locations):
                raise ValueError("MLM batch contains no maskable tokens.")
            chosen = locations[torch.randint(len(locations), (1,)).item()]
            selected[chosen[0], chosen[1]] = True
        labels = ids.clone().masked_fill(~selected, -100)
        replacement = torch.rand(ids.shape)
        ids[selected & replacement.lt(.8)] = self.tok.mask_token_id
        random_positions = selected & replacement.ge(.8) & replacement.lt(.9)
        ids[random_positions] = torch.randint(len(self.tok), ids.shape)[random_positions]
        batch["input_ids"], batch["labels"] = ids, labels
        return batch


def masked_loss(model, batch):
    """Compute vocabulary logits only at masked positions, with the exact MLM loss."""
    labels = batch["labels"]
    text = {k: v for k, v in batch.items() if k != "labels"}
    hidden = model.roberta(**text).last_hidden_state
    selected = labels.ne(-100)
    return nn.functional.cross_entropy(model.lm_head(hidden[selected]), labels[selected])


def soft_loss(logits, targets):
    return nn.functional.binary_cross_entropy_with_logits(logits[:, 1] - logits[:, 0], targets)


def agreement_pseudo(boosting, neural, boost_threshold=.99, neural_threshold=.90):
    b, n = np.asarray(boosting), np.asarray(neural)
    if b.shape != n.shape or not np.isfinite(b).all() or not np.isfinite(n).all():
        raise ValueError("Unaligned or invalid teacher probabilities.")
    if ((b < 0) | (b > 1) | (n < 0) | (n > 1)).any():
        raise ValueError("Teacher outputs must be probabilities.")
    keep = ((b >= .5) == (n >= .5)) & (np.maximum(b, 1-b) >= boost_threshold) & (np.maximum(n, 1-n) >= neural_threshold)
    return keep, (b + n) / 2


def predict(model, frame, tok, device, batch_size=8, progress=None):
    dataset = TextDataset(frame[["text"]], tok, 512)
    order = np.argsort(dataset.lengths, kind="stable")
    pad = PadBatch(tok, 128 if device.type == "mps" else 8)
    result = np.empty(len(frame), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = pad([dataset[int(i)] for i in indices])
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits.cpu().double()
            result[indices] = logits.softmax(-1)[:, 1].numpy()
            if (start // batch_size + 1) % 25 == 0 or start + batch_size >= len(order):
                if device.type == "mps":
                    torch.mps.synchronize(); torch.mps.empty_cache()
                if progress:
                    progress(min(start + batch_size, len(order)), len(order))
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise RuntimeError("Invalid RoBERTa probabilities.")
    return result


def cpu_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return obj


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def train_epoch(frame, config, directory, epoch, device, progress, task="classification", pseudo=None, adapted=None):
    """One recoverable epoch; extra pseudo batches never add optimizer updates."""
    directory = Path(directory)
    output = directory / f"epoch_{epoch}"
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    previous = directory / f"epoch_{epoch - 1}" if epoch > 1 else None
    tok = tokenizer()
    data = TextDataset(frame[["text"]], tok, config["max_length"]) if task == "mlm" else SoftDataset(frame, tok, config["max_length"])
    sampler = LengthBatches(data.lengths, config["batch_size"], config["seed"])
    sampler.epoch = epoch - 1
    multiple = 128 if device.type == "mps" else 8
    collator = MaskPad(tok, multiple) if task == "mlm" else SoftPad(tok, multiple)
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=collator)
    model = load_model(task, previous, adapted if previous is None else None, config["seed"]).to(device)
    groups = []
    for encoder in [True, False]:
        for decay in [True, False]:
            params = [p for name, p in model.named_parameters() if p.requires_grad
                      and name.startswith("roberta.") == encoder and (p.ndim > 1) == decay]
            groups.append({"params": params, "lr": config["encoder_lr"] if encoder else config["head_lr"],
                           "weight_decay": config["weight_decay"] if decay else 0.})
    optimizer = torch.optim.AdamW(groups, fused=True if device.type == "mps" else None)
    updates_per_epoch = math.ceil(len(loader) / config["gradient_accumulation"])
    total = updates_per_epoch * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total * .1), total)
    signature = {"frame": frame_hash(frame), "pseudo": frame_hash(pseudo) if pseudo is not None else None,
                 "configuration": config, "task": task, "adapted": str(adapted)}
    updates = 0
    if previous is not None:
        state = torch.load(previous / "trainer_state.pt", map_location="cpu", weights_only=False)
        assert state["signature"] == signature, "Resume inputs changed."
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(state["device_rng"])
        updates = state["updates"]
    pseudo_data, pseudo_batches, pseudo_index = None, None, 0
    if pseudo is not None and len(pseudo):
        pseudo_data = SoftDataset(pseudo, tok, config["max_length"])
        pseudo_sampler = LengthBatches(pseudo_data.lengths, config["batch_size"], config["seed"] + 100)
        pseudo_sampler.epoch = epoch - 1
        pseudo_batches = list(pseudo_sampler)
    model.train(); optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    total_loss, seen, pseudo_loss_sum, pseudo_batches_seen = 0., 0, 0., 0
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch.pop("target", None)
        loss = masked_loss(model, batch) if task == "mlm" else soft_loss(model(**batch).logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite RoBERTa training loss.")
        window_start = ((step - 1) // config["gradient_accumulation"]) * config["gradient_accumulation"]
        window = min(config["gradient_accumulation"], len(loader) - window_start)
        (loss / window).backward()
        count = len(batch["input_ids"])
        total_loss += loss.item() * count; seen += count
        if step % config["gradient_accumulation"] == 0 or step == len(loader):
            if pseudo_batches:
                indices = pseudo_batches[pseudo_index % len(pseudo_batches)]; pseudo_index += 1
                pb = SoftPad(tok, multiple)([pseudo_data[int(i)] for i in indices])
                pb = {k: v.to(device) for k, v in pb.items()}; pt = pb.pop("target")
                auxiliary = soft_loss(model(**pb).logits, pt)
                if not torch.isfinite(auxiliary):
                    raise RuntimeError("Nonfinite pseudo-label loss.")
                (config["pseudo_weight"] * auxiliary).backward()
                pseudo_loss_sum += auxiliary.item(); pseudo_batches_seen += 1
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); updates += 1
        if step % 50 == 0 or step == len(loader):
            if device.type == "mps":
                torch.mps.synchronize(); torch.mps.empty_cache()
            row = {"epoch": epoch, "batch": step, "batches": len(loader), "updates": updates,
                   "loss": total_loss / seen, "pseudo_loss": pseudo_loss_sum / max(pseudo_batches_seen, 1),
                   "elapsed_s": time.perf_counter() - started,
                   "driver_gb": torch.mps.driver_allocated_memory() / 1e9 if device.type == "mps" else None,
                   "gradient_norm": float(norm)}
            progress(row)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True); tok.save_pretrained(output)
    torch.save({"optimizer": cpu_tree(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
                "updates": updates, "signature": signature, "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "device_rng": torch.mps.get_rng_state() if device.type == "mps" else None}, output / "trainer_state.pt")
    result = {"epoch": epoch, "updates": updates, "training_rows": len(frame), "loss": total_loss / seen,
              "pseudo_rows_available": len(pseudo) if pseudo is not None else 0,
              "pseudo_batches_used": pseudo_batches_seen, "elapsed_s": time.perf_counter() - started,
              "signature": signature}
    (output / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
