"""Shared I/O and inference for the portable raw-text RoBERTa entrypoints."""
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import tempfile


MODEL_NAME = "FacebookAI/roberta-base"
MODEL_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
PACKAGE_ROOT = Path(__file__).resolve().parent
RECIPE = {
    "epochs": 2,
    "microbatch": 4,
    "gradient_accumulation": 4,
    "max_length": 512,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "seed": 42,
    "cpu_threads": 10,
    "dtype": "float32",
    "attention": "sdpa",
    "learning_rate_schedule": "constant; no warmup",
    "weight_decay_scope": "all trainable parameters, including bias and normalization",
}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv_exclusive(path, rows, fields):
    """Publish a complete CSV atomically, refusing even a concurrent overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def id_digest(records):
    return hashlib.sha256("\n".join(row["id"] for row in records).encode("utf-8")).hexdigest()


def package_versions():
    return {name: version(name) for name in (
        "torch", "transformers", "huggingface-hub", "tokenizers", "safetensors", "numpy"
    )}


def resolve_device(torch, requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is unavailable")
    return requested


def seed_training(torch, device, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def epoch_order(size, seed, epoch):
    # The original experiment shuffles independently from torch's dropout RNG.
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def update_groups(order, microbatch, accumulation):
    effective_batch = microbatch * accumulation
    for start in range(0, len(order), effective_batch):
        indices = order[start:start + effective_batch]
        micros = [indices[pos:pos + microbatch] for pos in range(0, len(indices), microbatch)]
        yield indices, micros


def tokenize(tokenizer, records, max_length):
    # Keep the original text byte-for-byte at the Python string level, including whitespace.
    encoded = tokenizer(
        [row["text"] for row in records], truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    )
    return {name: values for name, values in encoded.items()
            if name in ("input_ids", "attention_mask", "token_type_ids")}


def predict_records(torch, model, tokenizer, records, device, max_length, batch_size):
    probabilities, margins = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            encoded = tokenize(tokenizer, records[start:start + batch_size], max_length)
            batch = {name: values.to(device) for name, values in encoded.items()}
            # Match the original experiment's float64 CPU softmax of FP32 logits.
            logits = model(**batch).logits.detach().cpu().to(dtype=torch.float64)
            if logits.shape[1] != 2 or not torch.isfinite(logits).all().item():
                raise RuntimeError("Expected finite logits for classes 0 (human) and 1 (AI)")
            probabilities.extend(logits.softmax(-1)[:, 1].tolist())
            margins.extend((logits[:, 1] - logits[:, 0]).tolist())
    if len(probabilities) != len(records) or not all(
        math.isfinite(probability) and 0 <= probability <= 1 for probability in probabilities
    ):
        raise RuntimeError("Invalid prediction coverage or probabilities")
    return probabilities, margins


def auroc(labels, scores):
    """Mann-Whitney AUROC with tied ranks, as in the archived training worker."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels))
    concordance, preceding_negatives, start = 0.0, 0, 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[start][0]:
            stop += 1
        group_positives = sum(label for _, label in ordered[start:stop])
        group_negatives = stop - start - group_positives
        concordance += group_positives * (preceding_negatives + 0.5 * group_negatives)
        preceding_negatives += group_negatives
        start = stop
    return concordance / (positives * negatives)
