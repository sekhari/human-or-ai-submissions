"""Data, tokenization, checkpoint, and inference utilities."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


MODEL_NAME = "openai-community/gpt2"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA device is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("The requested MPS device is unavailable.")
    if device.type not in {"cuda", "mps", "cpu"}:
        raise ValueError("Device must be auto, cpu, mps, or cuda.")
    return device


def read_text_csv(path: str | Path, *, labeled: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    required = {"id", "text", "label"} if labeled else {"id", "text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"No rows found in {path}.")
    if frame["id"].str.strip().eq("").any() or frame["id"].duplicated().any():
        raise ValueError(f"IDs must be nonempty and unique in {path}.")
    if not frame["text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError(f"Text must be a nonempty string in every row of {path}.")
    if labeled:
        labels = pd.to_numeric(frame["label"], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("Training labels must be 0 or 1.")
        frame["label"] = labels.astype(np.int64)
    return frame


def load_data(train_csv: str | Path, val_ids_csv: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return training rows and validation rows in the saved validation-ID order."""
    frame = read_text_csv(train_csv, labeled=True)
    ids = pd.read_csv(val_ids_csv, dtype={"id": str}, keep_default_na=False)
    if "id" not in ids or ids.empty:
        raise ValueError("The validation ID file must contain a nonempty id column.")
    if ids["id"].str.strip().eq("").any() or ids["id"].duplicated().any():
        raise ValueError("Validation IDs must be nonempty and unique.")
    unknown = set(ids["id"]) - set(frame["id"])
    if unknown:
        raise ValueError(f"Validation IDs absent from training data: {sorted(unknown)[:5]}")
    validation = frame.set_index("id").loc[ids["id"].tolist()].reset_index()
    training = frame.loc[~frame["id"].isin(ids["id"])].reset_index(drop=True)
    for name, part in (("training", training), ("validation", validation)):
        if set(part["label"]) != {0, 1}:
            raise ValueError(f"The {name} partition must contain both label classes.")
    return training, validation


def make_tokenizer(model_name_or_dir: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_dir), use_fast=True)
    if tokenizer.eos_token is None:
        raise ValueError("The tokenizer must define an end-of-sequence token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


class EncodedPassages(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.rows[index])


def encode_frame(frame: pd.DataFrame, tokenizer, max_length: int) -> EncodedPassages:
    if not 1 <= max_length <= 1024:
        raise ValueError("max_length must be between 1 and 1024 for GPT-2.")
    rows: list[dict[str, Any]] = []
    texts = frame["text"].tolist()
    labels = frame["label"].tolist() if "label" in frame else None
    for start in range(0, len(texts), 256):
        encoded = tokenizer(
            texts[start:start + 256], truncation=True, max_length=max_length,
            padding=False, return_attention_mask=True, return_token_type_ids=False,
        )
        for offset, input_ids in enumerate(encoded["input_ids"]):
            if not input_ids:
                raise ValueError("A passage produced an empty token sequence.")
            row: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"][offset],
            }
            if labels is not None:
                row["labels"] = float(labels[start + offset])
            rows.append(row)
    return EncodedPassages(rows)


def make_loader(dataset, tokenizer, batch_size: int, shuffle: bool = False, seed: int = 42):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
        generator=generator, num_workers=0, pin_memory=False,
    )


@torch.inference_mode()
def predict_probabilities(model, loader, device) -> np.ndarray:
    model.eval()
    predictions = []
    for batch in loader:
        inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        with autocast_context(device):
            logits = model(**inputs).logits.squeeze(-1)
        predictions.append(torch.sigmoid(logits.float()).cpu().numpy())
    result = np.concatenate(predictions).astype(np.float64) if predictions else np.empty(0, dtype=np.float64)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise RuntimeError("Predicted probabilities must be finite and within [0, 1].")
    return result


def autocast_context(device):
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return contextlib.nullcontext()


def flush_file(stream):
    stream.flush()
    try:
        os.fsync(stream.fileno())
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
        flush_file(stream)
    os.replace(temporary, destination)


def recover_model_directory(destination: str | Path) -> Path:
    destination = Path(destination)
    previous = destination.with_name(destination.name + ".previous")
    if not destination.exists() and previous.is_dir():
        previous.rename(destination)
    return destination


def save_model(model, tokenizer, destination: str | Path) -> None:
    """Export weights and tokenizer together, preserving the prior directory until commit."""
    destination = recover_model_directory(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    previous = destination.with_name(destination.name + ".previous")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    model.save_pretrained(temporary, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    if previous.exists():
        shutil.rmtree(previous)
    if destination.exists():
        destination.rename(previous)
    try:
        temporary.rename(destination)
    except BaseException:
        if previous.exists() and not destination.exists():
            previous.rename(destination)
        raise
    if previous.exists():
        shutil.rmtree(previous)
