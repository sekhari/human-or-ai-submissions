"""Train the existing TextCNN with reusable, train-only token vocabularies."""

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import TextCNN, predict_probabilities, run_epoch
from preprocess import clean_text


@dataclass(frozen=True)
class CNNConfig:
    max_vocab: int = 30000
    max_length: int = 768
    embedding_dim: int = 100
    embedding_mode: str = "random"
    embedding_path: str | None = None
    embedding_lowercase_fallback: bool = False
    filters: int = 64
    dropout: float = 0.5
    batch_size: int = 64
    epochs: int = 30
    patience: int = 3
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    seed: int = 12
    device: str = "auto"

    def __post_init__(self):
        if self.max_vocab < 2 or self.max_length < 5:
            raise ValueError("Need max_vocab >= 2 and max_length >= 5 for the CNN kernels")
        if min(self.batch_size, self.epochs, self.patience, self.embedding_dim, self.filters) < 1:
            raise ValueError("Counts must be positive")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid dropout or optimizer settings")
        if self.embedding_mode not in ("random", "frozen", "finetune"):
            raise ValueError("Embedding mode must be random, frozen, or finetune")
        if self.embedding_mode != "random" and not self.embedding_path:
            raise ValueError("Pretrained modes require --embeddings pointing to a text vector file")


class SequenceEncoder:
    """Preserve case and explicit newline tokens; optionally run the old cleaner."""

    pattern = re.compile(r"\r\n|[\r\n]|\w+|[^\w\s]", re.UNICODE)

    def __init__(self, mode, max_vocab, max_length):
        if mode not in ("cnn_raw", "cnn_clean"):
            raise ValueError(f"Unknown neural model: {mode}")
        self.mode, self.max_vocab, self.max_length = mode, max_vocab, max_length
        self.vocabulary = None

    def tokenize(self, text):
        if self.mode == "cnn_clean":
            text = clean_text(text, lowercase=False)
        return ["\n" if token in ("\r", "\r\n") else token for token in self.pattern.findall(text)]

    def fit(self, texts):
        counts = Counter(token for text in texts for token in self.tokenize(text))
        self.vocabulary = {
            token: index for index, (token, _) in enumerate(
                counts.most_common(self.max_vocab - 2), start=2,
            )
        }
        return self

    def encode(self, texts):
        if self.vocabulary is None:
            raise ValueError("Fit the vocabulary on training texts first")
        texts = list(texts)
        encoded = np.zeros((len(texts), self.max_length), dtype=np.int64)
        for row, text in enumerate(texts):
            tokens = self.tokenize(text)[:self.max_length]
            encoded[row, :len(tokens)] = [self.vocabulary.get(token, 1) for token in tokens]
        return encoded


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_loader(tokens, labels, config, shuffle=False):
    tensors = [torch.as_tensor(tokens, dtype=torch.long)]
    if labels is not None:
        tensors.append(torch.as_tensor(np.asarray(labels), dtype=torch.float32))
    return DataLoader(TensorDataset(*tensors), batch_size=config.batch_size, shuffle=shuffle)


def validate_vector_file(path, dimension):
    """Fail before a long comparison if the vector path or dimensions are wrong."""
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            parts = line.split()
            if not parts:
                continue
            size = int(parts[1]) if len(parts) == 2 and all(part.isdigit() for part in parts) else len(parts) - 1
            if size != dimension:
                raise ValueError(f"Embedding file has {size} dimensions; pass --embedding-dim {size}")
            return
    raise ValueError("Embedding file is empty")


def load_word_vectors(path, vocabulary, dimension, lowercase_fallback=False):
    """Read GloVe/plain word vectors, retaining only training-vocabulary entries.

    Exact matches take priority over an optional lowercase lookup. Lowercase
    lookup initializes vectors only; it does not change tokenizer IDs or text.
    A word2vec-style count/dimension header is also accepted.
    """
    needed = set(vocabulary)
    if lowercase_fallback:
        needed.update(token.lower() for token in vocabulary)
    found, digest, first_content, checked_dimension = {}, hashlib.sha256(), True, False
    with Path(path).open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            digest.update(line)
            parts = line.split(maxsplit=1)
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"Invalid embedding row at line {line_number}")
            token_bytes, values = parts
            if first_content and token_bytes.isdigit() and values.strip().isdigit():
                if int(values) != dimension:
                    raise ValueError(f"Embedding file has {int(values)} dimensions; --embedding-dim is {dimension}")
                first_content = False
                continue
            first_content = False
            token = token_bytes.decode("utf-8")
            if token in needed or not checked_dimension:
                vector = np.fromstring(values.decode("utf-8"), dtype=np.float32, sep=" ")
                if vector.size != dimension or not np.isfinite(vector).all():
                    raise ValueError(f"Expected {dimension} finite values at embedding line {line_number}")
                checked_dimension = True
                if token in needed:
                    found[token] = vector
    matched, exact, lower = {}, 0, 0
    for token, token_id in vocabulary.items():
        if token in found:
            matched[token_id] = found[token]
            exact += 1
        elif lowercase_fallback and token.lower() in found:
            matched[token_id] = found[token.lower()]
            lower += 1
    if not matched:
        raise ValueError("No training-vocabulary tokens matched the embedding file")
    report = {
        "source": str(Path(path).resolve()), "source_sha256": digest.hexdigest(),
        "dimension": dimension, "vocabulary_tokens": len(vocabulary),
        "matched_tokens": len(matched), "exact_matches": exact,
        "lowercase_matches": lower, "coverage": len(matched) / len(vocabulary),
        "unmatched_tokens": len(vocabulary) - len(matched),
    }
    return matched, report


def initialize(texts, mode, config, load_pretrained=True):
    seed_training(config.seed)
    device = select_device(config.device)
    encoder = SequenceEncoder(mode, config.max_vocab, config.max_length).fit(texts)
    model = TextCNN(
        config.max_vocab, embedding_dim=config.embedding_dim,
        filters=config.filters, dropout=config.dropout,
    )
    model.embedding_report = {"mode": config.embedding_mode, "dimension": config.embedding_dim}
    if config.embedding_mode != "random" and load_pretrained:
        vectors, report = load_word_vectors(
            config.embedding_path, encoder.vocabulary, config.embedding_dim,
            config.embedding_lowercase_fallback,
        )
        with torch.no_grad():
            for token_id, vector in vectors.items():
                model.embedding.weight[token_id].copy_(torch.from_numpy(vector))
        model.embedding_report.update(report)
        print(
            f"  Pretrained vocabulary coverage: {report['matched_tokens']}/{report['vocabulary_tokens']} "
            f"({report['coverage']:.1%}); lowercase matches={report['lowercase_matches']}", flush=True,
        )
    model.embedding.weight.requires_grad_(config.embedding_mode != "frozen")
    model = model.to(device)
    return model, encoder, device


def save_checkpoint(path, model, encoder, config, epoch, **metadata):
    """Save immediately and atomically, including the exact token-to-ID mapping."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "vocabulary": encoder.vocabulary, "mode": encoder.mode,
        "config": asdict(config), "epoch": epoch,
        "embedding_report": model.embedding_report,
        "torch_version": str(torch.__version__), **metadata,
    }, temporary)
    temporary.replace(path)


class CNNPredictor:
    def __init__(self, model, encoder, config, device):
        self.model, self.encoder, self.config, self.device = model, encoder, config, device

    def predict(self, texts):
        loader = make_loader(self.encoder.encode(texts), None, self.config)
        return predict_probabilities(self.model, loader, self.device)


def load_predictor(path, device="auto"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = replace(CNNConfig(**checkpoint["config"]), device=device)
    # Inference uses saved weights; it never needs the original vector download.
    model, encoder, selected_device = initialize([], checkpoint["mode"], config, load_pretrained=False)
    encoder.vocabulary = checkpoint["vocabulary"]
    model.load_state_dict(checkpoint["state_dict"])
    model.embedding_report = checkpoint.get("embedding_report", {})
    return CNNPredictor(model, encoder, config, selected_device)


def train_with_early_stopping(
    train_text,
    train_labels,
    validation_text,
    validation_labels,
    mode,
    config,
    destination,
):
    """Train on the training partition and keep the best validation checkpoint."""
    if len(np.unique(validation_labels)) != 2:
        raise ValueError("Early stopping requires both labels")
    model, encoder, device = initialize(train_text, mode, config)
    train_loader = make_loader(encoder.encode(train_text), train_labels, config, shuffle=True)
    validation_loader = make_loader(
        encoder.encode(validation_text), validation_labels, config
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "embedding_report.json").write_text(
        json.dumps(model.embedding_report, indent=2) + "\n"
    )
    best_auc, best_epoch, stale, history = -np.inf, 0, 0, []
    print(
        f"  Training on {device}: train={len(train_text)}, "
        f"validation={len(validation_text)}",
        flush=True,
    )
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, probabilities = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            return_probabilities=True,
        )
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError("Training produced a non-finite loss")
        auc = float(roc_auc_score(validation_labels, probabilities))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_auroc": auc,
            }
        )
        pd.DataFrame(history).to_csv(destination / "history.csv", index=False)
        print(
            f"  Epoch {epoch:02d}: train loss={train_loss:.4f}, "
            f"validation AUROC={auc:.4f}",
            flush=True,
        )
        if auc > best_auc:
            best_auc, best_epoch, stale = auc, epoch, 0
            save_checkpoint(
                destination / "model.pt",
                model,
                encoder,
                config,
                epoch,
                validation_auroc=auc,
            )
        else:
            stale += 1
            if stale >= config.patience:
                print(f"  Early stopping; selected epoch {best_epoch}", flush=True)
                break
    return load_predictor(destination / "model.pt", device=config.device), best_epoch


def refit(texts, labels, mode, config, epochs, destination):
    """Refit from scratch for a fixed epoch count, without observing the holdout."""
    if epochs < 1:
        raise ValueError("Refit epochs must be positive")
    model, encoder, device = initialize(texts, mode, config)
    loader = make_loader(encoder.encode(texts), labels, config, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    history = []
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "embedding_report.json").write_text(json.dumps(model.embedding_report, indent=2) + "\n")
    print(f"  Refit on {len(texts)} rows for {epochs} epochs on {device}", flush=True)
    for epoch in range(1, epochs + 1):
        loss = run_epoch(model, loader, criterion, device, optimizer)
        if not np.isfinite(loss):
            raise RuntimeError("Refit produced a non-finite loss")
        history.append({"epoch": epoch, "train_loss": loss})
        pd.DataFrame(history).to_csv(destination / "refit_history.csv", index=False)
        print(f"  Refit epoch {epoch:02d}: loss={loss:.4f}", flush=True)
    save_checkpoint(destination / "model.pt", model, encoder, config, epochs)
    return CNNPredictor(model, encoder, config, device)
