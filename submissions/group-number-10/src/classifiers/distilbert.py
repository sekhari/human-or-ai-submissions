"""DistilBERT fine-tuning (PyTorch + Hugging Face Transformers). Owner: Marcos.

Human (0) vs AI (1). Implements the team interface (fit / predict_proba /
get_config) so it plugs into ``src.evaluate``.

* Tokenization: truncation to ``max_length`` (<= 512) with dynamic padding per batch.
* Optimizer: AdamW (no weight decay on bias/LayerNorm) + linear LR schedule.
* Device: CUDA -> Apple MPS -> CPU (``src.utils.get_device``).
* If validation data is passed to ``fit``, the epoch with the best validation
  AUROC is kept (and optionally saved to disk); otherwise the last epoch is kept.
* Probabilities: softmax(logits)[:, 1].
* Optional ``newline_marker`` (e.g. "[NL]"): every "\n" becomes this added token, because the
  BERT tokenizer otherwise treats line breaks as plain whitespace and drops them.
  ``space_marker`` (e.g. "[DS]") does the same for runs of 2+ spaces.
* Pretrained weights download to the Hugging Face cache, not into the repo.
  Fine-tuned checkpoints go under ``checkpoints/`` (git-ignored).

Inference without retraining:

    from src.classifiers.distilbert import predict_proba_from_checkpoint
    probs = predict_proba_from_checkpoint(["some passage", ...], "checkpoints/distilbert/<run>")
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AddedToken,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.layout import augment_layouts
from src.utils import get_device, set_seed

CONFIG_FILE = "team10_distilbert.json"


@dataclass
class DistilBertConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    learning_rate: float = 2e-5
    batch_size: int = 16
    eval_batch_size: int = 64
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42
    newline_marker: str | None = None
    space_marker: str | None = None
    bf16: bool = False  # bfloat16 autocast for forward passes; weights and optimizer state stay fp32
    gradient_checkpointing: bool = False  # recompute activations in backward to save memory
    layout_aug: float = 0.0  # share of training passages re-laid-out each epoch (src.layout), label-independent


def whitespace_markers(cfg: DistilBertConfig) -> list[str]:
    return [m for m in (cfg.newline_marker, cfg.space_marker) if m]


def mark_whitespace(text: str, cfg: DistilBertConfig) -> str:
    """Make whitespace the tokenizer would discard visible as marker tokens."""
    if cfg.space_marker:
        text = re.sub(r" {2,}", f" {cfg.space_marker} ", text)  # before newlines add their own spaces
    if cfg.newline_marker:
        text = text.replace("\n", f" {cfg.newline_marker} ")
    return text


def add_marker_tokens(tokenizer, model, cfg: DistilBertConfig) -> None:
    """Register marker tokens (matched before lowercasing) and grow the embedding matrix to match."""
    new = [m for m in whitespace_markers(cfg) if m not in tokenizer.get_vocab()]
    if new:
        tokenizer.add_tokens([AddedToken(m, normalized=False) for m in new])
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


class _EncodedDataset(Dataset):
    def __init__(self, encodings: dict[str, list], labels: np.ndarray | None = None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {k: v[i] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


class DistilBertClassifier:
    def __init__(self, config: DistilBertConfig | None = None, device: str | None = None, **overrides: Any) -> None:
        self.config = replace(config or DistilBertConfig(), **overrides)
        self.device = get_device(device)
        self.model: torch.nn.Module | None = None
        self.tokenizer = None
        self.history: list[dict[str, Any]] = []
        self.best_epoch: int | None = None

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, num_labels=2)
        add_marker_tokens(self.tokenizer, self.model, self.config)
        self.model.to(self.device)

    def _encode(self, texts: Sequence[str]) -> dict[str, list]:
        if whitespace_markers(self.config):
            texts = [mark_whitespace(t, self.config) for t in texts]
        enc = self.tokenizer(list(texts), truncation=True, max_length=self.config.max_length)
        return {k: enc[k] for k in ("input_ids", "attention_mask")}

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.config.bf16)

    # --------------------------------------------------------------- training
    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        val_texts: Sequence[str] | None = None,
        val_labels: Sequence[int] | None = None,
        checkpoint_dir: Path | str | None = None,
        log: Callable[[str], None] = print,
        log_every: int = 50,
    ) -> "DistilBertClassifier":
        cfg = self.config
        set_seed(cfg.seed)
        self._build()
        model = self.model
        if cfg.gradient_checkpointing:
            # Non-reentrant: DeBERTa feeds shared relative-position embeddings into every layer, which the
            # reentrant variant backpropagates through twice ("backward through the graph a second time").
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        generator = torch.Generator().manual_seed(cfg.seed)

        def make_loader(texts: Sequence[str]) -> DataLoader:
            return DataLoader(
                _EncodedDataset(self._encode(texts), np.asarray(train_labels)),
                batch_size=cfg.batch_size,
                shuffle=True,
                generator=generator,
                collate_fn=DataCollatorWithPadding(self.tokenizer),
            )

        train_loader = make_loader(train_texts)

        no_decay = ("bias", "LayerNorm.weight")
        groups = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=cfg.learning_rate)
        total_steps = len(train_loader) * cfg.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)

        self.history, self.best_epoch = [], None
        best_auc, best_state = -np.inf, None
        has_val = val_texts is not None and val_labels is not None

        for epoch in range(1, cfg.epochs + 1):
            if cfg.layout_aug:  # fresh random layouts every epoch; same number of batches
                train_loader = make_loader(augment_layouts(list(train_texts), cfg.layout_aug,
                                                           np.random.default_rng([cfg.seed, epoch])))
            model.train()
            t0 = time.perf_counter()
            loss_sum, n_seen = 0.0, 0
            for step, batch in enumerate(train_loader, 1):
                batch = self._to_device(batch)
                with self._autocast():
                    loss = model(**batch).loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                bs = batch["labels"].shape[0]
                loss_sum += loss.item() * bs
                n_seen += bs
                if log_every and step % log_every == 0:
                    log(f"epoch {epoch} step {step}/{len(train_loader)} train_loss {loss_sum / n_seen:.4f} "
                        f"({time.perf_counter() - t0:.0f}s)")

            record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": loss_sum / n_seen,  # mean loss over the epoch's training batches
                "train_seconds": round(time.perf_counter() - t0, 2),
                "lr_end": scheduler.get_last_lr()[0],
            }
            if self.device.type == "cuda":
                record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(self.device) / 1e9, 1)
            if has_val:
                val_probs, val_loss = self._predict(val_texts, val_labels)
                record["val_loss"] = val_loss
                record["val_auc"] = float(roc_auc_score(val_labels, val_probs))
                if record["val_auc"] > best_auc:
                    best_auc, self.best_epoch = record["val_auc"], epoch
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            self.history.append(record)
            log(json.dumps(record))
            if has_val and checkpoint_dir is not None and self.best_epoch == epoch:
                self.save(checkpoint_dir)  # after history is updated, so the saved metadata is complete

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            self.best_epoch = cfg.epochs
            if checkpoint_dir is not None:
                self.save(checkpoint_dir)
        return self

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def _predict(self, texts: Sequence[str], labels: Sequence[int] | None = None) -> tuple[np.ndarray, float | None]:
        if len(texts) == 0:
            return np.empty(0, dtype=np.float64), None
        self.model.eval()
        enc = self._encode(texts)
        order = np.argsort([len(x) for x in enc["input_ids"]], kind="stable")  # length-sorted batches = less padding
        enc_sorted = {k: [v[i] for i in order] for k, v in enc.items()}
        y_sorted = None if labels is None else np.asarray(labels)[order]
        loader = DataLoader(_EncodedDataset(enc_sorted, y_sorted), batch_size=self.config.eval_batch_size,
                            shuffle=False, collate_fn=DataCollatorWithPadding(self.tokenizer))
        probs, loss_sum = [], 0.0
        for batch in loader:
            with self._autocast():
                out = self.model(**self._to_device(batch))
            probs.append(torch.softmax(out.logits.float(), dim=-1)[:, 1].cpu().numpy())
            if labels is not None:
                loss_sum += out.loss.item() * batch["input_ids"].shape[0]
        p = np.empty(len(texts), dtype=np.float64)
        p[order] = np.concatenate(probs)  # restore input order
        return p, (loss_sum / len(texts) if labels is not None else None)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """1-D array of P(AI) in [0, 1], one per input, in input order."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call fit() or DistilBertClassifier.load().")
        return self._predict(texts)[0]

    def get_config(self) -> dict[str, Any]:
        return {**asdict(self.config), "best_epoch": self.best_epoch}

    # ----------------------------------------------------------- persistence
    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        meta = {"config": asdict(self.config), "best_epoch": self.best_epoch, "history": self.history}
        (directory / CONFIG_FILE).write_text(json.dumps(meta, indent=2) + "\n")
        return directory

    @classmethod
    def load(cls, directory: Path | str, device: str | None = None) -> "DistilBertClassifier":
        directory = Path(directory)
        meta = json.loads((directory / CONFIG_FILE).read_text())
        known = {f.name for f in fields(DistilBertConfig)}
        clf = cls(DistilBertConfig(**{k: v for k, v in meta["config"].items() if k in known}), device=device)
        clf.tokenizer = AutoTokenizer.from_pretrained(directory)
        clf.model = AutoModelForSequenceClassification.from_pretrained(directory).to(clf.device)
        clf.best_epoch, clf.history = meta.get("best_epoch"), meta.get("history", [])
        return clf


def predict_proba_from_checkpoint(texts: Sequence[str], checkpoint_dir: Path | str, device: str | None = None) -> np.ndarray:
    """Load a saved checkpoint and return P(AI) for each passage, without retraining."""
    return DistilBertClassifier.load(checkpoint_dir, device=device).predict_proba(texts)
