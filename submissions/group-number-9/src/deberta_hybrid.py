"""Cached DeBERTa representations and a neural text/style fusion head."""
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.deberta import PadBatch, TextDataset


def extract(model, frame, tokenizer, device, batch_size=8, max_length=512, progress=None):
    """Length-sort inference, then restore the exact input row order."""
    data = TextDataset(frame, tokenizer, max_length)
    order = np.argsort(data.lengths, kind="stable")
    pad = PadBatch(tokenizer, 128 if device.type == "mps" else 8)
    vectors = np.empty((len(data), model.config.pooler_hidden_size), dtype=np.float32)
    logits = np.empty(len(data), dtype=np.float32)
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = pad([data[int(i)] for i in indices])
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model.deberta(**batch).last_hidden_state
            pooled = model.pooler(hidden)
            scores = model.classifier(model.dropout(pooled))
            vectors[indices] = pooled.float().cpu().numpy()
            logits[indices] = (scores[:, 1] - scores[:, 0]).float().cpu().numpy()
            done = min(start + batch_size, len(order))
            if done % (batch_size * 25) == 0 or done == len(order):
                if device.type == "mps":
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                if progress:
                    progress({"rows": done, "total_rows": len(data),
                              "elapsed_s": time.perf_counter() - started})
    if not np.isfinite(vectors).all() or not np.isfinite(logits).all():
        raise RuntimeError("Nonfinite encoder representations.")
    return vectors, logits


def prepare_inputs(train_vectors, train_style, eval_vectors, eval_style, mode):
    """Fit both normalizers on fitting rows only; numeric counts get log1p."""
    if mode not in {"hybrid", "text_only", "style_only"}:
        raise ValueError(mode)
    text_scaler = StandardScaler().fit(train_vectors)
    style_scaler = StandardScaler().fit(np.log1p(train_style))
    def transform(vectors, style):
        t = np.clip(text_scaler.transform(vectors), -10, 10).astype(np.float32)
        s = np.clip(style_scaler.transform(np.log1p(style)), -10, 10).astype(np.float32)
        if mode == "text_only":
            s = s[:, :0]
        if mode == "style_only":
            t = t[:, :0]
        return np.concatenate([t, s], axis=1)
    dim = train_vectors.shape[1] if mode != "style_only" else 0
    return transform(train_vectors, train_style), transform(eval_vectors, eval_style), dim, text_scaler, style_scaler


class FusionHead(nn.Module):
    def __init__(self, text_dim, style_dim, width=64, dropout=0.2):
        super().__init__()
        self.text_dim, self.style_dim = text_dim, style_dim
        self.text = nn.Sequential(nn.Linear(text_dim, 64), nn.GELU(), nn.LayerNorm(64)) if text_dim else None
        self.style = nn.Sequential(nn.Linear(style_dim, 32), nn.GELU(), nn.LayerNorm(32)) if style_dim else None
        size = (64 if text_dim else 0) + (32 if style_dim else 0)
        self.classifier = nn.Sequential(nn.Linear(size, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 2))

    def forward(self, inputs):
        parts = []
        if self.text is not None:
            parts.append(self.text(inputs[:, :self.text_dim]))
        if self.style is not None:
            parts.append(self.style(inputs[:, self.text_dim:]))
        return self.classifier(torch.cat(parts, dim=1))


def head_probabilities(model, inputs):
    model.eval()
    with torch.inference_mode():
        return model(torch.as_tensor(inputs, dtype=torch.float32)).double().softmax(-1)[:, 1].numpy()


def train_head(inputs, labels, eval_inputs, config, seed, epochs, checkpoints=()):
    """The cached encoder is frozen; only this fusion MLP learns here."""
    torch.manual_seed(seed)
    model = FusionHead(config["text_dim"], inputs.shape[1] - config["text_dim"],
                       config["width"], config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    x = torch.as_tensor(inputs, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    predictions, history = {}, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for indices in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite fusion-head loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(indices)
        history.append({"epoch": epoch, "training_loss": total / len(x)})
        if epoch in checkpoints:
            predictions[epoch] = head_probabilities(model, eval_inputs)
    return model, predictions, history
