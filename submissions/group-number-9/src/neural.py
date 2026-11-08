"""Pretrained-model features: language-model surprisal and curvature, and MiniLM sentence embeddings.

Weights come from the Hugging Face Hub (gpt2, gpt2-medium, facebook/opt-125m,
sentence-transformers/all-MiniLM-L6-v2) and are used only as fixed feature extractors. Nothing here
sees labels.
"""
import os

# Fail with an out-of-memory error instead of pushing the whole machine into swap.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")  # must not exceed the high ratio

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def free_device_memory():
    """MPS keeps freed buffers cached unless told otherwise."""
    if DEVICE == "mps":
        torch.mps.empty_cache()


def _batches(texts, batch_size):
    """Yield (positions, texts), longest first, so padding is small."""
    order = np.argsort([-len(t) for t in texts])
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield idx, [texts[i] for i in idx]


def _surprisal_stats(n, r):
    """Summaries of per-token negative log-probability n and predicted rank r."""
    half = len(n) // 2
    return {
        "nll_mean": n.mean(), "nll_std": n.std(),
        "nll_q10": np.quantile(n, 0.1), "nll_q50": np.quantile(n, 0.5), "nll_q90": np.quantile(n, 0.9),
        "nll_first_half": n[:half].mean(), "nll_second_half": n[half:].mean(),
        "top1": (r == 0).mean(), "top10": (r < 10).mean(), "top100": (r < 100).mean(),
        "rare": (r >= 1000).mean(),
        "diff_std": np.diff(n).std() if len(n) > 1 else 0.0,
    }


def _curvature_stats(g, u, v):
    """Analytic conditional probability curvature, following Fast-DetectGPT (Bao et al., 2024).

    g is each token's log-probability minus its expectation under the model's own predictive
    distribution, u that expectation, and v its variance; this adjusts for how uncertain the model
    was at each position.
    """
    z = g / np.sqrt(v)
    return {
        "curv_total": g.sum() / np.sqrt(v.sum()), "curv_z_mean": z.mean(), "curv_z_std": z.std(),
        "curv_z_q10": np.quantile(z, 0.1), "curv_z_q90": np.quantile(z, 0.9),
        "curv_entropy_mean": -u.mean(), "curv_entropy_std": u.std(),
    }


@torch.no_grad()
def _per_token_stats(texts, model_name, stats, max_length, batch_size):
    """Run a language model over texts; stats maps per-token arrays to one row per document."""
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE).eval()
    rows = [None] * len(texts)
    for idx, chunk in _batches(texts, batch_size):
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
        logp = model(**enc).logits[:, :-1].log_softmax(-1)  # about 0.8 GB per batch at full length
        observed = logp.gather(-1, enc.input_ids[:, 1:].unsqueeze(-1))
        prob = logp.exp()
        expected = (prob * logp).sum(-1)
        variance = (prob * logp.square()).sum(-1) - expected.square()
        arrays = {
            "nll": (-observed.squeeze(-1)).cpu().numpy(),
            "rank": (logp > observed).sum(-1).cpu().numpy(),
            "gap": (observed.squeeze(-1) - expected).cpu().numpy(),
            "expected": expected.cpu().numpy(),
            "variance": variance.clamp_min(1e-6).cpu().numpy(),
        }
        mask = enc.attention_mask[:, 1:].bool().cpu().numpy()
        del logp, observed, prob, expected, variance, enc
        free_device_memory()
        for i, pos in enumerate(idx):
            rows[pos] = stats({k: v[i][mask[i]] for k, v in arrays.items()})
    return pd.DataFrame(rows)


def surprisal_features(texts, model_name="gpt2", max_length=512, batch_size=8):
    """Per-token surprisal summaries: AI text is often more predictable to a language model."""
    def stats(a):
        return {f"gpt2_{k}": v for k, v in _surprisal_stats(a["nll"], a["rank"]).items()}
    return _per_token_stats(texts, model_name, stats, max_length, batch_size)


def curvature_features(texts, model_name="gpt2", max_length=512, batch_size=8):
    def stats(a):
        return _curvature_stats(a["gap"], a["expected"], a["variance"])
    return _per_token_stats(texts, model_name, stats, max_length, batch_size)


def language_model_features(texts, model_name, prefix, max_length=512, batch_size=8):
    """Surprisal and curvature summaries from one model, with its name as a column prefix."""
    def stats(a):
        row = {**_surprisal_stats(a["nll"], a["rank"]), **_curvature_stats(a["gap"], a["expected"], a["variance"])}
        return {f"{prefix}_{k}": v for k, v in row.items()}
    return _per_token_stats(texts, model_name, stats, max_length, batch_size)


@torch.no_grad()
def sentence_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", max_length=256, batch_size=64):
    """Mean-pooled, L2-normalized token embeddings."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for idx, chunk in _batches(texts, batch_size):
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
        hidden = model(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        out[idx] = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
    return out
