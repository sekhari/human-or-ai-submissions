"""Frozen GPT-2 statistics, matching origin/main's src.neural in one forward pass.

Vocabulary reductions run in token chunks to bound peak memory. No weights are
trained and no labels, unlabeled competition data, or test distribution are used.
"""
import hashlib

import numpy as np
import pandas as pd
import torch


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def summarize(n, r, g, u, v):
    if len(n) == 0:
        raise ValueError("GPT-2 needs at least two tokens per passage")
    half = max(1, len(n) // 2)
    z = g / np.sqrt(v)
    return {
        "gpt2_nll_mean": n.mean(), "gpt2_nll_std": n.std(),
        "gpt2_nll_q10": np.quantile(n, .1), "gpt2_nll_q50": np.quantile(n, .5),
        "gpt2_nll_q90": np.quantile(n, .9),
        "gpt2_nll_first_half": n[:half].mean(),
        "gpt2_nll_second_half": n[half:].mean() if len(n) > half else n.mean(),
        "gpt2_top1": (r == 0).mean(), "gpt2_top10": (r < 10).mean(),
        "gpt2_top100": (r < 100).mean(), "gpt2_rare": (r >= 1000).mean(),
        "gpt2_diff_std": np.diff(n).std() if len(n) > 1 else 0.,
        "curv_total": g.sum() / np.sqrt(v.sum()),
        "curv_z_mean": z.mean(), "curv_z_std": z.std(),
        "curv_z_q10": np.quantile(z, .1), "curv_z_q90": np.quantile(z, .9),
        "curv_entropy_mean": -u.mean(), "curv_entropy_std": u.std(),
    }


@torch.inference_mode()
def from_logits(logits, input_ids, attention_mask, token_chunk=32):
    """Exact next-token moments/ranks; exclude padding and the unpredicted first token."""
    parts = [[] for _ in range(6)]
    for start in range(0, logits.shape[1] - 1, token_chunk):
        stop = min(start + token_chunk, logits.shape[1] - 1)
        logp = logits[:, start:stop].float().log_softmax(-1)
        observed = logp.gather(-1, input_ids[:, start + 1:stop + 1, None])
        rank = (logp > observed).sum(-1)
        prob = logp.exp()
        expected = (prob * logp).sum(-1)
        variance = ((prob * logp.square()).sum(-1) - expected.square()).clamp_min(1e-6)
        values = [-observed.squeeze(-1), rank, observed.squeeze(-1) - expected,
                  expected, variance, attention_mask[:, start + 1:stop + 1]]
        for dst, value in zip(parts, values):
            dst.append(value.cpu().numpy())
        del logp, observed, rank, prob, expected, variance, values
    arrays = [np.concatenate(p, axis=1) for p in parts]
    rows = []
    for n, r, g, u, v, mask in zip(*arrays):
        m = mask.astype(bool)
        rows.append(summarize(n[m], r[m], g[m], u[m], v[m]))
    result = pd.DataFrame(rows)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite GPT-2 features")
    return result


@torch.inference_mode()
def extract_batch(texts, tokenizer, model, device, max_length=512):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt").to(device)
    output = model(**enc, use_cache=False)
    result = from_logits(output.logits, enc.input_ids, enc.attention_mask)
    del output, enc
    return result
