"""Binoculars / cross-model features from a modern base+instruct pair (default Qwen3.5-2B).

Binoculars (Hans et al. 2024) compares how surprising a text is to an "observer" model
with how surprising the "performer" model *expects* it to be:
  bino = log-PPL_observer(x) / X-PPL(x),   X-PPL = mean_t  -sum_v p_perf(v|x<t) log p_obs(v|x<t)
LLM text is low on both and the ratio is low; human text is surprising relative to what
an LLM would predict. Here observer = base model, performer = instruct model (same
tokenizer), as in the paper. We also emit per-model perplexity/entropy/rank stats and a
cross-model Fast-DetectGPT score (sampling = performer, scoring = observer).
Frozen models only: no fine-tuning, no generation, no labels.

Usage (GPU with >= 24 GB and bf16, e.g. A6000/A100/L40S):
  python yuchen_bino_features.py --data data --out out/qwen35_2b --limit 16   # smoke
  python yuchen_bino_features.py --data data --out out/qwen35_2b
Writes {out}/bino_{split}.csv (id + features), resumable per split.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


def load_lm(name: str, device, dtype):
    import transformers as tf
    last = None
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            m = getattr(tf, cls).from_pretrained(name, dtype=dtype)
            print(f"loaded {name} via {cls} ({type(m).__name__})", flush=True)
            return m.to(device).eval()
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"{cls} failed for {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    raise last


def stats(l: np.ndarray, e: np.ndarray, r: np.ndarray, p: str) -> dict:
    return {f"{p}logp_mean": l.mean(), f"{p}logp_std": l.std(), f"{p}logp_p10": np.percentile(l, 10),
            f"{p}logp_p25": np.percentile(l, 25), f"{p}logp_median": np.median(l), f"{p}logp_min": l.min(),
            f"{p}ent_mean": e.mean(), f"{p}ent_std": e.std(), f"{p}rank1_share": (r == 1).mean(),
            f"{p}rank10_share": (r <= 10).mean(), f"{p}logrank_mean": np.log(r).mean()}


@torch.no_grad()
def score_batch(obs, perf, ids, mask) -> list[dict]:
    lo_all = obs(input_ids=ids, attention_mask=mask).logits[:, :-1]
    lp_all = perf(input_ids=ids, attention_mask=mask).logits[:, :-1]
    target, tmask = ids[:, 1:], mask[:, 1:].bool()
    out = []
    for i in range(ids.shape[0]):        # per row in fp32 to bound memory (vocab ~248k)
        m = tmask[i]
        n = int(m.sum())
        if n < 2:
            out.append({}); continue
        t = target[i][m]
        Lo = torch.log_softmax(lo_all[i][m].float(), -1)        # observer log-probs  [n, V]
        Lp = torch.log_softmax(lp_all[i][m].float(), -1)        # performer log-probs [n, V]
        Po, Pp = Lo.exp(), Lp.exp()
        lo = Lo.gather(-1, t[:, None]).squeeze(-1)              # log p_obs(x_t)
        lpf = Lp.gather(-1, t[:, None]).squeeze(-1)             # log p_perf(x_t)
        ent_o = -(Po * Lo).sum(-1); ent_p = -(Pp * Lp).sum(-1)
        xent = -(Pp * Lo).sum(-1)                                # cross-entropy perf -> obs
        mu_x = -xent                                             # E_perf[log p_obs]
        var_x = (Pp * Lo.pow(2)).sum(-1) - mu_x.pow(2)           # Var_perf[log p_obs]
        var_o = (Po * Lo.pow(2)).sum(-1) - (Po * Lo).sum(-1).pow(2)
        rank_o = (Lo > lo[:, None]).sum(-1) + 1
        rank_p = (Lp > lpf[:, None]).sum(-1) + 1
        f = lambda x: x.cpu().numpy().astype(np.float64)  # noqa: E731
        lo_, lpf_, eo, ep, xe, vx, vo = map(f, (lo, lpf, ent_o, ent_p, xent, var_x, var_o))
        d = {"q_ntok": n}
        d.update(stats(lo_, eo, f(rank_o), "qo_"))
        d.update(stats(lpf_, ep, f(rank_p), "qp_"))
        d.update({
            "q_bino": (-lo_.mean()) / max(xe.mean(), 1e-6),                      # Binoculars score
            "q_bino_perf": (-lpf_.mean()) / max(xe.mean(), 1e-6),
            "q_xent_mean": xe.mean(),
            "q_fastdetect_cross": (lo_.sum() + xe.sum()) / math.sqrt(max(vx.sum(), 1e-6)),
            "q_fastdetect_obs": (lo_.sum() + eo.sum()) / math.sqrt(max(vo.sum(), 1e-6)),
            "q_logp_gap": lpf_.mean() - lo_.mean(),                              # instruct vs base
            "q_logp_gap_std": (lpf_ - lo_).std(),
            "q_tok_bino_std": ((-lo_) / np.maximum(xe, 1e-6)).std(),
        })
        out.append(d)
    return out


def run_split(name, a, tok, obs, perf, device):
    path = a.out / (f"bino_{name}_smoke.csv" if a.limit else f"bino_{name}.csv")
    if path.exists() and not a.limit:
        print(f"{name}: exists, skipping"); return
    df = pd.read_csv(a.data / f"{name}.csv")
    if a.limit:
        df = df.head(a.limit)
    order = np.argsort(df["text"].str.len().to_numpy())
    rows, t0 = {}, time.time()
    for b in range(0, len(order), a.batch_size):
        idx = order[b:b + a.batch_size]
        texts = [tok.eos_token + t for t in df["text"].iloc[idx]]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=a.max_tokens)
        for j, f in zip(idx, score_batch(obs, perf, enc["input_ids"].to(device), enc["attention_mask"].to(device))):
            rows[int(j)] = f
        if (b // a.batch_size) % 25 == 0:
            print(f"{name}: {b + len(idx)}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    feats = pd.DataFrame([rows[i] for i in range(len(df))])
    feats.insert(0, "id", df["id"].to_numpy())
    bad = feats.drop(columns="id").replace([np.inf, -np.inf], np.nan).isna().any(axis=1).sum()
    feats.to_csv(path, index=False)
    print(f"{name}: wrote {path} {feats.shape}, rows with NaN/inf: {bad}, {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observer", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--performer", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    print(f"gpu {torch.cuda.get_device_name()}  dtype {dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.observer)
    tok2 = AutoTokenizer.from_pretrained(a.performer)
    assert tok.get_vocab() == tok2.get_vocab(), "observer/performer tokenizers differ"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # right padding keeps causal / recurrent layers exact
    obs, perf = load_lm(a.observer, device, dtype), load_lm(a.performer, device, dtype)
    for s in a.splits:
        run_split(s, a, tok, obs, perf, device)


if __name__ == "__main__":
    main()
