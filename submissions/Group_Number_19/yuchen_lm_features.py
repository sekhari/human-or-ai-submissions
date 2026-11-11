"""GPT-2 token-probability features (zero-shot detector signals) for every text.

For each text, a frozen causal LM (default: gpt2, 124M) scores every token. From the
per-token log-probability, predictive entropy and rank we compute:
  lm_logp_*        mean / std / percentiles of log p(token)        (perplexity family)
  lm_ent_mean      mean predictive entropy
  lm_bino          mean(-log p) / mean(entropy)  (single-model Binoculars-style ratio)
  lm_fastdetect    Fast-DetectGPT analytic score: (sum log p - sum E[log p]) / sqrt(sum Var[log p])
  lm_rank_*        share of tokens ranked 1 / top-10, mean log-rank
LLM text tends to have higher log p, lower entropy ratio and higher Fast-DetectGPT score,
independent of topic or genre. No labels are used; the model is not fine-tuned.

Usage (from the repository root; downloads gpt2 from Hugging Face on first run):
  python submissions/Group_Number_19/yuchen_lm_features.py --limit 20     # smoke test
  python submissions/Group_Number_19/yuchen_lm_features.py                # train, test, unlabeled
Writes results/yuchen/lm_features/lm_{split}.csv (id + features). Resumable per split.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def score_batch(model, enc, device) -> list[dict[str, float]]:
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    target = ids[:, 1:]
    tmask = mask[:, 1:].bool()
    logprobs = torch.log_softmax(logits, dim=-1)
    probs = logprobs.exp()
    lp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)                 # log p(token)
    mu = (probs * logprobs).sum(-1)                                           # E[log p] = -entropy
    var = (probs * logprobs.pow(2)).sum(-1) - mu.pow(2)                       # Var[log p]
    rank = (logprobs > lp.unsqueeze(-1)).sum(-1) + 1                          # 1 = argmax
    out = []
    for i in range(ids.shape[0]):
        m = tmask[i]
        n = int(m.sum())
        if n < 2:
            out.append({}); continue
        l, e, v, r = lp[i][m].cpu().numpy(), (-mu[i][m]).cpu().numpy(), var[i][m].cpu().numpy(), rank[i][m].cpu().numpy()
        out.append({
            "lm_ntok": n,
            "lm_logp_mean": l.mean(), "lm_logp_std": l.std(),
            "lm_logp_p10": np.percentile(l, 10), "lm_logp_p25": np.percentile(l, 25),
            "lm_logp_median": np.median(l), "lm_logp_min": l.min(),
            "lm_ent_mean": e.mean(), "lm_ent_std": e.std(),
            "lm_bino": (-l.mean()) / max(e.mean(), 1e-6),
            "lm_fastdetect": (l.sum() + e.sum()) / math.sqrt(max(v.sum(), 1e-6)),
            "lm_fastdetect_tok": ((l + e) / np.sqrt(np.maximum(v, 1e-6))).mean(),
            "lm_rank1_share": (r == 1).mean(), "lm_rank10_share": (r <= 10).mean(),
            "lm_logrank_mean": np.log(r).mean(),
        })
    return out


def run_split(name, data, out, tok, model, device, max_tokens, batch_size, limit):
    path = out / f"lm_{name}.csv"
    if path.exists() and not limit:
        print(f"{name}: exists, skipping ({path})"); return
    df = pd.read_csv(data / f"{name}.csv")
    if limit:
        df = df.head(limit)
    # sort by length so batches pad little; restore order at the end
    order = np.argsort(df["text"].str.len().to_numpy())
    rows: dict[int, dict] = {}
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        texts = [tok.bos_token + t for t in df["text"].iloc[idx]]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
        for j, f in zip(idx, score_batch(model, enc, device)):
            rows[int(j)] = f
        if (b // batch_size) % 20 == 0:
            done = b + len(idx)
            print(f"{name}: {done}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    feats = pd.DataFrame([rows[i] for i in range(len(df))])
    feats.insert(0, "id", df["id"].to_numpy())
    if limit:
        path = out / f"lm_{name}_smoke.csv"
    feats.to_csv(path, index=False)
    print(f"{name}: wrote {path} {feats.shape} in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "yuchen" / "lm_features")
    ap.add_argument("--splits", nargs="+", default=["test", "train", "unlabeled"])
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows (smoke test)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.model).to(device).eval()
    print(f"model {a.model} on {device}", flush=True)
    for s in a.splits:
        run_split(s, a.data, a.out, tok, model, device, a.max_tokens, a.batch_size, a.limit)


if __name__ == "__main__":
    main()
