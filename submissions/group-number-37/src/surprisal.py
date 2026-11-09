"""token predictability under a pretrained lm, independent of this dataset's formatting.

text is whitespace-flattened first so paragraph breaks cannot leak in. the discrepancy feature
follows the analytic criterion from fast-detectgpt: observed log prob against its expectation
under the model's own predictive distribution.
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAMES = ["lp_mean", "lp_std", "lp_q10", "lp_q50", "lp_q90", "top1", "top5", "top10", "top100", "entropy", "discrepancy",
         "lp_first", "lp_second", "lp_autocorr", "n_tokens"]


def flatten(t):
    return re.sub(r"\s+", " ", t).strip()


@torch.no_grad()
def surprisal_matrix(texts, model_name="gpt2", max_len=384, batch=8, device=None, log_every=50, checkpoint=None):
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name, token=False)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, token=False).to(device).eval()
    out = np.zeros((len(texts), len(NAMES)), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    first = 0
    if checkpoint and Path(checkpoint).exists():
        saved = np.load(checkpoint)
        out, first = saved["out"], int(saved["next"])
        print("resuming at", first, flush=True)
    for b, start in enumerate(range(first, len(texts), batch)):
        ids = order[start:start + batch]
        enc = tok([flatten(texts[i]) for i in ids], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        logp = torch.log_softmax(model(**enc).logits[:, :-1], -1)
        tgt, mask = enc.input_ids[:, 1:], enc.attention_mask[:, 1:].bool()
        lp = logp.gather(-1, tgt[..., None])[..., 0]
        p = logp.exp()
        ent = -(p * logp).sum(-1)
        var = (p * logp.pow(2)).sum(-1) - ent.pow(2)
        rank = (logp > lp[..., None]).sum(-1)
        lp, ent, var, rank, mask = [x.float().cpu().numpy() if x is not mask else x.cpu().numpy()
                                    for x in (lp, ent, var, rank, mask)]
        for row, i in enumerate(ids):
            m = mask[row]
            if m.sum() < 3:
                continue
            a, r = lp[row][m], rank[row][m]
            h = len(a) // 2
            out[i] = [a.mean(), a.std(), *np.quantile(a, [.1, .5, .9]), (r < 1).mean(), (r < 5).mean(), (r < 10).mean(),
                      (r < 100).mean(), ent[row][m].mean(),
                      (a.sum() + ent[row][m].sum()) / np.sqrt(var[row][m].sum() + 1e-6),
                      a[:h].mean(), a[h:].mean(), np.corrcoef(a[:-1], a[1:])[0, 1], len(a)]
        if b % log_every == 0:
            print(f"{start}/{len(texts)}", flush=True)
            torch.mps.empty_cache() if device == "mps" else None
            if checkpoint:
                np.savez(checkpoint, out=out, next=start + batch)
    return np.nan_to_num(out)
