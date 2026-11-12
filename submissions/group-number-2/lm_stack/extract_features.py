"""Step 3 - LM-based features for every passage in train / unlabeled / test.

Loads Qwen2.5-<size> (observer / "base") and Qwen2.5-<size>-Instruct (performer / "instr"),
which share one tokenizer. Each passage is normalize()d, truncated to 512 tokens and scored
by both models. Per passage (padding ignored):

  base_lp_mean, base_lp_std, instr_lp_mean, instr_lp_std   token log-prob of the actual token
  base_logrank_mean, instr_logrank_mean                    log(1-based rank of actual token)
  base_entropy_mean, instr_entropy_mean                    entropy of predictive distribution
  bino_ppl, bino_xppl, binoculars                          Binoculars (ahans30/Binoculars)
  n_tokens                                                 tokens after truncation

Token t is scored from the logits at position t-1; the first token is never scored.
Binoculars follows the reference implementation exactly:
  bino_ppl  = mean CE of performer logits vs. next token          (shifted, as reference perplexity())
  bino_xppl = mean CE between softmax(observer) and performer logits
              over all non-pad positions, unshifted               (as reference entropy())
  binoculars = bino_ppl / bino_xppl                               (lower = more AI-like)

Output: <out>/<split>_qwen<size>.parquet, built from per-chunk files in
<out>/parts/qwen<size>/<split>/ so an interrupted run resumes where it stopped.

Colab:
  !git clone <repo> && %cd <repo>/submissions/group-number-2/lm_stack
  !pip install -r requirements.txt
  # put train.csv / unlabeled.csv / test.csv in data/
  !python extract_features.py --model-size 0.5B --out features
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from normalize import normalize

SEED = 42
MAX_TOKENS = 512
HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ("train", "unlabeled", "test")
FEATURE_COLS = [
    "n_tokens",
    "base_lp_mean", "base_lp_std", "instr_lp_mean", "instr_lp_std",
    "base_logrank_mean", "instr_logrank_mean",
    "base_entropy_mean", "instr_entropy_mean",
    "bino_ppl", "bino_xppl", "binoculars",
]


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_models(size, device, dtype="float32"):
    """Return (tokenizer, base_model, instr_model) in the given dtype (name from DTYPES)."""
    base_name, instr_name = f"Qwen/Qwen2.5-{size}", f"Qwen/Qwen2.5-{size}-Instruct"
    tok = AutoTokenizer.from_pretrained(base_name)
    tok_instr = AutoTokenizer.from_pretrained(instr_name)
    if tok.get_vocab() != tok_instr.get_vocab():  # as reference assert_tokenizer_consistency
        raise ValueError(f"{base_name} and {instr_name} tokenizers differ")
    tok.padding_side = "right"  # keeps position ids of real tokens identical to the unpadded run
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    models = [
        AutoModelForCausalLM.from_pretrained(n, dtype=DTYPES[dtype]).to(device).eval()
        for n in (base_name, instr_name)
    ]
    if models[0].config.vocab_size != models[1].config.vocab_size:
        raise ValueError("observer / performer vocab sizes differ")
    return tok, models[0], models[1]


@torch.inference_mode()
def forward_logits(texts, tok, base, instr, max_tokens=MAX_TOKENS):
    """Tokenize (right-padded, truncated) and run both models. Returns (enc, base_logits, instr_logits)."""
    device = base.device
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_tokens, return_token_type_ids=False).to(device)
    base_logits = base(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
    instr_logits = instr(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
    return enc, base_logits, instr_logits


def token_level(logits, ids):
    """Per-token stats for ONE unpadded sequence.

    logits: (L, V), ids: (L,). Position t-1 predicts token t, so the first token is never
    scored and the last position's logits are unused. Returns tensors of length L-1.
    """
    logp = torch.log_softmax(logits[:-1].float(), dim=-1)            # (L-1, V)
    target = ids[1:]
    lp = logp.gather(1, target[:, None]).squeeze(1)                    # (L-1,)
    rank = (logp > lp[:, None]).sum(dim=1) + 1                         # 1-based, ties favour target
    entropy = -(logp.exp() * logp).sum(dim=1)
    return {"lp": lp, "logrank": torch.log(rank.float()), "entropy": entropy}


def binoculars_parts(observer_logits, performer_logits, ids):
    """Reference Binoculars for ONE unpadded sequence. Returns (ppl, x_ppl, score)."""
    # perplexity(): shifted CE of performer logits vs. actual next token
    perf_logp_shift = torch.log_softmax(performer_logits[:-1].float(), dim=-1)
    ppl = -perf_logp_shift.gather(1, ids[1:, None]).squeeze(1).mean()
    # entropy(): CE(target=softmax(observer), input=performer logits), all positions, unshifted
    p = torch.softmax(observer_logits.float(), dim=-1)
    q_logp = torch.log_softmax(performer_logits.float(), dim=-1)
    x_ppl = -(p * q_logp).sum(dim=1).mean()
    return ppl.item(), x_ppl.item(), (ppl / x_ppl).item()


def features_from_logits(enc, base_logits, instr_logits):
    out = []
    for i in range(enc.input_ids.shape[0]):
        n = int(enc.attention_mask[i].sum())
        assert bool(enc.attention_mask[i, :n].all()), "expected right padding"
        ids = enc.input_ids[i, :n]
        bl, il = base_logits[i, :n], instr_logits[i, :n]
        row = {"n_tokens": n}
        if n < 2:  # nothing to score; keep finite and let n_tokens flag it
            row.update({c: 0.0 for c in FEATURE_COLS if c != "n_tokens"})
            out.append(row)
            continue
        for prefix, lg in (("base", bl), ("instr", il)):
            t = token_level(lg, ids)
            row[f"{prefix}_lp_mean"] = t["lp"].mean().item()
            row[f"{prefix}_lp_std"] = t["lp"].std(unbiased=False).item()
            row[f"{prefix}_logrank_mean"] = t["logrank"].mean().item()
            row[f"{prefix}_entropy_mean"] = t["entropy"].mean().item()
        row["bino_ppl"], row["bino_xppl"], row["binoculars"] = binoculars_parts(bl, il, ids)
        out.append(row)
    return out


def featurize(texts, tok, base, instr, batch_size, max_tokens=MAX_TOKENS):
    """Features for a list of (already normalized) texts, in input order.

    Texts are batched longest-first to minimise padding; results are restored to input order.
    """
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    rows = [None] * len(texts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc, bl, il = forward_logits([texts[i] for i in idx], tok, base, instr, max_tokens)
        for i, r in zip(idx, features_from_logits(enc, bl, il)):
            rows[i] = r
        del enc, bl, il
    return rows


def _atomic_parquet(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def check_meta(part_dir, args, n_rows):
    """Record the settings that change feature values next to the chunk files. On resume,
    refuse to continue if any differ, so chunks with different cutoffs / dtypes never mix."""
    meta = {"model_size": args.model_size, "max_length": args.max_length,
            "dtype": args.dtype, "chunk_size": args.chunk_size, "n_rows": n_rows}
    path = os.path.join(part_dir, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != meta:
            raise RuntimeError(f"{part_dir} was started with {old}, now {meta}. "
                               "Use a different --out or delete that folder.")
    else:
        with open(path, "w") as f:
            json.dump(meta, f)


def extract_split(split, args, tok, base, instr):
    df = pd.read_csv(os.path.join(args.data_dir, f"{split}.csv"))
    if args.limit:
        df = df.head(args.limit)
    tag = f"qwen{args.model_size}"
    part_dir = os.path.join(args.out, "parts", tag, split)
    os.makedirs(part_dir, exist_ok=True)
    check_meta(part_dir, args, len(df))
    n_chunks = (len(df) + args.chunk_size - 1) // args.chunk_size
    t0 = time.time()
    for c in range(n_chunks):
        chunk = df.iloc[c * args.chunk_size:(c + 1) * args.chunk_size]
        path = os.path.join(part_dir, f"chunk_{c * args.chunk_size:06d}.parquet")
        if os.path.exists(path):
            done = pd.read_parquet(path, columns=["id"])
            if done["id"].tolist() != chunk["id"].tolist():
                raise RuntimeError(f"{path} does not match current data/chunk size; delete {part_dir}")
            continue
        texts = [normalize(t) for t in chunk["text"].astype(str)]
        feats = pd.DataFrame(
            featurize(texts, tok, base, instr, args.batch_size, args.max_length), columns=FEATURE_COLS)
        feats.insert(0, "id", chunk["id"].to_numpy())
        _atomic_parquet(feats, path)
        print(f"[{tag} {split}] chunk {c + 1}/{n_chunks} done ({time.time() - t0:.0f}s)", flush=True)

    parts = [pd.read_parquet(os.path.join(part_dir, f"chunk_{c * args.chunk_size:06d}.parquet"))
             for c in range(n_chunks)]
    full = pd.concat(parts, ignore_index=True)
    assert full["id"].tolist() == df["id"].tolist(), "id order mismatch"
    assert np.isfinite(full[FEATURE_COLS].to_numpy(dtype=float)).all(), "non-finite features"
    full["max_length"] = args.max_length  # recorded in the output; not a model feature
    out_path = os.path.join(args.out, f"{split}_{tag}.parquet")
    _atomic_parquet(full, out_path)
    print(f"[{tag} {split}] wrote {out_path} ({len(full)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-size", choices=["0.5B", "1.5B"], required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "features"))
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
                    help="any of train unlabeled test, processed in the order given")
    ap.add_argument("--dtype", choices=list(DTYPES), default="float32")
    ap.add_argument("--max-length", type=int, default=MAX_TOKENS, help="token cutoff per passage")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=256, help="rows per resumable part file")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--limit", type=int, default=None, help="only first N rows per split (smoke test)")
    args = ap.parse_args()

    set_seeds()
    device = pick_device(args.device)
    print(f"device={device} model-size={args.model_size} dtype={args.dtype} "
          f"batch={args.batch_size} max-length={args.max_length} splits={args.splits}", flush=True)
    tok, base, instr = load_models(args.model_size, device, args.dtype)
    for split in args.splits:
        extract_split(split, args, tok, base, instr)


if __name__ == "__main__":
    main()
