"""Compute GPT-2 "surprise" features for every passage.

Run once from submissions/group15/:
    python experiments/gpt2_features.py --quick   # 32 passages, just to check it works
    python experiments/gpt2_features.py           # all 16,000 passages -> data/gpt2_features.npz

Each split is saved as soon as it is done (data/gpt2_feats_<split>.npy), so if the run
stops halfway, running it again skips the splits that are already finished.
"""

import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from common import load_data

MODEL_NAME = "gpt2"        # smallest GPT-2 (124M parameters), public weights on Hugging Face
MAX_TOKENS = 256           # every batch is padded/cut to exactly this length (~190 words)
BATCH_SIZE = 8
OUT_FILE = "data/gpt2_features.npz"
FEATURE_NAMES = ["mean", "std", "median", "max", "frac_low"]


def pick_device():
    if torch.backends.mps.is_available():   # Apple chip
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def surprise_features(texts, tokenizer, model, device):
    """For each passage: how surprised GPT-2 is by each token, summarized into 5 numbers."""
    rows = []
    texts = list(texts)
    t0 = time.time()
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        # same length for every batch -> the Apple chip does not slow down over time
        enc = tokenizer(batch, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=MAX_TOKENS).to(device)
        logits = model(**enc).logits                      # (batch, tokens, vocab)

        # surprise of token t = -log P(token t | tokens before it)
        targets = enc["input_ids"][:, 1:]
        surprise = F.cross_entropy(logits[:, :-1].transpose(1, 2).float(),
                                   targets, reduction="none")   # (batch, tokens-1)
        mask = enc["attention_mask"][:, 1:].bool()             # ignore padding

        for i in range(len(batch)):
            s = surprise[i][mask[i]].cpu().numpy()
            rows.append([s.mean(), s.std(), np.median(s), s.max(), (s < 1.0).mean()])

        done = start + len(batch)
        if (start // BATCH_SIZE) % 25 == 0 or done == len(texts):
            elapsed = time.time() - t0
            left = elapsed / done * (len(texts) - done)
            print(f"  {done}/{len(texts)}   elapsed {elapsed/60:.1f} min, ~{left/60:.1f} min left")
    return np.array(rows, dtype=np.float32)


if __name__ == "__main__":
    device = pick_device()
    print("device:", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token          # GPT-2 has no padding token by default
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()

    train, unlabeled, test = load_data()

    if "--quick" in sys.argv:
        feats = surprise_features(train["text"][:32], tokenizer, model, device)
        print(FEATURE_NAMES)
        print("human (label 0) average:", feats[train["label"][:32].values == 0].mean(axis=0).round(3))
        print("AI    (label 1) average:", feats[train["label"][:32].values == 1].mean(axis=0).round(3))
        sys.exit()

    out = {}
    for name, df in [("train", train), ("unlabeled", unlabeled), ("test", test)]:
        part_file = f"data/gpt2_feats_{name}.npy"
        if os.path.exists(part_file):
            print(f"--- {name}: already done, loading {part_file} ---")
            feats = np.load(part_file)
        else:
            print(f"--- {name}: {len(df)} passages ---")
            feats = surprise_features(df["text"], tokenizer, model, device)
            np.save(part_file, feats)
        out[f"{name}_ids"] = df["id"].values
        out[f"{name}_feats"] = feats

    np.savez(OUT_FILE, feature_names=np.array(FEATURE_NAMES), **out)
    print("saved", OUT_FILE)
