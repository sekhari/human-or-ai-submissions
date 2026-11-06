"""fine-tune a small pretrained encoder on cleaned text, with optional attack augmentation."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from .attacks import augment_random
from .normalize import clean

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _collate(tok, max_len):
    def f(batch):
        text = [b[0] for b in batch]
        enc = tok(text, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        if len(batch[0]) > 1:
            enc["labels"] = torch.tensor([b[1] for b in batch], dtype=torch.float32)
            enc["w"] = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        return enc
    return f


@torch.no_grad()
def predict(model, tok, texts, max_len=256, batch=48):
    model.eval()
    order = np.argsort([len(t) for t in texts])[::-1]
    cleaned_text = [clean(t) for t in texts]
    out = np.zeros(len(texts))
    for i in range(0, len(texts), batch):
        ids = order[i:i + batch]
        enc = tok([cleaned_text[j] for j in ids], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
        logit = model(**enc).logits.float().cpu().numpy()
        out[ids] = logit[:, 1] - logit[:, 0]
    return out


def train_encoder(texts, y, name="distilroberta-base", epochs=2, lr=2e-5, batch=16, max_len=256, aug_rate=0., weights=None,
                  seed=3737, log_every=100, smoothing=0.05, dropout=0.):
    _seed(seed)
    tok = AutoTokenizer.from_pretrained(name)
    # dropout off: with it on the model learns clean text means ai, see reports/FINDINGS.md section 8
    cfg = AutoConfig.from_pretrained(name, num_labels=2, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
                                     classifier_dropout=dropout, attn_implementation="eager")
    model = AutoModelForSequenceClassification.from_pretrained(name, config=cfg, use_safetensors=True).to(DEVICE)
    w = np.ones(len(texts)) if weights is None else np.asarray(weights, dtype=float)
    steps = epochs * (len(texts) // batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    sched = get_cosine_schedule_with_warmup(opt, int(.06 * steps), steps)
    for ep in range(epochs):
        t = list(texts)
        if aug_rate:
            r = random.Random(seed + ep)
            pick = [i for i in range(len(t)) if r.random() < aug_rate]
            for i, a in zip(pick, augment_random([t[i] for i in pick], seed=seed + ep)):
                t[i] = a
        data = [(clean(a), float(b), float(c)) for a, b, c in zip(t, y, w)]
        loader = DataLoader(data, batch_size=batch, shuffle=True, collate_fn=_collate(tok, max_len),
                            generator=torch.Generator().manual_seed(seed + ep))
        model.train()
        for i, enc in enumerate(loader):
            yb, wb = enc.pop("labels").to(DEVICE), enc.pop("w").to(DEVICE)
            enc = enc.to(DEVICE)
            logit = model(**enc).logits
            diff = logit[:, 1] - logit[:, 0]
            target = yb * (1 - smoothing) + smoothing / 2
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(diff, target, reduction="none") * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); sched.step(); opt.zero_grad()
            if i % log_every == 0:
                print(f"ep {ep} step {i}/{len(loader)} loss {loss.item():.4f}", flush=True)
    return model, tok
