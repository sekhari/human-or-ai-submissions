"""why do confident errors land only on human rows, and why does text noise raise auc?

hypothesis: dropout during training teaches clean text means ai. with dropout off at prediction everything looks
clean, so human rows flip to ai and scores pile up. checks train mode vs eval mode on one model, then varies dropout
and label smoothing. uses 1600 fit rows and 500 validation rows, so compare variants with each other only.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score as auc
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from src.attacks import apply
from src.data import fixed_split, load_all
from src.harness import REPORTS
from src.normalize import clean

NAME, DEVICE = "distilroberta-base", "mps" if torch.backends.mps.is_available() else "cpu"
tr, _, _ = load_all()
f, v = fixed_split(tr)
y, tx = tr.label.values, tr.text.values
a, b = f[:1600], v[:500]
tok = AutoTokenizer.from_pretrained(NAME)
clean_fit = [clean(t) for t in tx[a]]


def fit(dropout, smooth, epochs=2, seed=3737):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    cfg = AutoConfig.from_pretrained(NAME, num_labels=2, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
                                     classifier_dropout=dropout, attn_implementation="eager")
    m = AutoModelForSequenceClassification.from_pretrained(NAME, config=cfg, use_safetensors=True).to(DEVICE)
    steps = epochs * (len(a) // 16)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=.01)
    sch = get_cosine_schedule_with_warmup(opt, int(.06 * steps), steps)
    rng, losses = np.random.default_rng(seed), []
    for _ in range(epochs):
        m.train()
        order = rng.permutation(len(a))
        for i in range(0, len(a) - 15, 16):
            ids = order[i:i + 16]
            e = tok([clean_fit[j] for j in ids], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            yb = torch.tensor(y[a][ids], dtype=torch.float32, device=DEVICE)
            lg = m(**e).logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lg[:, 1] - lg[:, 0], yb * (1 - smooth) + smooth / 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step(); sch.step(); opt.zero_grad()
            losses.append(loss.item())
    return m, float(np.mean(losses[-50:]))


@torch.no_grad()
def score(m, texts, train_mode=False):
    m.train(train_mode)
    out = np.zeros(len(texts))
    for i in range(0, len(texts), 32):
        e = tok([clean(t) for t in texts[i:i + 32]], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        lg = m(**e).logits.float().cpu().numpy()
        out[i:i + 32] = lg[:, 1] - lg[:, 0]
    return out


val, yv = list(tx[b]), y[b]
fit_sub, yf = list(tx[a[:500]]), y[a[:500]]
rows = []
for name, d, s in (("default dropout .1 smooth .05", .1, .05), ("dropout .1 no smoothing", .1, 0.), ("no dropout smooth .05", 0., .05),
                   ("no dropout no smoothing", 0., 0.), ("dropout .03 no smoothing", .03, 0.)):
    m, last = fit(d, s)
    ev, fe = score(m, val), score(m, fit_sub)
    row = dict(model=name, last_loss=last, fit_auc=auc(yf, fe), val_auc=auc(yv, ev), val_human_above0=(ev[yv == 0] > 0).mean(),
               val_ai_below0=(ev[yv == 1] < 0).mean(), val_typos=auc(yv, score(m, apply("typos", val))),
               val_case_noise=auc(yv, score(m, apply("case_noise", val))))
    if d > 0:
        tm = np.mean([score(m, val, train_mode=True) for _ in range(4)], 0)
        row |= dict(val_train_mode_auc=auc(yv, tm), val_train_mode_human_above0=(tm[yv == 0] > 0).mean())
    rows.append(row)
    print({k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).round(4).to_csv(REPORTS / "13_encoder_dropout.csv", index=False)
    del m
    torch.mps.empty_cache() if DEVICE == "mps" else None
