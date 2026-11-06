"""fine-tune an encoder and save its scores for all 16000 rows.

--holdout fixed   train on the 8000 rows, report the 2000 row validation set
--holdout big     also drop the book and poetry domain from training and report on it
--holdout none    train on all 10000 labeled rows for the final model
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc

from src.attacks import apply
from src.data import OUT, fixed_split, load_all
from src.domains import infer_domains
from src.encoder import predict, train_encoder

p = argparse.ArgumentParser()
p.add_argument("--name", default="distilroberta-base")
p.add_argument("--holdout", default="fixed", choices=["fixed", "big", "none"])
p.add_argument("--aug", type=float, default=0.)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--dropout", type=float, default=0.)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--max-len", type=int, default=256)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--seed", type=int, default=3737)
a = p.parse_args()

tr, un, te = load_all()
texts = np.array(pd.concat([tr.text, un.text, te.text], ignore_index=True), dtype=object)
y, n = tr.label.values, len(tr)
dom, _ = infer_domains(texts)
big = int(np.bincount(dom[n + len(un):]).argmax())
fit_i, val_i = fixed_split(tr)
if a.holdout == "none":
    use, report = np.arange(n), {}
elif a.holdout == "fixed":
    use, report = fit_i, {"val": val_i}
else:
    use = fit_i[dom[fit_i] != big]
    report = {"val_seen": val_i[dom[val_i] != big], "book_poetry": np.flatnonzero(dom[:n] == big)}

t0 = time.time()
model, tok = train_encoder(list(texts[use]), y[use], name=a.name, epochs=a.epochs, lr=a.lr, batch=a.batch, max_len=a.max_len,
                           aug_rate=a.aug, seed=a.seed, dropout=a.dropout)
train_s = round(time.time() - t0)
scores = predict(model, tok, list(texts), max_len=a.max_len)
tag = f"{a.name.split('/')[-1]}_{a.holdout}_aug{a.aug}_s{a.seed}_ep{a.epochs}_drop{a.dropout}"
OUT.mkdir(exist_ok=True)
np.save(OUT / f"enc_{tag}.npy", scores)
res = dict(tag=tag, rows=len(use), train_seconds=train_s, train_auc=auc(y[use], scores[use]))
for k, idx in report.items():
    res[k] = auc(y[idx], scores[idx])
if "val" in report:
    for atk in ("flatten_ws", "homoglyph", "zero_width", "drop_articles", "typos", "char_noise", "synonyms", "case_noise"):
        res[f"val_{atk}"] = auc(y[val_i], predict(model, tok, apply(atk, list(texts[val_i])), max_len=a.max_len))
print(json.dumps(res, indent=1), flush=True)
rep = Path(__file__).resolve().parents[1] / "reports"
with open(rep / "06_encoder.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
