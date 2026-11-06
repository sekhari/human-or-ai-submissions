"""encoder with each inferred domain held out, worst domains first. resumable, saves after every domain.

prints the three way blend on all finished domains using the lodo reference and surprisal scores from experiment 14.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as auc

from src.data import OUT
from src.encoder import predict, train_encoder
from src.harness import Ctx, REPORTS

c = Ctx()
y, texts = c.y, c.texts
ls = dict(np.load(OUT / "lodo_scores.npz"))
path = OUT / "lodo_encoder.npz"
store = dict(np.load(path)) if path.exists() else {}
r = lambda x: rankdata(x) / len(x)
WS = ((1, 0, 0), (.7, .3, 0), (.8, .2, 0), (.8, 0, .2), (.8, .1, .1), (.7, .2, .1), (.7, .1, .2), (.6, .2, .2), (.5, .3, .2), (.5, .25, .25), (.4, .3, .3))


def report():
    done = sorted(int(k[4:]) for k in store)
    rows = []
    for ws in WS:
        per = {d: auc(y[ls[f"idx_{d}"]], ws[0] * r(ls[f"ref_{d}"]) + ws[1] * r(ls[f"sur_{d}"]) + ws[2] * r(store[f"enc_{d}"])) for d in done}
        rows.append(dict(reference=ws[0], surprisal=ws[1], encoder=ws[2], mean=np.mean(list(per.values())), worst=min(per.values()), **{f"d{d}": p for d, p in per.items()}))
    enc = {d: auc(y[ls[f"idx_{d}"]], store[f"enc_{d}"]) for d in done}
    df = pd.DataFrame(rows).round(4)
    df.to_csv(REPORTS / "19_encoder_lodo_blend.csv", index=False)
    print("encoder alone", {k: round(v, 4) for k, v in enc.items()}, "mean", round(np.mean(list(enc.values())), 4), flush=True)
    print(df[["reference", "surprisal", "encoder", "mean", "worst"]].to_string(index=False), flush=True)


for d in (9, 4, 0, 3, 8, 1, 7, 6, 2, 5):
    if f"enc_{d}" in store:
        continue
    t0 = time.time()
    held = ls[f"idx_{d}"]
    keep = np.setdiff1d(np.arange(c.n), held)
    m, tok = train_encoder(list(texts[keep]), y[keep], log_every=250)
    store[f"enc_{d}"] = predict(m, tok, list(texts[held]))
    np.savez(path, **store)
    print(f"domain {d} done in {round(time.time() - t0)}s", flush=True)
    report()
    del m
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
