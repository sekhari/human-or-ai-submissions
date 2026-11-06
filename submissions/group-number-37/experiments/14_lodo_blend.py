"""does adding formatting-free models help on domains the blend has never seen?

for each inferred domain: fit every base model without it, score it, then rank average inside that domain. each weight
is judged across all domains, so this measures transfer, not fit to one validation set.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as auc

from src.baseline import fit_lexical, fit_style
from src.data import OUT
from src.harness import Ctx, REPORTS
from src.models import fit_clean_lexical, fit_gbm
from src.validate import lodo_splits

c = Ctx()
texts, y, n = c.texts, c.y, c.n
S = c.surprisal()
cache = OUT / "lodo_scores.npz"
store = dict(np.load(cache)) if cache.exists() else {}
for d, a, b in lodo_splits(c.dom[:n], y):
    if f"ref_{d}" in store:
        continue
    t = list(texts[a])
    lx, st = fit_lexical(t, y[a]), fit_style(t, y[a])
    new = list(texts[b])
    store[f"lex_{d}"], store[f"style_{d}"] = lx(new), st(new)
    store[f"ref_{d}"] = .35 * store[f"lex_{d}"] + .65 * store[f"style_{d}"]
    store[f"clean_{d}"] = fit_clean_lexical(t, y[a])(new)
    store[f"sur_{d}"] = fit_gbm(S[a], y[a])(S[b])
    store[f"idx_{d}"] = b
    np.savez(cache, **store)
    print("domain", d, "done", flush=True)

domains = sorted(int(k[4:]) for k in store if k.startswith("idx_"))
r = lambda x: rankdata(x) / len(x)
rows = []
for extra in ("sur", "clean"):
    for w in (0, .05, .1, .15, .2, .3, .4, .5, .7, 1.):
        per = [auc(y[store[f"idx_{d}"]], (1 - w) * r(store[f"ref_{d}"]) + w * r(store[f"{extra}_{d}"])) for d in domains]
        rows.append(dict(extra=extra, weight=w, mean=np.mean(per), worst=min(per), **{f"d{d}": p for d, p in zip(domains, per)}))
out = pd.DataFrame(rows).round(4)
out.to_csv(REPORTS / "14_lodo_blend.csv", index=False)
print(out.to_string(index=False))
