"""do models that ignore formatting add anything to the reference?

fits each base model on the 8000 rows, scores validation and test, then:
  - blends by rank on validation, with weights picked by cv on validation folds
  - measures agreement on test per domain. no labels: low agreement shows where the models rely on different things.
"""
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold

from src.baseline import fit_lexical, fit_style
from src.data import OUT
from src.harness import Ctx, REPORTS
from src.models import fit_clean_lexical, fit_gbm

c = Ctx()
texts, y, f, v, te = c.texts, c.y, c.fit_i, c.val_i, c.te_i
use = np.r_[v, te]
S = c.surprisal()
SZ = np.c_[S, c.domain_z(S)]
scores = {}


def cached(name, fn):
    p = OUT / f"base_{name}.npy"
    if p.exists():
        return np.load(p)
    out = np.full(len(texts), np.nan)
    out[use] = fn()
    np.save(p, out)
    return out


lx = None
scores["style_raw"] = cached("style_raw", lambda: fit_style(list(texts[f]), y[f])(list(texts[use])))
scores["lex_raw"] = cached("lex_raw", lambda: fit_lexical(list(texts[f]), y[f])(list(texts[use])))
scores["clean_lr"] = cached("clean_lr", lambda: fit_clean_lexical(list(texts[f]), y[f])(list(texts[use])))
scores["surprisal"] = cached("surprisal", lambda: fit_gbm(S[f], y[f])(S[use]))
scores["surprisal_z"] = cached("surprisal_z", lambda: fit_gbm(SZ[f], y[f])(SZ[use]))
for p in sorted(OUT.glob("enc_*_fixed_*.npy")):
    scores[p.stem[4:]] = np.load(p)
scores["reference"] = .35 * scores["lex_raw"] + .65 * scores["style_raw"]


def r(x):
    return rankdata(x) / len(x)


rows = [dict(model=k, val=auc(y[v], s[v])) for k, s in scores.items()]
print(pd.DataFrame(rows).round(4).sort_values("val", ascending=False).to_string(index=False), flush=True)
pd.DataFrame(rows).round(4).to_csv(REPORTS / "12_base_val.csv", index=False)

extras = [k for k in scores if k not in ("reference", "style_raw", "lex_raw")]
folds = list(StratifiedKFold(5, shuffle=True, random_state=3737).split(v, y[v]))
out = []
for k in extras:
    for w in (.05, .1, .2, .3, .5, .7):
        s = (1 - w) * r(scores["reference"][v]) + w * r(scores[k][v])
        cv = np.mean([auc(y[v][b], s[b]) for _, b in folds])
        out.append(dict(extra=k, weight=w, val=auc(y[v], s), cv_mean=cv))
out = pd.DataFrame(out).round(5)
out.to_csv(REPORTS / "12_blends.csv", index=False)
print(out.sort_values("val", ascending=False).head(15).to_string(index=False), flush=True)

agree = []
for k in extras:
    row = dict(extra=k, all_test=spearmanr(scores["reference"][te], scores[k][te])[0])
    for d in range(10):
        m = te[c.dom[te] == d]
        row[f"d{d}"] = spearmanr(scores["reference"][m], scores[k][m])[0]
    agree.append(row)
agree = pd.DataFrame(agree).round(3)
agree.to_csv(REPORTS / "12_test_agreement.csv", index=False)
print("big domain is", c.big)
print(agree.to_string(index=False))
