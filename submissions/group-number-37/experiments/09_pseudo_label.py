"""report q3: confidence based self-training with unlabeled.csv.

validation labels are never touched while picking pseudo labels. compares weights and thresholds, on
the fixed split and on the held out book and poetry domain.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc

from src.harness import Ctx, REPORTS
from src.models import fit_clean_lexical

c = Ctx()
y, texts, n = c.y, c.texts, c.n
big_rows = np.flatnonzero(c.dom[:n] == c.big)


def run(fit_rows, pool, frac, weight, rounds=1):
    """self-train on the most confident frac of the pool."""
    fit = fit_clean_lexical(list(texts[fit_rows]), y[fit_rows])
    for _ in range(rounds):
        s = fit(list(texts[pool]))
        k = int(len(pool) * frac / 2)
        o = np.argsort(s)
        lo, hi = o[:k], o[-k:]
        idx = np.r_[fit_rows, pool[lo], pool[hi]]
        lab = np.r_[y[fit_rows], np.zeros(k), np.ones(k)]
        w = np.r_[np.ones(len(fit_rows)), np.full(2 * k, weight)]
        fit = fit_clean_lexical(list(texts[idx]), lab, sample_weight=w)
    return fit


rows = []
base = fit_clean_lexical(list(texts[c.fit_i]), y[c.fit_i])
rows.append(dict(method="supervised only", val=auc(y[c.val_i], base(list(texts[c.val_i])))))
keep = c.fit_i[c.dom[c.fit_i] != c.big]
base_h = fit_clean_lexical(list(texts[keep]), y[keep])
rows[-1]["held_book_poetry"] = auc(y[big_rows], base_h(list(texts[big_rows])))
print(rows[-1], flush=True)
for pool_name, pool in (("unlabeled", c.un_i), ("unlabeled+test", np.r_[c.un_i, c.te_i])):
    for frac, weight, rounds in ((.3, .5, 1), (.6, .5, 1), (.3, 1., 1), (.6, .5, 2)):
        m = run(c.fit_i, pool, frac, weight, rounds)
        h = run(keep, pool, frac, weight, rounds)
        rows.append(dict(method=f"{pool_name} top {frac} w={weight} rounds={rounds}", val=auc(y[c.val_i], m(list(texts[c.val_i]))),
                         held_book_poetry=auc(y[big_rows], h(list(texts[big_rows])))))
        print(rows[-1], flush=True)
pd.DataFrame(rows).round(4).to_csv(REPORTS / "09_pseudo_label.csv", index=False)
