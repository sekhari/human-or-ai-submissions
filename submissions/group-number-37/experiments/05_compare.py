"""compare cheap families on the fixed split and leave one domain out.

prints a table and writes reports/05_compare.csv. the reference recipe sees raw text, so it can use
the double newline signal. everything else sees cleaned text or surprisal only.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc

from src.baseline import fit_lexical, fit_style
from src.data import OUT, fixed_split, load_all
from src.domains import infer_domains
from src.models import fit_clean_lexical, fit_gbm
from src.validate import lodo_splits

REPORT = Path(__file__).resolve().parents[1] / "reports"
tr, un, te = load_all()
texts = np.array(pd.concat([tr.text, un.text, te.text], ignore_index=True), dtype=object)
y = tr.label.values
n = len(tr)
dom, tops = infer_domains(texts)
S = np.load(OUT / "surprisal_gpt2.npy")
fit_i, val_i = fixed_split(tr)
big = int(np.bincount(dom[n + len(un):]).argmax())
print("test heavy domain", big, tops[big], flush=True)


def domain_z(s):
    """standardize surprisal inside each inferred domain, over all rows. no labels used."""
    z = np.array(s, dtype=np.float64)
    for d in np.unique(dom):
        m = dom == d
        z[m] = (s[m] - s[m].mean(0)) / (s[m].std(0) + 1e-6)
    return z


Z = domain_z(S)
SZ = np.c_[S, Z]


def text_family(fit):
    def f(a, b):
        return fit(list(texts[a]), y[a])(list(texts[b]))
    return f


def row_family(mat):
    def f(a, b):
        return fit_gbm(mat[a], y[a])(mat[b])
    return f


def ref(a, b):
    lx, st = fit_lexical(texts[a], y[a]), fit_style(texts[a], y[a])
    return .35 * lx(texts[b]) + .65 * st(texts[b])


models = {
    "reference raw blend": ref,
    "reference style only": text_family(fit_style),
    "clean word+char lr": text_family(fit_clean_lexical),
    "surprisal gbm": row_family(S),
    "surprisal + domain z gbm": row_family(SZ),
}
rows, lodo = [], []
for name, f in models.items():
    t = time.time()
    s = f(fit_i, val_i)
    row = dict(model=name, val=auc(y[val_i], s), seconds=round(time.time() - t))
    per = {}
    for d, a, b in lodo_splits(dom[:n], y):
        per[int(d)] = auc(y[b], f(a, b))
    row |= dict(lodo_mean=np.mean(list(per.values())), lodo_min=min(per.values()), lodo_book_poetry=per.get(big, np.nan))
    rows.append(row)
    lodo.append(pd.Series(per, name=name))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
REPORT.mkdir(exist_ok=True)
pd.DataFrame(rows).round(4).to_csv(REPORT / "05_compare.csv", index=False)
pd.DataFrame(lodo).round(4).to_csv(REPORT / "05_lodo_by_domain.csv")
print(pd.DataFrame(lodo).round(3).T)
