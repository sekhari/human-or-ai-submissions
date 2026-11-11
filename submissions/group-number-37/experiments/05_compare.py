"""compare model families on the fixed split and with each inferred domain held out.

usage: 05_compare.py [family ...]   with no args runs every family that needs no gpu features.
writes reports/05_compare.csv and reports/05_lodo_by_domain.csv.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc

from src.baseline import fit_lexical, fit_style
from src.harness import Ctx, REPORTS, upsert
from src.models import cleaned, fit_clean_lexical, fit_gbm
from src.validate import lodo_splits

c = Ctx()
texts, y, n = c.texts, c.y, c.n
print("test heavy domain", c.big, c.tops[c.big], "train rows", (c.dom[:n] == c.big).sum(), flush=True)


def on_text(fit):
    return lambda a: (lambda f: lambda b: f(list(texts[b])))(fit(list(texts[a]), y[a]))


def on_rows(mat, **kw):
    return lambda a: (lambda f: lambda b: f(mat[b]))(fit_gbm(mat[a], y[a], **kw))


def reference(a):
    lx, st = fit_lexical(list(texts[a]), y[a]), fit_style(list(texts[a]), y[a])
    return lambda b: .35 * lx(list(texts[b])) + .65 * st(list(texts[b]))


def dnl_rule(a):
    return lambda b: np.array(["\n\n" in t for t in texts[b]], dtype=float)


def clean_style(a):
    st = fit_style(cleaned(texts[a]), y[a])
    return lambda b: st(cleaned(texts[b]))


families = {
    "double newline rule": dnl_rule,
    "reference blend raw": reference,
    "reference lexical raw": on_text(fit_lexical),
    "reference style raw": on_text(fit_style),
    "style on cleaned text": clean_style,
    "clean word+char lr": on_text(fit_clean_lexical),
    "clean word lr": on_text(lambda t, l: fit_clean_lexical(t, l, char=False)),
    "clean char lr": on_text(lambda t, l: fit_clean_lexical(t, l, word=False)),
}
if (c_ := Path(__file__).resolve().parents[1] / "results" / "surprisal_gpt2.npy").exists():
    S = c.surprisal()
    families["surprisal gbm"] = on_rows(S)
    families["surprisal + domain z gbm"] = on_rows(np.c_[S, c.domain_z(S)])

pick = sys.argv[1:] or list(families)
lodo_path = REPORTS / "05_lodo_by_domain.csv"
for name in pick:
    t = time.time()
    fit = families[name](c.fit_i)
    train_s = round(time.time() - t)
    row = dict(model=name, train_auc=auc(y[c.fit_i], fit(c.fit_i)), val=auc(y[c.val_i], fit(c.val_i)), train_seconds=train_s)
    per = {}
    for d, a, b in lodo_splits(c.dom[:n], y):
        per[int(d)] = auc(y[b], families[name](a)(b))
    row |= dict(lodo_mean=float(np.mean(list(per.values()))), lodo_min=min(per.values()), lodo_book_poetry=per[c.big])
    upsert("05_compare.csv", row)
    upsert("05_lodo_by_domain.csv", {"model": name, **{str(k): v for k, v in per.items()}})
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, round(time.time() - t), "s", flush=True)
