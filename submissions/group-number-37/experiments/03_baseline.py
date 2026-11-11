"""reference scores on the fixed split, plus what happens without the double newline signal."""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import roc_auc_score as auc

from src.baseline import fit_lexical, fit_style
from src.data import load_all, fixed_split

train, _, test = load_all()
tr, va = fixed_split(train)
Xt, yt, Xv, yv = train.text.values[tr], train.label.values[tr], train.text.values[va], train.label.values[va]
flat = lambda a: np.array([re.sub(r"\s+", " ", t).strip() for t in a], dtype=object)
dn = lambda a: np.array([re.sub(r"\n\s*\n", "\n", t) for t in a], dtype=object)

t = time.time(); lex = fit_lexical(Xt, yt); print("lex fit", round(time.time() - t), flush=True)
t = time.time(); sty = fit_style(Xt, yt); print("style fit", round(time.time() - t), flush=True)
blend = lambda x: .35 * lex(x) + .65 * sty(x)
print("train auc lex", auc(yt, lex(Xt)), "style", auc(yt, sty(Xt)))
for name, f in (("as is", lambda a: a), ("no double newline", dn), ("all whitespace flat", flat)):
    x = f(Xv)
    print(f"{name:22s} lex {auc(yv, lex(x)):.4f} style {auc(yv, sty(x)):.4f} blend {auc(yv, blend(x)):.4f}", flush=True)

print("--- retrained with double newlines removed everywhere")
lex2, sty2 = fit_lexical(dn(Xt), yt), fit_style(dn(Xt), yt)
x = dn(Xv); print(f"lex {auc(yv, lex2(x)):.4f} style {auc(yv, sty2(x)):.4f} blend {auc(yv, .35*lex2(x)+.65*sty2(x)):.4f}", flush=True)
print("--- retrained flat")
lex3, sty3 = fit_lexical(flat(Xt), yt), fit_style(flat(Xt), yt)
x = flat(Xv); print(f"lex {auc(yv, lex3(x)):.4f} style {auc(yv, sty3(x)):.4f} blend {auc(yv, .35*lex3(x)+.65*sty3(x)):.4f}", flush=True)
has = np.array(["\n\n" in s for s in test.text])
p = sty(test.text.values); print("style score on test with dnl mean", p[has].mean(), "without", p[~has].mean())
