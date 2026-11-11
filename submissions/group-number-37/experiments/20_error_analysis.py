"""report q4: subgroup auroc and the most confident errors on the fixed validation rows.

usage: 20_error_analysis.py ref_weight sur_weight enc_weight. scores come from experiments 06 and 12, fit on the 8000 rows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as auc

from src.data import OUT, fixed_split, load
from src.domains import infer_domains
from src.harness import Ctx, REPORTS

w = [float(x) for x in (sys.argv[1:4] or (.8, .1, .1))]
c = Ctx()
y, v, t = c.y, c.val_i, c.texts
ref = .35 * np.load(OUT / "base_lex_raw.npy") + .65 * np.load(OUT / "base_style_raw.npy")
sur, enc = np.load(OUT / "base_surprisal.npy"), np.load(OUT / "enc_distilroberta-base_fixed_aug0.0_s3737_ep2_drop0.0.npy")
r = lambda x: rankdata(x) / len(x)
p = w[0] * r(ref[v]) + w[1] * r(sur[v]) + w[2] * r(enc[v])
d = pd.DataFrame(dict(id=c.tr.id.values[v], y=y[v], p=p, ref=r(ref[v]), sur=r(sur[v]), enc=r(enc[v]), words=[len(s.split()) for s in t[v]],
                      lines=[s.count("\n") for s in t[v]], dnl=["\n\n" in s for s in t[v]], domain=c.dom[v]))
print("weights", w, "val", round(auc(d.y, d.p), 5))
rows = []
for name, m in [(f"words {a} to {b}", (d.words >= a) & (d.words < b)) for a, b in ((0, 150), (150, 200), (200, 250), (250, 9999))]:
    rows.append(dict(group=name, n=int(m.sum()), auroc=auc(d.y[m], d.p[m])))
for name, m in (("has line breaks", d.lines > 0), ("single block", d.lines == 0)):
    rows.append(dict(group=name, n=int(m.sum()), auroc=auc(d.y[m], d.p[m])))
for k in range(10):
    m = d.domain == k
    rows.append(dict(group=f"domain {k}: {' '.join(c.tops[k][-3:])}", n=int(m.sum()), auroc=auc(d.y[m], d.p[m])))
sg = pd.DataFrame(rows).round(4)
sg.to_csv(REPORTS / "20_subgroups.csv", index=False)
print(sg.to_string(index=False))
thr = d.p.median()
fp, fn = d[(d.y == 0)].sort_values("p", ascending=False).head(8), d[(d.y == 1)].sort_values("p").head(8)
e = pd.concat([fp.assign(error="FP"), fn.assign(error="FN")])
e[["id", "error", "p", "ref", "sur", "enc", "words", "lines", "dnl", "domain"]].round(3).to_csv(REPORTS / "20_errors.csv", index=False)
print("threshold, median blended rank", round(thr, 3), "errors at it", int(((d.p > thr) != d.y).sum()), "of", len(d))
for _, x in e.iterrows():
    print(x.error, x.id, f"p={x.p:.3f} ref={x.ref:.2f} sur={x.sur:.2f} enc={x.enc:.2f} words={x.words} lines={x.lines} dnl={x.dnl} dom={x.domain}", "|", repr(t[x.id - 1][:170]))
