"""Extend the topic-masking grid (best values were at the grid edge) and compare blends.

All selection uses leave-one-pseudo-domain-out (LODO) predictions inside the training
partition. Out-of-cluster predictions are cached so blends are compared on the same folds.
"""
import json, itertools
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer as V
from sklearn.metrics import roc_auc_score as auc
from common import SEED, load_train, split
from components import NAMES, NOFMT, fit_char, fit_dist, fit_gb, fit_word
from features import perturb_format

tr, va = split(load_train("data/train.csv")); y, yv = tr.label.values, va.label.values
X = TruncatedSVD(50, random_state=SEED).fit_transform(V(sublinear_tf=True, min_df=3, stop_words="english").fit_transform(tr.text))
cl = KMeans(6, n_init=10, random_state=SEED).fit_predict(X); folds = [k for k in range(6) if y[cl == k].min() != y[cl == k].max()]
T = tr.text.tolist(); GRID = [(k, c) for k in (1000, 2000, 5000) for c in (10, 30)]
oof = {}
for k in folds:
    te = cl == k; a = [s for s, m in zip(T, te) if not m]; b = [s for s, m in zip(T, te) if m]; ya = y[~te]
    p = {"word": fit_word(a, ya)(b), "char": fit_char(a, ya)(b), "stylo": fit_gb(a, ya, list(range(len(NAMES))))(b), "stylo_nofmt": fit_gb(a, ya, NOFMT)(b)}
    for kk, c in GRID: p[f"dist_{kk}_{c}"] = fit_dist(a, ya, kk, c)(b)
    for n, v in p.items(): oof.setdefault(n, {})[k] = v
    print("fold", k, {n: round(auc(y[te], v), 4) for n, v in p.items()}, flush=True)
score = lambda f: [auc(y[cl == k], f(k)) for k in folds]
comp = {n: score(lambda k: oof[n][k]) for n in oof}
best = max((n for n in oof if n.startswith("dist_")), key=lambda n: np.mean(comp[n])); kk, c = map(int, best.split("_")[1:])
def mix(p, w): return sum(w[n] * p[n] for n in w)
cands = {}
for fmt in (True, False):
    s = "stylo" if fmt else "stylo_nofmt"
    for d in (0, .2, .35, .5):
        cands[f"{'fmt' if fmt else 'nofmt'}_base+dist{d}"] = {"word": .25 * (1 - d), "char": .25 * (1 - d), s: .5 * (1 - d), "dist": d}
    for a in (.35, .5, .65):
        cands[f"{'fmt' if fmt else 'nofmt'}_dist{a}+stylo"] = {"dist": a, s: 1 - a}
rows = []
for name, w in cands.items():
    sc = score(lambda k: mix({**{n: oof[n][k] for n in ("word", "char", "stylo", "stylo_nofmt")}, "dist": oof[best][k]}, w))
    rows.append(dict(name=name, weights=w, lodo_mean=float(np.mean(sc)), lodo_worst=float(min(sc))))
P = {"word": fit_word(T, y), "char": fit_char(T, y), "stylo": fit_gb(T, y, list(range(len(NAMES)))), "stylo_nofmt": fit_gb(T, y, NOFMT), "dist": fit_dist(T, y, kk, c)}
vt, vp = va.text.tolist(), va.text.map(perturb_format).tolist(); pv = {n: f(vt) for n, f in P.items()}; pp = {n: f(vp) for n, f in P.items()}
for r in rows: r["val"] = auc(yv, mix(pv, r["weights"])); r["val_perturbed"] = auc(yv, mix(pp, r["weights"]))
out = dict(components={n: dict(mean=float(np.mean(v)), worst=float(min(v))) for n, v in comp.items()}, best_dist=dict(k=kk, C=c),
           dist_val=auc(yv, pv["dist"]), dist_val_perturbed=auc(yv, pp["dist"]), blends=rows)
json.dump(out, open("results/q5_distort2.json", "w"), indent=1)
print(pd.DataFrame(out["components"]).T.round(4).to_string()); print("best", best, round(out["dist_val"], 4))
print(pd.DataFrame(rows).drop(columns="weights").sort_values("lodo_mean", ascending=False).round(4).to_string())
