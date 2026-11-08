"""Q5 follow-up: does a topic-masked model improve generalization to unseen domains?

Selection uses leave-one-pseudo-domain-out (LODO) predictions inside the 8,000-row training
partition. The fixed validation set is scored once per chosen configuration.
"""
import json, time
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer as V
from sklearn.metrics import roc_auc_score as auc
from common import SEED, load_train, split
from components import NAMES, NOFMT, fit_char, fit_dist, fit_gb, fit_word
from features import perturb_format

tr, va = split(load_train("data/train.csv")); y, yv = tr.label.values, va.label.values
GRID = [(k, c) for k in (100, 300, 1000) for c in (1, 10)]
X = TruncatedSVD(50, random_state=SEED).fit_transform(V(sublinear_tf=True, min_df=3, stop_words="english").fit_transform(tr.text))
cl = KMeans(6, n_init=10, random_state=SEED).fit_predict(X)
folds = [k for k in range(6) if y[cl == k].min() != y[cl == k].max()]
T = tr.text.tolist(); oof = {}; t0 = time.time()
for k in folds:
    te = cl == k; a = [s for s, m in zip(T, te) if not m]; b = [s for s, m in zip(T, te) if m]; ya = y[~te]
    p = {"word": fit_word(a, ya)(b), "char": fit_char(a, ya)(b), "stylo": fit_gb(a, ya, list(range(len(NAMES))))(b), "stylo_nofmt": fit_gb(a, ya, NOFMT)(b)}
    for kk, c in GRID: p[f"dist_{kk}_{c}"] = fit_dist(a, ya, kk, c)(b)
    for n, v in p.items(): oof.setdefault(n, {})[k] = v
    print("fold", k, {n: round(auc(y[te], v), 4) for n, v in p.items()}, round(time.time() - t0), flush=True)
lodo = lambda f: [auc(y[cl == k], f(k)) for k in folds]
res = {"components": {n: dict(mean=float(np.mean(lodo(lambda k: oof[n][k]))), worst=float(min(lodo(lambda k: oof[n][k])))) for n in oof}}
best = max((n for n in oof if n.startswith("dist_")), key=lambda n: res["components"][n]["mean"]); kk, c = map(int, best.split("_")[1:])
def blend(p, w, fmt): s = "stylo" if fmt else "stylo_nofmt"; return (1 - w) * (.25 * p["word"] + .25 * p["char"] + .5 * p[s]) + w * p["dist"]
cands = [(fmt, w) for fmt in (True, False) for w in (0, .2, .35, .5)]
for fmt, w in cands:
    s = lodo(lambda k: blend({**{n: oof[n][k] for n in ("word", "char", "stylo", "stylo_nofmt")}, "dist": oof[best][k]}, w, fmt))
    res.setdefault("blends", []).append(dict(format=fmt, w_dist=w, lodo_mean=float(np.mean(s)), lodo_worst=float(min(s)), per_fold=s))
# Score each blend once on the fixed validation set, clean and format-perturbed.
P = {"word": fit_word(T, y), "char": fit_char(T, y), "stylo": fit_gb(T, y, list(range(len(NAMES)))), "stylo_nofmt": fit_gb(T, y, NOFMT), "dist": fit_dist(T, y, kk, c)}
vt, vp = va.text.tolist(), va.text.map(perturb_format).tolist(); pv = {n: f(vt) for n, f in P.items()}; pp = {n: f(vp) for n, f in P.items()}
res["dist_val"] = auc(yv, pv["dist"]); res["dist_val_perturbed"] = auc(yv, pp["dist"]); res["best_dist"] = dict(k=kk, C=c)
for b in res["blends"]:
    b["val"] = auc(yv, blend(pv, b["w_dist"], b["format"])); b["val_perturbed"] = auc(yv, blend(pp, b["w_dist"], b["format"]))
json.dump(res, open("results/q5_distort.json", "w"), indent=1)
print(pd.DataFrame(res["components"]).T.round(4).to_string()); print("best", best, res["dist_val"], res["dist_val_perturbed"])
print(pd.DataFrame(res["blends"]).drop(columns="per_fold").round(4).to_string())
