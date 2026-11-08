"""Control for the masking experiment: same tokenizer and n-grams, with fewer or no masked words.

If an unmasked model scores as well on pseudo-domains, the gain comes from the tokenizer
(punctuation tokens, word trigrams), not from removing topic words.
"""
import json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer as V
from sklearn.metrics import roc_auc_score as auc
from common import SEED, load_train, split
from exp_distort import fit_word, fit_char, fit_gb, fit_dist, NOFMT
from features import perturb_format

tr, va = split(load_train("data/train.csv")); y, yv = tr.label.values, va.label.values
X = TruncatedSVD(50, random_state=SEED).fit_transform(V(sublinear_tf=True, min_df=3, stop_words="english").fit_transform(tr.text))
cl = KMeans(6, n_init=10, random_state=SEED).fit_predict(X); folds = [k for k in range(6) if y[cl == k].min() != y[cl == k].max()]
T = tr.text.tolist(); KS = [5000, 10000, None]; oof = {}
for k in folds:
    te = cl == k; a = [s for s, m in zip(T, te) if not m]; b = [s for s, m in zip(T, te) if m]; ya = y[~te]
    p = {"word": fit_word(a, ya)(b), "char": fit_char(a, ya)(b), "stylo_nofmt": fit_gb(a, ya, NOFMT)(b)}
    for kk in KS: p[f"dist_{kk or 'all'}"] = fit_dist(a, ya, kk, 30)(b)
    for n, v in p.items(): oof.setdefault(n, {})[k] = v
    print("fold", k, {n: round(auc(y[te], v), 4) for n, v in p.items()}, flush=True)
score = lambda f: [auc(y[cl == k], f(k)) for k in folds]
out = {"components": {}, "blends": []}
for n in oof:
    s = score(lambda k: oof[n][k]); out["components"][n] = dict(mean=float(np.mean(s)), worst=float(min(s)))
vt, vp = va.text.tolist(), va.text.map(perturb_format).tolist()
base = {"word": fit_word(T, y), "char": fit_char(T, y), "stylo_nofmt": fit_gb(T, y, NOFMT)}
pv = {n: f(vt) for n, f in base.items()}; pp = {n: f(vp) for n, f in base.items()}
for kk in KS:
    f = fit_dist(T, y, kk, 30); n = f"dist_{kk or 'all'}"; pv[n] = f(vt); pp[n] = f(vp)
    out["components"][n].update(val=auc(yv, pv[n]), val_perturbed=auc(yv, pp[n]))
    for d in (.35, .5):
        w = {"word": .25 * (1 - d), "char": .25 * (1 - d), "stylo_nofmt": .5 * (1 - d), n: d}
        s = score(lambda k: sum(v * oof[c][k] for c, v in w.items()))
        out["blends"].append(dict(dist=n, w_dist=d, lodo_mean=float(np.mean(s)), lodo_worst=float(min(s)),
                                  val=auc(yv, sum(v * pv[c] for c, v in w.items())), val_perturbed=auc(yv, sum(v * pp[c] for c, v in w.items()))))
json.dump(out, open("results/q5_distort3.json", "w"), indent=1)
print(pd.DataFrame(out["components"]).T.round(4).to_string()); print(pd.DataFrame(out["blends"]).round(4).to_string())
