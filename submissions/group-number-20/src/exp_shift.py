"""Q5: does format normalization or ensembling improve robustness to domain and formatting shift?

Reads tuned settings from results/q1_models.json. Clean validation, a format-perturbed copy of
validation, and leave-one-pseudo-domain-out CV are measured with the same split and hyperparameters.
"""
import json, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.feature_extraction.text import TfidfVectorizer as V
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as auc
from common import SEED, load_train, split
from features import normalize, perturb_format, stylometric_matrix

Q = {r["name"]: r for r in json.load(open("results/q1_models.json"))}
tr, va = split(load_train("data/train.csv")); y, yv = tr.label.values, va.label.values
vp = va.text.map(perturb_format)
B = dict(sublinear_tf=True, strip_accents="unicode", lowercase=True)
def fit_tfidf(tx, yy, kind):
    q = Q["word_lr" if kind == "word" else "char_lr"]["best"]; ng = tuple(q["ngram"])
    v = V(analyzer="word", ngram_range=ng, min_df=2, max_df=.95, **B) if kind == "word" else V(analyzer="char_wb", ngram_range=ng, min_df=3, max_features=400000, **B)
    m = LR(C=q["value"], max_iter=3000).fit(v.fit_transform(tx), yy)
    return lambda t: m.decision_function(v.transform(t))
def fit_stylo(tx, yy):
    q = Q["stylo_gbdt"]["best"]
    m = HGB(learning_rate=q["lr"], max_leaf_nodes=q["leaf"], max_iter=300, random_state=SEED).fit(stylometric_matrix(tx), yy)
    return lambda t: m.predict_proba(stylometric_matrix(t))[:, 1]
n = lambda s: pd.Series(s).map(normalize)
def fits(tx, yy):
    w, c = fit_tfidf(tx, yy, "word"), fit_tfidf(tx, yy, "char")
    nw, nc = fit_tfidf(n(tx), yy, "word"), fit_tfidf(n(tx), yy, "char")
    return {"word_raw": w, "char_raw": c, "word_norm": lambda t: nw(n(t)), "char_norm": lambda t: nc(n(t))}
ens = lambda p, k: sum(rankdata(p[x]) for x in k) / len(k)
E_NORM, E_RAW = ("word_norm", "char_norm"), ("word_raw", "char_raw")
def evaluate(f, t):
    p = {k: g(t) for k, g in f.items()}; p["ens_raw"] = ens(p, E_RAW); p["ens_norm"] = ens(p, E_NORM); return p
t0 = time.time(); F = fits(tr.text, y); clean = evaluate(F, va.text); pert = evaluate(F, vp)
res = {k: dict(val=auc(yv, clean[k]), val_perturbed=auc(yv, pert[k])) for k in clean}
# Pseudo-domains: KMeans over training text. These are unsupervised stand-ins for unseen domains.
X = TruncatedSVD(50, random_state=SEED).fit_transform(V(sublinear_tf=True, min_df=3, stop_words="english").fit_transform(tr.text))
cl = KMeans(6, n_init=10, random_state=SEED).fit_predict(X); per = []
for k in range(6):
    te = cl == k; G = fits(tr.text[~te].reset_index(drop=True), y[~te]); P = evaluate(G, tr.text[te].reset_index(drop=True))
    r = {m: auc(y[te], P[m]) for m in P}; r.update(cluster=k, n=int(te.sum()), ai_rate=float(y[te].mean())); per.append(r); print(r, flush=True)
d = pd.DataFrame(per)
for k in res: res[k].update(lodo_mean=float(d[k].mean()), lodo_worst=float(d[k].min()))
json.dump(dict(configs=res, clusters=per, seconds=time.time() - t0), open("results/q5_shift.json", "w"), indent=1)
print(pd.DataFrame(res).T.round(4))
