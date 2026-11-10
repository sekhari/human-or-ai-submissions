"""Q5 follow-up: how far does the blank-line artifact carry, and does it look stable in test?

Compares text-only, stylometric, and blended scores on validation, pseudo-domains, and within
format strata. The test check uses only text-model scores and the blank-line flag, never labels.
"""
import json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.feature_extraction.text import TfidfVectorizer as V
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold
from common import SEED, load_train, split
from features import NAMES, normalize, stylometric_matrix

Q = {r["name"]: r for r in json.load(open("results/q1_models.json"))}
tr, va = split(load_train("data/train.csv")); y, yv = tr.label.values, va.label.values
te = pd.read_csv("data/test.csv")
B = dict(sublinear_tf=True, strip_accents="unicode", lowercase=True)
FMT = [NAMES.index(n) for n in ("newline", "bullet")]; KEEP = [i for i in range(len(NAMES)) if i not in FMT]
def text_fit(t, yy):
    w, c = Q["word_lr"]["best"], Q["char_lr"]["best"]; t = [normalize(x) for x in t]
    a = V(analyzer="word", ngram_range=tuple(w["ngram"]), min_df=2, max_df=.95, **B); b = V(analyzer="char_wb", ngram_range=tuple(c["ngram"]), min_df=3, max_features=400000, **B)
    m, n = LR(C=w["value"], max_iter=3000).fit(a.fit_transform(t), yy), LR(C=c["value"], max_iter=3000).fit(b.fit_transform(t), yy)
    def f(x):
        x = [normalize(s) for s in x]; return (rankdata(m.decision_function(a.transform(x))) + rankdata(n.decision_function(b.transform(x)))) / 2
    return f
def gb(t, yy, cols):
    q = Q["stylo_gbdt"]["best"]; m = HGB(learning_rate=q["lr"], max_leaf_nodes=q["leaf"], max_iter=300, random_state=SEED).fit(stylometric_matrix(t)[:, cols], yy)
    return lambda x: m.predict_proba(stylometric_matrix(x)[:, cols])[:, 1]
ALL = list(range(len(NAMES)))
def models(t, yy):
    T, S, N = text_fit(t, yy), gb(t, yy, ALL), gb(t, yy, KEEP); return T, S, N
def score(M, x):
    T, S, N = M; p = dict(text=T(x), stylo_all=S(x), stylo_nofmt=N(x))
    r = {k: rankdata(v) for k, v in p.items()}
    p["blend"] = (r["text"] + r["stylo_all"]) / 2; p["blend_nofmt"] = (r["text"] + r["stylo_nofmt"]) / 2
    dbl = np.array(["\n\n" in s for s in x]).astype(float)
    p["text_plus_flag"] = r["text"] / len(x) + dbl * 1.0
    return p
M = models(tr.text.tolist(), y); P = score(M, va.text.tolist()); vt = va.text.tolist()
dbl = np.array(["\n\n" in s for s in vt])
res = {k: dict(val=auc(yv, v), no_blank_line=auc(yv[~dbl], v[~dbl]), one_line=auc(yv[~va.text.str.contains("\n").values], v[~va.text.str.contains("\n").values])) for k, v in P.items()}
X = TruncatedSVD(50, random_state=SEED).fit_transform(V(sublinear_tf=True, min_df=3, stop_words="english").fit_transform(tr.text))
cl = KMeans(6, n_init=10, random_state=SEED).fit_predict(X); per = []
for k in range(6):
    m = cl == k
    if y[m].min() == y[m].max(): continue
    G = models(tr.text[~m].tolist(), y[~m]); q = score(G, tr.text[m].tolist()); per.append({"cluster": k, "n": int(m.sum()), **{a: auc(y[m], b) for a, b in q.items()}}); print(per[-1], flush=True)
d = pd.DataFrame(per)
for k in res: res[k].update(lodo_mean=float(d[k].mean()), lodo_worst=float(d[k].min()))
# Label-free check: do test rows with blank lines score like known-AI text under a text-only model?
oof = np.zeros(len(tr))
for i, j in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y):
    w = Q["word_lr"]["best"]; a = V(ngram_range=tuple(w["ngram"]), min_df=2, max_df=.95, **B); l = LR(C=w["value"], max_iter=3000).fit(a.fit_transform(tr.text.iloc[i]), y[i])
    oof[j] = l.predict_proba(a.transform(tr.text.iloc[j]))[:, 1]
w = Q["word_lr"]["best"]; a = V(ngram_range=tuple(w["ngram"]), min_df=2, max_df=.95, **B); l = LR(C=w["value"], max_iter=3000).fit(a.fit_transform(tr.text), y)
pt = l.predict_proba(a.transform(te.text))[:, 1]; tb = te.text.str.contains("\n\n").values; rb = tr.text.str.contains("\n\n").values
chk = dict(train_oof_mean_blank=float(oof[rb].mean()), train_oof_mean_noblank_human=float(oof[~rb & (y == 0)].mean()), train_oof_mean_noblank_ai=float(oof[~rb & (y == 1)].mean()),
           test_mean_blank=float(pt[tb].mean()), test_mean_noblank=float(pt[~tb].mean()), test_share_blank=float(tb.mean()))
json.dump(dict(configs=res, clusters=per, label_free=chk), open("results/q5_format.json", "w"), indent=1)
print(pd.DataFrame(res).T.round(4)); print(chk)
