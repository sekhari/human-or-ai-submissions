"""Q1: model comparison. Hyperparameters use group CV inside the 8,000-row training partition only."""
import json, time, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.feature_extraction.text import TfidfVectorizer as V
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from common import SEED, load_train, split
from features import stylometric_matrix
from neardup import groups

tr, va = split(load_train("data/train.csv")); y, yv = tr.label.values, va.label.values
fold = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(tr, y, groups(tr.text.tolist(), .7)[0]))
sc = lambda m, X: m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]
def vec(kind, ng):
    a = dict(sublinear_tf=True, strip_accents="unicode", lowercase=True, ngram_range=ng)
    return V(analyzer="word", min_df=2, max_df=.95, **a) if kind == "word" else V(analyzer="char_wb", min_df=3, max_features=400000, **a)
def tfidf_grid(name, kind, ngs, mk, vals, pn):
    rows = []
    for ng in ngs:
        cache = []
        for i, j in fold:
            v = vec(kind, ng); cache.append((v.fit_transform(tr.text.iloc[i]), v.transform(tr.text.iloc[j]), i, j))
        for c in vals:
            s = [auc(y[j], sc(mk(c).fit(a, y[i]), b)) for a, b, i, j in cache]
            rows.append(dict(ngram=ng, value=c, cv=float(np.mean(s)), sd=float(np.std(s)))); print(name, rows[-1], flush=True)
    b = max(rows, key=lambda r: r["cv"]); t0 = time.time(); v = vec(kind, b["ngram"]); X = v.fit_transform(tr.text); m = mk(b["value"]).fit(X, y); ft = time.time() - t0
    return dict(name=name, param=pn, grid=rows, best=b, train=auc(y, sc(m, X)), val=auc(yv, sc(m, v.transform(va.text))), seconds=ft)
out = [
 tfidf_grid("word_lr", "word", [(1, 1), (1, 2)], lambda c: LR(C=c, max_iter=3000), [.3, 1, 3, 10, 30], "C"),
 tfidf_grid("char_lr", "char", [(3, 5), (2, 6)], lambda c: LR(C=c, max_iter=3000), [1, 3, 10, 30, 100], "C"),
 tfidf_grid("char_svm", "char", [(3, 5)], lambda c: LinearSVC(C=c, max_iter=5000), [.03, .1, .3, 1], "C"),
]
S, Sv = stylometric_matrix(tr.text), stylometric_matrix(va.text); rows = []
for lr in (.03, .1):
    for leaf in (15, 31):
        mk = lambda: HGB(learning_rate=lr, max_leaf_nodes=leaf, max_iter=300, random_state=SEED)
        s = [auc(y[j], mk().fit(S[i], y[i]).predict_proba(S[j])[:, 1]) for i, j in fold]
        rows.append(dict(lr=lr, leaf=leaf, cv=float(np.mean(s)), sd=float(np.std(s)))); print(rows[-1], flush=True)
b = max(rows, key=lambda r: r["cv"]); t0 = time.time(); m = HGB(learning_rate=b["lr"], max_leaf_nodes=b["leaf"], max_iter=300, random_state=SEED).fit(S, y)
out.append(dict(name="stylo_gbdt", grid=rows, best=b, train=auc(y, m.predict_proba(S)[:, 1]), val=auc(yv, m.predict_proba(Sv)[:, 1]), seconds=time.time() - t0))
json.dump(out, open("results/q1_models.json", "w"), indent=1, default=list)
for r in out: print(r["name"], r["best"], round(r["train"], 4), round(r["val"], 4), round(r["seconds"], 1))
