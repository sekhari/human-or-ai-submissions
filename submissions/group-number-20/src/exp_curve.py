"""Q2: learning curve, and Q3: self-training with unlabeled.csv, on the fixed validation set."""
import json, argparse
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as V
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import train_test_split
from common import SEED, load_train, split
from features import normalize

Q = {r["name"]: r for r in json.load(open("results/q1_models.json"))}["word_lr"]["best"]
tr, va = split(load_train("data/train.csv")); tr["text"] = tr.text.map(normalize); va["text"] = va.text.map(normalize)
un = pd.read_csv("data/unlabeled.csv"); un["text"] = un.text.map(normalize)
def fit(t, y):
    v = V(ngram_range=tuple(Q["ngram"]), min_df=2, max_df=.95, sublinear_tf=True, strip_accents="unicode")
    m = LR(C=Q["value"], max_iter=3000).fit(v.fit_transform(t), y); return lambda x: m.predict_proba(v.transform(x))[:, 1]
rows = []
for n in (500, 1000, 2000, 5000, 8000):
    a, b = [], []
    for s in range(5 if n < 8000 else 1):
        d = tr if n == 8000 else train_test_split(tr, train_size=n, stratify=tr.label, random_state=SEED + s)[0]
        f = fit(d.text, d.label); a.append(auc(d.label, f(d.text))); b.append(auc(va.label, f(va.text)))
    rows.append(dict(n=n, train=float(np.mean(a)), val=float(np.mean(b)), val_sd=float(np.std(b)), repeats=len(b))); print(rows[-1], flush=True)
base = fit(tr.text, tr.label); semi = []
for hi in (.95, .9, .8):
    f = base; added = pd.Series(False, index=un.index); labs = pd.Series(0, index=un.index); log = []
    for it in range(3):
        p = pd.Series(f(un.text), index=un.index); c = (p >= hi) | (p <= 1 - hi); labs[c] = (p[c] >= .5).astype(int); added = c
        f = fit(pd.concat([tr.text, un.text[added]]), pd.concat([tr.label, labs[added]]))
        log.append(dict(iter=it, added=int(added.sum()), pos_rate=float(labs[added].mean()), val=auc(va.label, f(va.text))))
    semi.append(dict(threshold=hi, iters=log)); print(semi[-1], flush=True)
json.dump(dict(setting=Q, curve=rows, supervised_val=auc(va.label, base(va.text)), semi=semi), open("results/q2_q3.json", "w"), indent=1)
