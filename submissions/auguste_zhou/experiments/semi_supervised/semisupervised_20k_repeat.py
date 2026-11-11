from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os, time
for x in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[x] = '2'
import numpy as np, pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
T = time.time()
d = pd.read_csv('data/train.csv')
u = pd.read_csv('data/unlabeled.csv')
y = d.label.to_numpy()
tr, va = train_test_split(np.arange(len(y)), test_size=2000, stratify=y, random_state=42)
s = d.text.astype(str)
su = u.text.astype(str)
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=8000, sublinear_tf=True, dtype=np.float32)
W = w.fit_transform(s.iloc[tr])
Wv = w.transform(s.iloc[va])
Wu = w.transform(su)
print('word', time.time() - T, flush=True)
C = c.fit_transform(s.iloc[tr])
Cv = c.transform(s.iloc[va])
Cu = c.transform(su)
print('char', time.time() - T, flush=True)
X = hstack([W, C], format='csr')
V = hstack([Wv, Cv], format='csr')
U = hstack([Wu, Cu], format='csr')
del W, C, Wv, Cv, Wu, Cu
print('shapes', X.shape, V.shape, U.shape, flush=True)

def fit(Xt, yt, weights=None):
    model = SGDClassifier(loss='log_loss', penalty=None, max_iter=25, tol=0.001, random_state=42)
    model.fit(Xt, yt, sample_weight=weights)
    return model
res = []

def add(name, m, n=0, n0=0, n1=0, conf=None, weight=1):
    r = dict(method=name, threshold=conf, pseudo_weight=weight, pseudo_total=n, pseudo_human=n0, pseudo_ai=n1, train_auc=roc_auc_score(y[tr], m.decision_function(X)), validation_auc=roc_auc_score(y[va], m.decision_function(V)), iterations=int(m.n_iter_), seconds=round(time.time() - T, 2))
    res.append(r)
    pd.DataFrame(res).to_csv('outputs/semisupervised_20k_repeat_results.csv', index=False)
    print(r, flush=True)
m = fit(X, y[tr])
add('supervised', m)
pu = m.predict_proba(U)[:, 1]
for conf in [0.99, 0.95]:
    ix = np.flatnonzero((pu >= conf) | (pu <= 1 - conf))
    pseudo = (pu[ix] >= conf).astype(int)
    XX = vstack([X, U[ix]], format='csr')
    yy = np.r_[y[tr], pseudo]
    mm = fit(XX, yy)
    add('pseudo_label', mm, len(ix), int((pseudo == 0).sum()), int((pseudo == 1).sum()), conf)
    if conf == 0.95:
        weights = np.r_[np.ones(len(tr)), np.full(len(ix), 0.25)]
        mm = fit(XX, yy, weights)
        add('pseudo_label_downweighted', mm, len(ix), int((pseudo == 0).sum()), int((pseudo == 1).sum()), conf, 0.25)
print('DONE', round(time.time() - T, 2), flush=True)
