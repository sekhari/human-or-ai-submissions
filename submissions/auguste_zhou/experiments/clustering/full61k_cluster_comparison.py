from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '2'
import time
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
T = time.time()
d = pd.read_csv('data/train.csv')
s = d.text.fillna('').astype(str)
y = d.label.to_numpy()
tr, va = train_test_split(np.arange(len(y)), test_size=2000, stratify=y, random_state=42)
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=26000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=35000, sublinear_tf=True, dtype=np.float32)
W = w.fit_transform(s.iloc[tr])
Wv = w.transform(s.iloc[va])
print('word', W.shape, 'seconds', round(time.time() - T, 2), flush=True)
C = c.fit_transform(s.iloc[tr])
Cv = c.transform(s.iloc[va])
print('char', C.shape, 'seconds', round(time.time() - T, 2), flush=True)
X = hstack([W, C], format='csr')
V = hstack([Wv, Cv], format='csr')
del C, Cv
svd = TruncatedSVD(n_components=50, n_iter=5, random_state=42)
Z = svd.fit_transform(W)
Zv = svd.transform(Wv)
print('SVD', Z.shape, 'seconds', round(time.time() - T, 2), flush=True)
rows = []

def fit(meth, seed, Xt, Xv):
    t = time.time()
    m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=3000, tol=0.0001, random_state=seed)
    m.fit(Xt, y[tr])
    return dict(method=meth, seed=seed, train_auc=roc_auc_score(y[tr], m.decision_function(Xt)), validation_auc=roc_auc_score(y[va], m.decision_function(Xv)), n_train=len(tr), n_validation=len(va), n_features=Xt.shape[1], fit_seconds=time.time() - t, n_iter=int(m.n_iter_[0]))
r = fit('original_61k', 42, X, V)
rows.append(r)
print(r, flush=True)
for seed in [13, 42, 123]:
    t = time.time()
    km = MiniBatchKMeans(n_clusters=10, random_state=seed, n_init=3, batch_size=512, max_iter=90)
    km.fit(Z)
    dt = km.transform(Z)
    dv = km.transform(Zv)
    scale = np.median(dt) + 1e-08
    st = np.exp(-dt / scale)
    sv = np.exp(-dv / scale)
    st /= st.sum(axis=1, keepdims=True)
    sv /= sv.sum(axis=1, keepdims=True)
    Xt = hstack([X, csr_matrix(st.astype(np.float32))], format='csr')
    Xv = hstack([V, csr_matrix(sv.astype(np.float32))], format='csr')
    print('clusters ready', seed, round(time.time() - t, 2), flush=True)
    r = fit('cluster_10_weight_1', seed, Xt, Xv)
    rows.append(r)
    pd.DataFrame(rows).to_csv('outputs/full61k_cluster_comparison.csv', index=False)
    print(r, 'total', round(time.time() - T, 2), flush=True)
print('DONE', round(time.time() - T, 2), flush=True)
