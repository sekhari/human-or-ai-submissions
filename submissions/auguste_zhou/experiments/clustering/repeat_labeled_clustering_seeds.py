from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
import time
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
start = time.time()
d = pd.read_csv('data/train.csv')
texts = d.text.fillna('').astype(str)
y = d.label.to_numpy()
tr, va = train_test_split(np.arange(len(y)), test_size=2000, stratify=y, random_state=42)
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=8000, sublinear_tf=True, dtype=np.float32)
W = w.fit_transform(texts.iloc[tr])
Wv = w.transform(texts.iloc[va])
print('word', round(time.time() - start, 2), flush=True)
C = c.fit_transform(texts.iloc[tr])
Cv = c.transform(texts.iloc[va])
print('char', round(time.time() - start, 2), flush=True)
X = hstack([W, C], format='csr')
V = hstack([Wv, Cv], format='csr')
svd = TruncatedSVD(n_components=50, n_iter=5, random_state=42)
Z = svd.fit_transform(W)
Zv = svd.transform(Wv)
print('svd', round(time.time() - start, 2), flush=True)
rows = []

def evaluate(seed, method, Xt, Xv):
    m = SGDClassifier(loss='log_loss', penalty=None, max_iter=30, tol=0.001, random_state=seed)
    m.fit(Xt, y[tr])
    return {'seed': seed, 'method': method, 'n_clusters': 0 if method == 'baseline' else 10, 'train_auc': roc_auc_score(y[tr], m.decision_function(Xt)), 'validation_auc': roc_auc_score(y[va], m.decision_function(Xv)), 'seconds_total': time.time() - start}
for seed in [13, 42, 123]:
    b = evaluate(seed, 'baseline', X, V)
    rows.append(b)
    print(b, flush=True)
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
    r = evaluate(seed, 'cluster_10_weight_1', Xt, Xv)
    rows.append(r)
    print(r, flush=True)
    pd.DataFrame(rows).to_csv('outputs/repeated_labeled_only_cluster_results.csv', index=False)
print('DONE', round(time.time() - start, 2))
