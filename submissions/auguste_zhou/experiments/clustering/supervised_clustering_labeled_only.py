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
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
start = time.time()
D = pd.read_csv('data/train.csv')
text = D.text.fillna('').astype(str)
y = D.label.values
tr, va = train_test_split(np.arange(len(y)), test_size=2000, stratify=y, random_state=42)
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=8000, sublinear_tf=True, dtype=np.float32)
W = w.fit_transform(text.iloc[tr])
Wv = w.transform(text.iloc[va])
print('word ready', time.time() - start, flush=True)
C = c.fit_transform(text.iloc[tr])
Cv = c.transform(text.iloc[va])
X = hstack([W, C], format='csr')
V = hstack([Wv, Cv], format='csr')
print('char ready', time.time() - start, X.shape, flush=True)
Zall = hstack([W, Wu], format='csr') if False else None
from scipy.sparse import vstack
svd = TruncatedSVD(n_components=50, n_iter=5, random_state=42)
Z = svd.fit_transform(W)
Zv = svd.transform(Wv)
print('SVD ready', time.time() - start, flush=True)

def fit_score(name, x, v, nclusters=0):
    m = SGDClassifier(loss='log_loss', penalty=None, max_iter=30, tol=0.001, random_state=42)
    m.fit(x, y[tr])
    t = roc_auc_score(y[tr], m.decision_function(x))
    a = roc_auc_score(y[va], m.decision_function(v))
    result = {'method': name, 'clusters': nclusters, 'train_auc': t, 'validation_auc': a, 'elapsed_seconds': time.time() - start}
    print('RESULT', result, flush=True)
    return result
results = [fit_score('TFIDF_supervised_baseline', X, V)]
for k in [10, 20]:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=512, max_iter=90)
    km.fit(Z)
    distances_train = km.transform(Z[:len(tr)])
    distances_val = km.transform(Zv)
    train_sim = np.exp(-distances_train / (np.median(distances_train) + 1e-08))
    val_sim = np.exp(-distances_val / (np.median(distances_train) + 1e-08))
    train_sim /= train_sim.sum(axis=1, keepdims=True)
    val_sim /= val_sim.sum(axis=1, keepdims=True)
    for wt in [1, 3]:
        XX = hstack([X, csr_matrix((wt * train_sim).astype('float32'))], format='csr')
        VV = hstack([V, csr_matrix((wt * val_sim).astype('float32'))], format='csr')
        results.append(fit_score('TFIDF_plus_cluster_similarity_weight_' + str(wt), XX, VV, k))
    pd.DataFrame(results).to_csv('outputs/supervised_clustering_labeled_only.csv', index=False)
print('DONE', time.time() - start, flush=True)
