from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import time, numpy as np, pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
start = time.time()
D = pd.read_csv('data/train.csv')
U = pd.read_csv('data/unlabeled.csv')
y = D.label.to_numpy()
tr, va = train_test_split(np.arange(len(y)), test_size=2000, random_state=42, stratify=y)
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=8000, sublinear_tf=True, dtype=np.float32)
print('vectorize', flush=True)
texts = D.text.fillna('').astype(str)
A = w.fit_transform(texts.iloc[tr])
B = w.transform(texts.iloc[va])
C = w.transform(U.text.fillna('').astype(str))
print('word done', round(time.time() - start, 1), flush=True)
a = c.fit_transform(texts.iloc[tr])
b = c.transform(texts.iloc[va])
u = c.transform(U.text.fillna('').astype(str))
X = hstack([A, a], format='csr')
V = hstack([B, b], format='csr')
Z = hstack([C, u], format='csr')
print('feature done', X.shape, round(time.time() - start, 1), flush=True)

def fit(x, yy):
    m = SGDClassifier(loss='log_loss', penalty=None, max_iter=30, tol=0.001, random_state=42, average=False)
    m.fit(x, yy)
    return m
m = fit(X, y[tr])
pv = m.predict_proba(V)[:, 1]
pu = m.predict_proba(Z)[:, 1]
results = [{'method': 'supervised_only', 'threshold': None, 'pseudo_rows': 0, 'pseudo_0': 0, 'pseudo_1': 0, 'train_auc': roc_auc_score(y[tr], m.predict_proba(X)[:, 1]), 'validation_auc': roc_auc_score(y[va], pv), 'elapsed_seconds': round(time.time() - start, 1)}]
print('baseline', results[-1], flush=True)
for conf in [0.99, 0.95]:
    selected = np.flatnonzero((pu >= conf) | (pu <= 1 - conf))
    pseudo = (pu[selected] >= conf).astype(int)
    xx = vstack([X, Z[selected]], format='csr')
    yy = np.concatenate([y[tr], pseudo])
    model = fit(xx, yy)
    results.append({'method': 'pseudo_label', 'threshold': conf, 'pseudo_rows': len(selected), 'pseudo_0': int((pseudo == 0).sum()), 'pseudo_1': int((pseudo == 1).sum()), 'train_auc': roc_auc_score(y[tr], model.predict_proba(X)[:, 1]), 'validation_auc': roc_auc_score(y[va], model.predict_proba(V)[:, 1]), 'elapsed_seconds': round(time.time() - start, 1)})
    print('pseudo', results[-1], flush=True)
pd.DataFrame(results).to_csv('outputs/unlabeled_pseudolabel_results.csv', index=False)
print('DONE', round(time.time() - start, 1))
