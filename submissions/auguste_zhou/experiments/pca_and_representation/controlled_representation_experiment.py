from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D = pd.read_csv('data/train.csv')
y = D.label.to_numpy()
s = D.text.fillna('').astype(str)
tr, va = train_test_split(np.arange(len(D)), test_size=2000, random_state=42, stratify=y)
rows = []
for name, nw, nc, hashc in [('reduced_hashed', 4000, 4096, True)]:
    t = time.perf_counter()
    print('BUILD', name, flush=True)
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=nw, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
    W = w.fit_transform(s.iloc[tr])
    Vw = w.transform(s.iloc[va])
    print('word', W.shape, time.perf_counter() - t, flush=True)
    if hashc:
        h = HashingVectorizer(analyzer='char', ngram_range=(3, 5), n_features=nc, alternate_sign=False, norm=None, dtype=np.float32)
        C = h.transform(s.iloc[tr])
        Vc = h.transform(s.iloc[va])
        tf = TfidfTransformer(sublinear_tf=True)
        C = tf.fit_transform(C)
        Vc = tf.transform(Vc)
    else:
        c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=nc, sublinear_tf=True, dtype=np.float32)
        C = c.fit_transform(s.iloc[tr])
        Vc = c.transform(s.iloc[va])
    print('char', C.shape, 'time', time.perf_counter() - t, flush=True)
    X = hstack([W, C], format='csr')
    V = hstack([Vw, Vc], format='csr')
    del C, Vc, W, Vw
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=3000, tol=0.0001)
    fit_t = time.perf_counter()
    model.fit(X, y[tr])
    seconds = time.perf_counter() - fit_t
    r = {'representation': name, 'word_features': nw, 'char_features': nc, 'actual_features': X.shape[1], 'solver': 'lbfgs', 'regularization': 'none', 'split_seed': 42, 'train_auc': roc_auc_score(y[tr], model.decision_function(X)), 'val_auc': roc_auc_score(y[va], model.decision_function(V)), 'fit_seconds': seconds, 'total_seconds': time.perf_counter() - t, 'n_iter': int(model.n_iter_[0])}
    rows.append(r)
    pd.DataFrame(rows).to_csv('outputs/controlled_representation_results.csv', index=False)
    print('RESULT', r, flush=True)
