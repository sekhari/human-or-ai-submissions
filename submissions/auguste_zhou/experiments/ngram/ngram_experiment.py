from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
import time, pandas as pd, numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D = pd.read_csv('data/train.csv')
text = D.text.fillna('').astype(str)
y = D.label.to_numpy()
tr, va = train_test_split(np.arange(len(y)), test_size=2000, stratify=y, random_state=42)
base = {'min_df': 2, 'sublinear_tf': True, 'strip_accents': 'unicode', 'dtype': np.float32}
configs = [('word_1_1', (1, 1), None, 26000), ('word_1_2', (1, 2), None, 26000), ('word_1_3', (1, 3), None, 26000), ('word_1_4', (1, 4), None, 26000), ('char_3_5', None, (3, 5), 35000), ('word_1_2_char_3_5', (1, 2), (3, 5), 26000), ('word_1_3_char_3_5', (1, 3), (3, 5), 26000)]
results = []
for name, wn, cn, nw in configs:
    t = time.time()
    parts = []
    partsval = []
    if wn:
        v = TfidfVectorizer(ngram_range=wn, max_features=nw, **base)
        parts.append(v.fit_transform(text.iloc[tr]))
        partsval.append(v.transform(text.iloc[va]))
    if cn:
        v = TfidfVectorizer(analyzer='char', ngram_range=cn, min_df=3, max_features=35000, sublinear_tf=True, dtype=np.float32)
        parts.append(v.fit_transform(text.iloc[tr]))
        partsval.append(v.transform(text.iloc[va]))
    X = hstack(parts, format='csr') if len(parts) > 1 else parts[0]
    V = hstack(partsval, format='csr') if len(partsval) > 1 else partsval[0]
    clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=400, tol=0.0001)
    clf.fit(X, y[tr])
    a = roc_auc_score(y[tr], clf.decision_function(X))
    b = roc_auc_score(y[va], clf.decision_function(V))
    row = {'model': name, 'word_ngram': str(wn), 'char_ngram': str(cn), 'features': X.shape[1], 'train_auc': a, 'val_auc': b, 'seconds': round(time.time() - t, 1), 'iterations': clf.n_iter_[0]}
    results.append(row)
    pd.DataFrame(results).to_csv('outputs/ngram_results.csv', index=False)
    print(row, flush=True)
