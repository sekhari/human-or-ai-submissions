from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '2'
import pandas as pd, numpy as np, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
D = pd.read_csv('data/train_starter_2000.csv').sample(n=400, random_state=42).reset_index(drop=True)
t = D.text.fillna('').astype(str)
y = D.label.to_numpy()
tr, va = train_test_split(np.arange(len(D)), test_size=80, random_state=42, stratify=y)
r = []
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2500, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
A = w.fit_transform(t.iloc[tr])
B = w.transform(t.iloc[va])
print('word ready', flush=True)
for ng in [(3, 5), (2, 8)]:
    st = time.time()
    c = TfidfVectorizer(analyzer='char', ngram_range=ng, min_df=3, max_features=2500, sublinear_tf=True, dtype=np.float32)
    C = c.fit_transform(t.iloc[tr])
    print('char fit', ng, time.time() - st, flush=True)
    E = c.transform(t.iloc[va])
    X = hstack([A, C], format='csr')
    V = hstack([B, E], format='csr')
    m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=700)
    m.fit(X, y[tr])
    row = {'character_ngrams': str(ng), 'training_examples': len(tr), 'validation_examples': len(va), 'features': X.shape[1], 'train_auroc': roc_auc_score(y[tr], m.decision_function(X)), 'validation_auroc': roc_auc_score(y[va], m.decision_function(V)), 'runtime_seconds': round(time.time() - st, 2)}
    r.append(row)
    pd.DataFrame(r).to_csv('outputs/char_2_8_screen_results.csv', index=False)
    print(row, flush=True)
