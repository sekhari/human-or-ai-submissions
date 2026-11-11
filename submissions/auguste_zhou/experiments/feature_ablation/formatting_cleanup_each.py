from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import re, time, json
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
BASE = 'outputs'
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample = pd.read_csv('data/sample_submission.csv')
pattern = re.compile('<\\s*br\\s*/?\\s*>', re.I)
block = re.compile('</?\\s*(?:p|div)\\b[^>]*>', re.I)
spaces = re.compile('[ \\t\\f\\v]+')
newlines = re.compile('\\n+')

def cleanup(txt):
    txt = str(txt) if pd.notna(txt) else ''
    txt = pattern.sub('\n', txt)
    txt = block.sub('\n', txt)
    txt = txt.replace('\r\n', '\n').replace('\r', '\n').replace('\\n', '\n') if False else txt.replace('\r\n', '\n').replace('\r', '\n')
    txt = spaces.sub(' ', txt)
    txt = re.sub(' *\\n *', '\n', txt)
    txt = newlines.sub(' ', txt)
    return txt.strip()
texts = train.text.fillna('').astype(str)
clean = texts.map(cleanup)
print('changed rows:', int((texts != clean).sum()), 'of', len(train), flush=True)
tr, va = train_test_split(np.arange(len(train)), test_size=2000, stratify=train.label, random_state=42)
y = train.label.values
rows = []
for key, corpus in [('original', texts)] if os.environ.get('MODE') == 'original' else [('normalized_formatting', clean)]:
    start = time.time()
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
    c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=12000, sublinear_tf=True, dtype=np.float32)
    X = hstack([w.fit_transform(corpus.iloc[tr]), c.fit_transform(corpus.iloc[tr])], format='csr')
    V = hstack([w.transform(corpus.iloc[va]), c.transform(corpus.iloc[va])], format='csr')
    m = SGDClassifier(loss='log_loss', penalty=None, max_iter=25, tol=0.001, random_state=42, alpha=0.0001)
    m.fit(X, y[tr])
    p = m.predict_proba(V)[:, 1]
    q = m.predict_proba(X)[:, 1]
    result = dict(preprocessing=key, train_auc=roc_auc_score(y[tr], q), val_auc=roc_auc_score(y[va], p), train_time_sec=time.time() - start, features=X.shape[1], iterations=int(m.n_iter_), changed_texts=int((texts != clean).sum()))
    print(result, flush=True)
    rows.append(result)
pd.DataFrame(rows).to_csv(f'{BASE}/formatting_cleanup_{key}.csv', index=False)
