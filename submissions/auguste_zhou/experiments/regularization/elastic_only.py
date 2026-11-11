from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import time, re, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
D = pd.read_csv('data/train.csv')
y = D.label.to_numpy()
text = D.text.fillna('').astype(str)
tr, va = train_test_split(np.arange(len(D)), test_size=2000, random_state=42, stratify=y)

def build_text_features(texts):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=4000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
    c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=4000, sublinear_tf=True, dtype=np.float32)
    a = hstack((w.fit_transform(texts.iloc[tr]), c.fit_transform(texts.iloc[tr])), format='csr')
    b = hstack((w.transform(texts.iloc[va]), c.transform(texts.iloc[va])), format='csr')
    return (a, b, w, c)
records = []

def fit(name, a, b, penalty=None, C=1, ratio=None, solver='sgd', max_iter=25, tol=0.001):
    tic = time.time()
    kw = dict(loss='log_loss', penalty=penalty, alpha=0.0001 / C, max_iter=max_iter, tol=tol, random_state=42, average=False)
    if ratio is not None:
        kw['l1_ratio'] = ratio
    model = SGDClassifier(**kw).fit(a, y[tr])
    train_auc = roc_auc_score(y[tr], model.decision_function(a))
    val_auc = roc_auc_score(y[va], model.decision_function(b))
    record = dict(experiment=name, train_auc=train_auc, validation_auc=val_auc, seconds=time.time() - tic, iterations=int(model.n_iter_), n_features=a.shape[1], penalty=penalty or 'none', C=C, l1_ratio=ratio, solver='sgd', alpha=kw['alpha'])
    records.append(record)
    pd.DataFrame(records).to_csv('outputs/elastic_net_standalone_results.csv', index=False)
    print(record, flush=True)
    return model
print('Building baseline representation', flush=True)
a, b, w, c = build_text_features(text)
print('shape', a.shape, b.shape, flush=True)
fit('baseline_full', a, b)
for C in (1.0,):
    for ratio in (0.5,):
        fit(f'elasticnet_C{C}_ratio{ratio}_full', a, b, penalty='elasticnet', C=C, ratio=ratio, solver='sgd', max_iter=25, tol=0.001)
print('ALL DONE', flush=True)
