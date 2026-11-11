"""Original supervised 61K TF-IDF baseline, unregularized logistic regression."""
import argparse
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
p = argparse.ArgumentParser()
p.add_argument('--train', default='data/train.csv')
p.add_argument('--test', default='data/test.csv')
p.add_argument('--sample', default='data/sample_submission.csv')
p.add_argument('--output', default='submission_baseline.csv')
p.add_argument('--validation', action='store_true')
p.add_argument('--split-seed', type=int, default=42)
a = p.parse_args()
d = pd.read_csv(a.train)
y = d.label.to_numpy()
s = d.text.fillna('').astype(str)
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=26000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=35000, sublinear_tf=True, dtype=np.float32)
if a.validation:
    tr, va = train_test_split(np.arange(len(y)), test_size=2000, random_state=a.split_seed, stratify=y)
    A = hstack([w.fit_transform(s.iloc[tr]), c.fit_transform(s.iloc[tr])], format='csr')
    B = hstack([w.transform(s.iloc[va]), c.transform(s.iloc[va])], format='csr')
    m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=3000)
    m.fit(A, y[tr])
    print('train AUROC', roc_auc_score(y[tr], m.predict_proba(A)[:, 1]))
    print('validation AUROC', roc_auc_score(y[va], m.predict_proba(B)[:, 1]))
else:
    A = hstack([w.fit_transform(s), c.fit_transform(s)], format='csr')
    m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=3000)
    m.fit(A, y)
    test = pd.read_csv(a.test)
    z = test.text.fillna('').astype(str)
    probs = np.empty(len(test))
    for i in range(0, len(test), 100):
        B = hstack([w.transform(z.iloc[i:i + 100]), c.transform(z.iloc[i:i + 100])], format='csr')
        probs[i:i + 100] = m.predict_proba(B)[:, 1]
    out = pd.DataFrame({'id': test.id, 'label': probs})
    sample = pd.read_csv(a.sample)
    assert out.id.equals(sample.id) and np.isfinite(probs).all() and np.all((probs >= 0) & (probs <= 1))
    out.to_csv(a.output, index=False)
    print('saved', a.output)
