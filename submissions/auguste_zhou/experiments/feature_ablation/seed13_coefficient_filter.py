from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import time, numpy as np, pandas as pd
from scipy.sparse import hstack
from scipy.special import expit
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
p = 'outputs/'
d = pd.read_csv('data/train.csv')
y = d.label.to_numpy()
s = d.text.fillna('').astype(str)
base, _ = train_test_split(np.arange(len(d)), test_size=2000, random_state=42, stratify=y)
z = TfidfVectorizer(min_df=3, max_features=2500, stop_words='english', dtype=np.float32).fit_transform(s.iloc[base])
grp = MiniBatchKMeans(n_clusters=24, random_state=42, batch_size=256, n_init=3, max_iter=80).fit_predict(z)
choices = []
for a, b in GroupShuffleSplit(n_splits=120, test_size=0.2, random_state=13).split(base, y[base], grp):
    if len(np.unique(y[base][a])) < 2 or len(np.unique(y[base][b])) < 2:
        continue
    score = abs(len(b) / 8000 - 0.2) + 1.5 * abs(y[base][b].mean() - 0.5)
    choices.append((score, a, b))
_, a, b = min(choices, key=lambda q: q[0])
tr = base[a]
va = base[b]
print('SPLIT', len(tr), len(va), float(y[va].mean()), flush=True)
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=4000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
a = w.fit_transform(s.iloc[tr])
b = w.transform(s.iloc[va])
print('WORD READY', flush=True)
h = HashingVectorizer(analyzer='char', ngram_range=(3, 5), n_features=4096, alternate_sign=False, norm=None, dtype=np.float32)
c = h.transform(s.iloc[tr])
e = h.transform(s.iloc[va])
tf = TfidfTransformer(sublinear_tf=True)
c = tf.fit_transform(c)
e = tf.transform(e)
x = hstack([a, c], format='csr')
v = hstack([b, e], format='csr')
print('FEATURES READY', x.shape, flush=True)
m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=3000)
m.fit(x, y[tr])
coef = m.coef_.ravel()
intercept = float(m.intercept_[0])
print('FIT READY', int(m.n_iter_[0]), 'norm', np.linalg.norm(coef), flush=True)
rows = []

def score(name, c):
    t = roc_auc_score(y[tr], x @ c + intercept)
    q = roc_auc_score(y[va], v @ c + intercept)
    rows.append(dict(setting=name, train_auc=t, group_val_auc=q, delta_vs_baseline=None, nonzero=int(np.count_nonzero(c)), max_abs_weight=float(np.max(np.abs(c)))))
    print('RESULT', name, round(t, 6), round(q, 6), flush=True)
score('baseline', coef)
for pct in (95, 90, 80, 70, 50):
    cap = np.percentile(np.abs(coef), pct)
    score('clip_p' + str(pct), np.clip(coef, -cap, cap))
for fac in (0.75, 0.5, 0.25, 0):
    c = coef.copy()
    k = 20
    idx = np.argpartition(np.abs(c), -k)[-k:]
    c[idx] *= fac
    score(f'top20_scale_{fac}', c)
for k in (20, 50, 100):
    c = coef.copy()
    idx = np.argpartition(np.abs(c), -k)[-k:]
    c[idx] = 0
    score(f'top{k}_removed', c)
result = pd.DataFrame(rows)
result['delta_vs_baseline'] = result.group_val_auc - result.group_val_auc.iloc[0]
result.to_csv(p + 'seed13_coefficient_filter_results.csv', index=False)
import joblib
joblib.dump(dict(model=m, word=w, hash=h, tf=tf, train=tr, validation=va, coefficients=coef, results=result), p + 'seed13_coefficient_filter_bundle.joblib', compress=3)
print('DONE', flush=True)
