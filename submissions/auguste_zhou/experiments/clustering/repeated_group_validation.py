from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
import time, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
D = pd.read_csv('data/train.csv')
y = D.label.to_numpy()
s = D.text.fillna('').astype(str)
base, official_val = train_test_split(np.arange(len(y)), test_size=2000, random_state=42, stratify=y)
zvec = TfidfVectorizer(min_df=3, max_features=2500, stop_words='english', dtype=np.float32)
Z = zvec.fit_transform(s.iloc[base])
cluster = MiniBatchKMeans(n_clusters=24, random_state=42, batch_size=256, n_init=3, max_iter=80)
groups = cluster.fit_predict(Z)
print('cluster sizes', np.bincount(groups), flush=True)
results = []
# Earlier seed 42 results are not required to run these additional splits.
for seed in [13, 123, 7]:
    choices = []
    for tr, va in GroupShuffleSplit(n_splits=120, test_size=0.2, random_state=seed).split(base, y[base], groups):
        if len(np.unique(y[base][tr])) < 2 or len(np.unique(y[base][va])) < 2:
            continue
        score = abs(len(va) / 8000 - 0.2) + 1.5 * abs(y[base][va].mean() - 0.5)
        choices.append((score, tr, va))
    choices.sort(key=lambda a: a[0])
    _, gtr, gva = choices[0]
    rtr, rva = train_test_split(np.arange(len(base)), test_size=len(gva), random_state=seed, stratify=y[base])
    print('seed', seed, 'holdout', len(gva), 'pos', y[base][gva].mean(), flush=True)
    for typ, tr, va in [('random', base[rtr], base[rva]), ('group_aware', base[gtr], base[gva])]:
        tick = time.perf_counter()
        w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=4000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
        A = w.fit_transform(s.iloc[tr])
        B = w.transform(s.iloc[va])
        h = HashingVectorizer(analyzer='char', ngram_range=(3, 5), n_features=4096, alternate_sign=False, norm=None, dtype=np.float32)
        C = h.transform(s.iloc[tr])
        E = h.transform(s.iloc[va])
        tf = TfidfTransformer(sublinear_tf=True)
        C = tf.fit_transform(C)
        E = tf.transform(E)
        X = hstack([A, C], format='csr')
        V = hstack([B, E], format='csr')
        m = LogisticRegression(penalty=None, max_iter=3000, solver='lbfgs')
        m.fit(X, y[tr])
        d = {'seed': seed, 'holdout_type': typ, 'train_n': len(tr), 'validation_n': len(va), 'positive_fraction': float(y[va].mean()), 'train_auc': roc_auc_score(y[tr], m.decision_function(X)), 'val_auc': roc_auc_score(y[va], m.decision_function(V)), 'runtime_s': time.perf_counter() - tick}
        results.append(d)
        pd.DataFrame(results).to_csv('outputs/repeated_group_validation_results.csv', index=False)
        print(d, flush=True)
print(pd.DataFrame(results).groupby('holdout_type').val_auc.agg(['mean', 'std', 'min', 'max']), flush=True)
