from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D = pd.read_csv('data/train.csv')
y = D.label.to_numpy()
tr, va = train_test_split(np.arange(len(y)), test_size=2000, stratify=y, random_state=42)
t = time.time()
v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=12000, sublinear_tf=True, dtype=np.float32)
a = v.fit_transform(D.text.iloc[tr])
b = v.transform(D.text.iloc[va])
z = TruncatedSVD(n_components=64, random_state=42, n_iter=3)
u = z.fit_transform(a)
w = z.transform(b)
sc = StandardScaler()
u = sc.fit_transform(u)
w = sc.transform(w)
p = PCA(n_components=32, random_state=42)
q = p.fit_transform(u)
r = p.transform(w)
print('representations ready', time.time() - t, flush=True)
out = []
for rep, xx, xy in [('SVD64_no_extra_PCA', u, w), ('SVD64_plus_PCA32', q, r)]:
    for name, params in [('Unregularized', dict(penalty=None, solver='lbfgs', max_iter=300)), ('ElasticNet_C1_l1ratio0.5', dict(penalty='elasticnet', solver='saga', C=1, l1_ratio=0.5, max_iter=100, tol=0.002)), ('L2_C1', dict(penalty='l2', solver='lbfgs', C=1, max_iter=300)), ('L1_C0.3', dict(penalty='l1', solver='liblinear', C=0.3, max_iter=300))]:
        for seed in [13, 42, 123]:
            model = LogisticRegression(**params, random_state=seed)
            t = time.time()
            model.fit(xx, y[tr])
            dur = time.time() - t
            row = dict(model=name, representation=rep, seed=seed, train_auc=roc_auc_score(y[tr], model.predict_proba(xx)[:, 1]), val_auc=roc_auc_score(y[va], model.predict_proba(xy)[:, 1]), train_seconds=round(dur, 2), iterations=int(model.n_iter_[0]))
            out.append(row)
            pd.DataFrame(out).to_csv('outputs/logistic_regularization_results.csv', index=False)
            print(row, flush=True)
