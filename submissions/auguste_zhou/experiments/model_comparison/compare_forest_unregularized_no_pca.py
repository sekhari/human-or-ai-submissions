from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os, time
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D = pd.read_csv('data/train.csv')
y = D.label.to_numpy()
tr, va = train_test_split(np.arange(len(D)), test_size=2000, stratify=y, random_state=42)
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=26000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=35000, sublinear_tf=True, dtype=np.float32)
t = time.time()
X = hstack([w.fit_transform(D.text.iloc[tr]), c.fit_transform(D.text.iloc[tr])], format='csr')
V = hstack([w.transform(D.text.iloc[va]), c.transform(D.text.iloc[va])], format='csr')
print('feature matrices', X.shape, V.shape, 'seconds', round(time.time() - t, 1), flush=True)
rows = []
for seed in [13, 42, 123]:
    for name, model in [('Unregularized Logistic Regression', LogisticRegression(penalty=None, solver='lbfgs', max_iter=3000, random_state=seed)), ('Random Forest', RandomForestClassifier(n_estimators=30, max_depth=12, min_samples_leaf=4, max_features=0.02, max_samples=0.8, n_jobs=4, random_state=seed))]:
        tic = time.time()
        model.fit(X, y[tr])
        train_auc = roc_auc_score(y[tr], model.predict_proba(X)[:, 1])
        val_auc = roc_auc_score(y[va], model.predict_proba(V)[:, 1])
        elapsed = time.time() - tic
        row = {'model': name, 'seed': seed, 'train_auc': train_auc, 'val_auc': val_auc, 'train_and_eval_seconds': round(elapsed, 2), 'features': 'word+char TF-IDF (61000 max), no PCA'}
        rows.append(row)
        print(row, flush=True)
pd.DataFrame(rows).to_csv('outputs/random_forest_vs_unregularized_no_pca.csv', index=False)
print('SUMMARY', pd.DataFrame(rows).groupby('model')[['train_auc', 'val_auc', 'train_and_eval_seconds']].agg(['mean', 'std']).round(6), flush=True)
