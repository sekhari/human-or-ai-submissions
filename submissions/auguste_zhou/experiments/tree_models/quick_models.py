from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os, time, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
D = pd.read_csv('data/train.csv')
y = D.label.values
tr, va = train_test_split(np.arange(len(D)), test_size=2000, stratify=y, random_state=42)
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
print('ready', round(time.time() - t, 1), flush=True)
out = []
for name in ['L2_Logistic', 'L1_Logistic', 'Linear_SVM', 'Random_Forest', 'XGBoost']:
    for rep, xx, xy in [('no_PCA', u, w), ('PCA_32', q, r)]:
        for seed in [13, 42, 123]:
            if name == 'L2_Logistic':
                m = LogisticRegression(C=1, solver='liblinear', random_state=seed)
            elif name == 'L1_Logistic':
                m = LogisticRegression(penalty='l1', C=0.15, solver='liblinear', random_state=seed)
            elif name == 'Linear_SVM':
                m = LinearSVC(C=0.1, random_state=seed)
            elif name == 'Random_Forest':
                m = RandomForestClassifier(n_estimators=40, max_depth=12, min_samples_leaf=4, n_jobs=2, random_state=seed)
            else:
                m = XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.09, n_jobs=2, random_state=seed, tree_method='hist')
            start = time.time()
            m.fit(xx, y[tr])
            sec = time.time() - start
            pred = m.decision_function(xy) if name == 'Linear_SVM' else m.predict_proba(xy)[:, 1]
            ptr = m.decision_function(xx) if name == 'Linear_SVM' else m.predict_proba(xx)[:, 1]
            row = dict(model=name, features=rep, seed=seed, train_auc=roc_auc_score(y[tr], ptr), val_auc=roc_auc_score(y[va], pred), seconds=round(sec, 2))
            out.append(row)
            pd.DataFrame(out).to_csv('outputs/five_model_pca_seed_comparison.csv', index=False)
            print(row, flush=True)
print('done', round(time.time() - t, 1), flush=True)
