from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os, time, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '2'
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
D = pd.read_csv('data/train.csv')
T = pd.read_csv('data/test.csv')
y = D.label.values
tr, va = train_test_split(np.arange(len(y)), test_size=2000, random_state=42, stratify=y)

def vecfit(txt, word, char):
    return hstack([word.fit_transform(txt), char.fit_transform(txt)], format='csr')
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=26000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=35000, sublinear_tf=True, dtype=np.float32)
a = vecfit(D.text.iloc[tr], w, c)
b = hstack([w.transform(D.text.iloc[va]), c.transform(D.text.iloc[va])], format='csr')
for name, m in [('L2_C1', LogisticRegression(C=1, solver='liblinear', max_iter=1500)), ('L2_C4', LogisticRegression(C=4, solver='liblinear', max_iter=1500)), ('L1_C0.3', LogisticRegression(C=0.3, penalty='l1', solver='liblinear', max_iter=1500)), ('SVM_C0.3', LinearSVC(C=0.3))]:
    st = time.time()
    m.fit(a, y[tr])
    s = m.decision_function(b) if name.startswith('SVM') else m.predict_proba(b)[:, 1]
    tt = m.decision_function(a) if name.startswith('SVM') else m.predict_proba(a)[:, 1]
    print(name, round(roc_auc_score(y[tr], tt), 6), round(roc_auc_score(y[va], s), 6), 'sec', round(time.time() - st, 2), flush=True)
