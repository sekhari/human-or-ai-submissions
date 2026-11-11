from pathlib import Path

# Run from the team folder; generated results stay separate from saved metrics.
Path('outputs').mkdir(exist_ok=True)

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
root = 'outputs'
tr = pd.read_csv('data/train.csv')
te = pd.read_csv('data/test.csv')
sm = pd.read_csv('data/sample_submission.csv')
text = tr.text.fillna('').astype(str)
test = te.text.fillna('').astype(str)
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=4000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
W = w.fit_transform(text)
Wt = w.transform(test)
print('word done', flush=True)
h = HashingVectorizer(analyzer='char', ngram_range=(3, 5), n_features=4096, alternate_sign=False, norm=None, dtype=np.float32)
C = h.transform(text)
Ct = h.transform(test)
print('char hashed', flush=True)
tf = TfidfTransformer(sublinear_tf=True)
C = tf.fit_transform(C)
Ct = tf.transform(Ct)
X = hstack([W, C], format='csr')
Xt = hstack([Wt, Ct], format='csr')
model = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=0.0001, l1_ratio=0.5, max_iter=25, tol=0.001, random_state=42, average=False)
model.fit(X, tr.label)
prob = model.predict_proba(Xt)[:, 1]
out = pd.DataFrame({'id': te.id, 'label': prob})
assert len(out) == len(sm) == 3000 and out.id.equals(sm.id) and np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all()
out.to_csv(root + '/submission_elastic_net.csv', index=False)
print('wrote', root + '/submission_elastic_net.csv', 'shape', X.shape, 'iters', model.n_iter_, flush=True)
print(out.head().to_string(index=False), flush=True)
