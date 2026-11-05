"""Find which stylometric features drive the very high GBDT score, and whether they hold on test."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score as auc
from common import SEED, load_train, split
from features import NAMES, stylometric_matrix
tr, va = split(load_train("data/train.csv")); y, yv = tr.label.values, va.label.values
S, V = stylometric_matrix(tr.text), stylometric_matrix(va.text)
te = pd.read_csv("data/test.csv"); un = pd.read_csv("data/unlabeled.csv"); T, U = stylometric_matrix(te.text), stylometric_matrix(un.text)
mk = lambda: HGB(learning_rate=.1, max_leaf_nodes=31, max_iter=300, random_state=SEED)
r = []
for i, n in enumerate(NAMES):
    u = auc(yv, V[:, i]); keep = [j for j in range(len(NAMES)) if j != i]
    r.append((n, round(max(u, 1 - u), 3), round(auc(yv, mk().fit(S[:, keep], y).predict_proba(V[:, keep])[:, 1]), 4)))
print(pd.DataFrame(r, columns=["f", "single_auc", "val_without"]).sort_values("single_auc", ascending=False).to_string())
for n in ("n_chars", "n_words", "word_len"):
    i = NAMES.index(n); print(n, "train h/a", np.percentile(S[y == 0, i], [1, 50, 99]), np.percentile(S[y == 1, i], [1, 50, 99]))
print("n_words mod10 AI share in train:"); w = S[:, 1].astype(int); print(pd.Series(y).groupby(w // 25 * 25).agg(["mean", "size"]).round(2).T.to_string())
w0 = pd.Series(tr.text.str.split().str.len()); print("exact word counts top:", w0.value_counts().head(8).to_dict())
print(pd.DataFrame({"mean": y}).groupby(w0.values)["mean"].agg(["mean", "size"]).sort_values("size", ascending=False).head(8).round(2).T.to_string())
p = mk().fit(S, y)
for n, X in (("val", V), ("unl", U), ("test", T)): print(n, "P(AI) quantiles", np.percentile(p.predict_proba(X)[:, 1], [5, 25, 50, 75, 95]).round(3))
