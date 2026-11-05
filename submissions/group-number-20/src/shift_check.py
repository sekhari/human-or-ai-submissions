"""Measure train/unlabeled/test distribution shift from text features only (no labels)."""
import json, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from common import SEED
t, u, e = [pd.read_csv(f"data/{n}.csv") for n in ("train", "unlabeled", "test")]
def auc(a, b):
    x = pd.concat([a.text, b.text]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    m = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2), LogisticRegression(max_iter=1000))
    p = cross_val_predict(m, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)
r = {"train_vs_test": auc(t, e), "train_vs_unlabeled": auc(t, u), "unlabeled_vs_test": auc(u, e)}
json.dump(r, open("results/shift.json", "w"), indent=2); print(r)
