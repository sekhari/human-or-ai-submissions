"""Shift-aware model comparison (exploratory, for improving the Kaggle model).

Why: the random 2,000-row validation set scored 0.966 but Kaggle public scored 0.868.
Test passages use far more unseen vocabulary (4.0% OOV vs 1.3% within train), i.e. new
topics/domains. A random split hides that, so here every model is scored by
"leave-topic-out" CV: train passages are clustered by topic (TF-IDF -> SVD -> KMeans)
and each fold holds out whole clusters. No test labels are used anywhere.

Optional transductive variant: vectorizers can be fit (unsupervised) on train +
unlabeled + test text so the vocabulary/IDF covers test-domain words.
"""
import sys, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from common import SEED, normalize_ws
from q1_model_comparison import StyleFeatures

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv"); un = pd.read_csv("data/unlabeled.csv")
X = normalize_ws(tr.text.values); y = tr.label.values
EXTRA = normalize_ws(np.concatenate([un.text.values, te.text.values]))

# ---- topic clusters (content words only, so clusters = topics, not style)
tv = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5, sublinear_tf=True)
Z = TruncatedSVD(100, random_state=SEED).fit_transform(tv.fit_transform(X))
Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
groups = KMeans(12, n_init=5, random_state=SEED).fit_predict(Z)
print("cluster sizes", np.bincount(groups), "AI share", np.round([y[groups == g].mean() for g in range(12)], 2))
folds = list(GroupKFold(n_splits=6).split(X, y, groups))

WORD = dict(lowercase=True, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]")


def make(name):
    """Return (vectorizer(s), classifier) for a candidate; vectorizer is fit per fold."""
    if name.startswith("word"):
        C = float(name.split("C")[1])
        return [TfidfVectorizer(ngram_range=(1, 2), min_df=5, **WORD)], LogisticRegression(C=C, max_iter=3000, solver="liblinear")
    if name.startswith("char"):
        C = float(name.split("C")[1])
        return [TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), lowercase=False, sublinear_tf=True,
                                min_df=3, max_features=300_000)], LogisticRegression(C=C, max_iter=3000, solver="liblinear")
    if name.startswith("func"):  # function words + punctuation only: topic-agnostic by construction
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        voc = sorted(ENGLISH_STOP_WORDS | set(list(".,;:!?'\"()-—–…")))
        return [TfidfVectorizer(vocabulary=None, ngram_range=(1, 3), min_df=5, sublinear_tf=True, lowercase=True,
                                token_pattern=r"(?u)\b\w+\b|[^\w\s]",
                                preprocessor=lambda t: " ".join(w if w in voc else "X" for w in
                                    __import__("re").findall(r"\w+|[^\w\s]", t.lower())))], \
               LogisticRegression(C=float(name.split("C")[1]), max_iter=3000, solver="liblinear")
    raise ValueError(name)


def fit_predict(name, Xa, ya, Xb, transductive=False):
    if name == "style":
        sf = StyleFeatures(); m = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=500, random_state=SEED, early_stopping=False)
        m.fit(sf.transform(Xa), ya); return m.predict_proba(sf.transform(Xb))[:, 1]
    vecs, clf = make(name)
    v = vecs[0]
    v.fit(np.concatenate([Xa, EXTRA]) if transductive else Xa)
    clf.fit(v.transform(Xa), ya)
    return clf.predict_proba(v.transform(Xb))[:, 1]


def evaluate(names, transductive=False):
    oof = {n: np.zeros(len(y)) for n in names}
    for a, b in folds:
        for n in names:
            oof[n][b] = fit_predict(n, X[a], y[a], X[b], transductive)
    return oof


def report(oof, tag=""):
    for n, p in oof.items():
        per = [roc_auc_score(y[groups == g], p[groups == g]) for g in range(12) if len(set(y[groups == g])) == 2]
        print(f"{tag}{n:14s} pooled={roc_auc_score(y, p):.4f}  mean-cluster={np.mean(per):.4f}  worst={np.min(per):.4f}")


if __name__ == "__main__":
    t0 = time.time()
    names = ["wordC30", "wordC3", "wordC1", "charC3", "charC10", "funcC1", "style"]
    oof = evaluate(names); report(oof)
    r = {n: rankdata(p) / len(p) for n, p in oof.items()}
    ens = {"ens word+char": (r["wordC3"] + r["charC10"]) / 2,
           "ens word+char+func": (r["wordC3"] + r["charC10"] + r["funcC1"]) / 3,
           "ens word+char+style": (r["wordC3"] + r["charC10"] + r["style"]) / 3,
           "ens all4": (r["wordC3"] + r["charC10"] + r["funcC1"] + r["style"]) / 4}
    report(ens)
    oof_t = evaluate(["wordC3", "charC10"], transductive=True); report(oof_t, "transd ")
    pd.DataFrame({**oof, "label": y, "group": groups}).to_csv("results_shift_oof.csv", index=False)
    print(f"{time.time()-t0:.0f}s")
