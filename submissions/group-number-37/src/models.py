"""model families compared in the experiments. each fit function returns text or row scorer."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .normalize import clean


def cleaned(texts):
    return [clean(t) for t in texts]


def fit_clean_lexical(texts, y, c=4.0, word_max=120000, char_max=200000, word=True, char=True, sample_weight=None, seed=3737):
    """tf-idf logistic regression on cleaned text, so formatting cannot be used."""
    vecs = []
    if word:
        vecs.append(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=word_max, sublinear_tf=True, dtype=np.float32))
    if char:
        vecs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=char_max, sublinear_tf=True,
                                    dtype=np.float32))
    t = cleaned(texts)
    x = hstack([v.fit_transform(t) for v in vecs], format="csr")
    clf = LogisticRegression(C=c, solver="liblinear", max_iter=500, random_state=seed).fit(x, y, sample_weight=sample_weight)

    def score(new):
        n = cleaned(new)
        return clf.decision_function(hstack([v.transform(n) for v in vecs], format="csr"))
    return score


def fit_gbm(x, y, iterations=500, depth=4, lr=.04, sample_weight=None, seed=3737, threads=8):
    """small gradient boosting on dense numeric features."""
    clf = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=lr, verbose=False, random_seed=seed,
                             allow_writing_files=False, thread_count=threads)
    clf.fit(x, y, sample_weight=sample_weight)
    return lambda new: clf.predict(new, prediction_type="RawFormulaVal")
