"""Final detector: weighted P(AI) of word LR, character LR, stylometric GBDT and, optionally,
a punctuation-aware word 1-3-gram model with rare words masked (see distort.py)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.feature_extraction.text import TfidfVectorizer as V
from sklearn.linear_model import LogisticRegression as LR
from common import SEED
from distort import distort, top_words
from features import NAMES, normalize, stylometric_matrix

B = dict(sublinear_tf=True, strip_accents="unicode", lowercase=True)
FORMAT = [NAMES.index(n) for n in ("newline", "bullet")]

class Detector:
    def __init__(self, use_format=True, dist_weight=0.0, dist_k=10000, dist_c=30):
        self.cols = [i for i in range(len(NAMES)) if use_format or i not in FORMAT]
        self.dist_weight, self.dist_k, self.dist_c = dist_weight, dist_k, dist_c
    def fit(self, texts, y):
        n = [normalize(t) for t in texts]
        self.wv = V(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=.95, **B)
        self.cv = V(analyzer="char_wb", ngram_range=(2, 6), min_df=3, max_features=400000, **B)
        self.w = LR(C=30, max_iter=3000).fit(self.wv.fit_transform(n), y)
        self.c = LR(C=10, max_iter=3000).fit(self.cv.fit_transform(n), y)
        self.g = HGB(learning_rate=.1, max_leaf_nodes=31, max_iter=300, random_state=SEED).fit(stylometric_matrix(texts)[:, self.cols], y)
        if self.dist_weight > 0:
            self.keep = top_words(texts, self.dist_k)
            self.dv = V(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3), min_df=3, sublinear_tf=True, lowercase=False, max_features=300000)
            self.d = LR(C=self.dist_c, max_iter=3000).fit(self.dv.fit_transform([distort(t, self.keep) for t in texts]), y)
        return self
    def parts(self, texts):
        n = [normalize(t) for t in texts]
        return (self.w.predict_proba(self.wv.transform(n))[:, 1], self.c.predict_proba(self.cv.transform(n))[:, 1],
                self.g.predict_proba(stylometric_matrix(texts)[:, self.cols])[:, 1])
    def predict(self, texts):
        w, c, g = self.parts(texts)
        p = .25 * w + .25 * c + .5 * g
        if getattr(self, "dist_weight", 0) > 0:
            d = self.d.predict_proba(self.dv.transform([distort(t, self.keep) for t in texts]))[:, 1]
            p = (1 - self.dist_weight) * p + self.dist_weight * d
        return p
