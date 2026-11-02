"""Final detector: mean P(AI) of word LR, character LR, and stylometric GBDT."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.feature_extraction.text import TfidfVectorizer as V
from sklearn.linear_model import LogisticRegression as LR
from common import SEED
from features import NAMES, normalize, stylometric_matrix

B = dict(sublinear_tf=True, strip_accents="unicode", lowercase=True)
FORMAT = [NAMES.index(n) for n in ("newline", "bullet")]

class Detector:
    def __init__(self, use_format=True):
        self.cols = [i for i in range(len(NAMES)) if use_format or i not in FORMAT]
    def fit(self, texts, y):
        n = [normalize(t) for t in texts]
        self.wv = V(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=.95, **B)
        self.cv = V(analyzer="char_wb", ngram_range=(2, 6), min_df=3, max_features=400000, **B)
        self.w = LR(C=30, max_iter=3000).fit(self.wv.fit_transform(n), y)
        self.c = LR(C=10, max_iter=3000).fit(self.cv.fit_transform(n), y)
        self.g = HGB(learning_rate=.1, max_leaf_nodes=31, max_iter=300, random_state=SEED).fit(stylometric_matrix(texts)[:, self.cols], y)
        return self
    def parts(self, texts):
        n = [normalize(t) for t in texts]
        return (self.w.predict_proba(self.wv.transform(n))[:, 1], self.c.predict_proba(self.cv.transform(n))[:, 1],
                self.g.predict_proba(stylometric_matrix(texts)[:, self.cols])[:, 1])
    def predict(self, texts):
        w, c, g = self.parts(texts)
        return .25 * w + .25 * c + .5 * g
