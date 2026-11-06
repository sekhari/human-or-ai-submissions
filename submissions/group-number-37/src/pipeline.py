"""final model: reference lexical and style models, rank blended with a formatting-free surprisal model."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.sparse import hstack
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .baseline import style_matrix

SURPRISAL_WEIGHT = .3


class Blend:
    def __init__(self, seed=3737, surprisal_weight=SURPRISAL_WEIGHT, threads=8):
        self.seed, self.w, self.threads = seed, surprisal_weight, threads

    def _catboost(self, **kw):
        return CatBoostClassifier(verbose=False, random_seed=self.seed, allow_writing_files=False, thread_count=self.threads, **kw)

    def fit(self, texts, y, surprisal):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80000, sublinear_tf=True, strip_accents="unicode",
                                    dtype=np.float32)
        self.char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=100000, sublinear_tf=True,
                                    dtype=np.float32)
        x = hstack((self.word.fit_transform(texts), self.char.fit_transform(texts)), format="csr")
        self.lexical = LogisticRegression(C=8, solver="liblinear", max_iter=500, random_state=self.seed).fit(x, y)
        self.style = self._catboost(iterations=850, depth=6, learning_rate=.06).fit(style_matrix(texts), y)
        self.surprisal = self._catboost(iterations=500, depth=4, learning_rate=.04).fit(surprisal, y)
        return self

    def parts(self, texts, surprisal):
        x = hstack((self.word.transform(texts), self.char.transform(texts)), format="csr")
        lex = self.lexical.decision_function(x)
        sty = self.style.predict(style_matrix(texts), prediction_type="RawFormulaVal")
        sur = self.surprisal.predict(surprisal, prediction_type="RawFormulaVal")
        return dict(reference=.35 * lex + .65 * sty, surprisal=sur)

    def score(self, texts, surprisal):
        """values in (0, 1]. ranks within this batch, which is all auroc needs."""
        p = self.parts(texts, surprisal)
        return (1 - self.w) * rankdata(p["reference"]) / len(texts) + self.w * rankdata(p["surprisal"]) / len(texts)
