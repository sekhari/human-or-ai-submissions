"""C1: word (1-2)-gram + character n-gram TF-IDF (+ text-distorted word n-grams) -> L2 logistic regression.

The course baseline family (a word TF-IDF + LR reference gets ~0.868 private AUROC). Character n-grams
add sub-word style signal (suffixes, punctuation habits, homoglyph-broken words). Vocabulary and IDF are fit
on the training rows only (the sklearn regex tokenizer, no pretrained tokenizer).

Two options target the test shift (unseen topics and generators, far more verse; compare with train.py --cv topic):
- `char_analyzer: char` takes n-grams over the raw text, case kept, so line breaks, blank lines and spacing
  are features; `char_wb` (the first version) only sees characters inside words.
- `distort_k` > 0 adds word (1-`distort_ngrams`)-grams over text-distorted input (features/distortion.py:
  words outside the k most frequent training words masked), i.e. function-word and punctuation patterns
  without topic. Line breaks are kept as tokens.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from features.distortion import Distorter

from .base import BaseModel


class TfidfLR(BaseModel):
    name = "tfidf_lr"

    def __init__(self, C: float = 4.0, word_ngrams=(1, 2), char_ngrams=(2, 5), word_min_df: int = 2,
                 char_min_df: int = 3, max_features: int = 300000, lowercase: bool = True,
                 char_analyzer: str = "char_wb", distort_k: int = 0, distort_ngrams: int = 3, seed: int = 0):
        self.C, self.seed = C, seed
        self.word = TfidfVectorizer(analyzer="word", ngram_range=tuple(word_ngrams), min_df=word_min_df,
                                    max_features=max_features, sublinear_tf=True, lowercase=lowercase)
        self.char = TfidfVectorizer(analyzer=char_analyzer, ngram_range=tuple(char_ngrams), min_df=char_min_df,
                                    max_features=max_features, sublinear_tf=True,
                                    lowercase=lowercase if char_analyzer == "char_wb" else False)
        self.distorter = Distorter(distort_k) if distort_k else None
        self.dist = TfidfVectorizer(analyzer="word", token_pattern=r"[^ \t\n]+|\n", ngram_range=(1, distort_ngrams),
                                    min_df=word_min_df, max_features=max_features, sublinear_tf=True,
                                    lowercase=False) if distort_k else None

    def _X(self, texts, fit=False):
        f = (lambda v, t: v.fit_transform(t)) if fit else (lambda v, t: v.transform(t))
        mats = [f(self.word, texts), f(self.char, texts)]
        if self.distorter:
            if fit:
                self.distorter.fit(texts)
            mats.append(f(self.dist, self.distorter.transform(texts)))
        return sp.hstack(mats).tocsr()

    def fit(self, texts, y, unlabeled_texts=None):
        X = self._X(texts, fit=True)
        self.clf = LogisticRegression(C=self.C, solver="liblinear", random_state=self.seed).fit(X, y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._X(texts))[:, 1].astype(np.float64)
