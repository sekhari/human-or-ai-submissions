"""A small word n-gram language model, used as a much simpler reference than GPT-2.

Fit it on unlabeled.csv only: every labeled and test document is then out-of-sample, so the features
do not behave differently between training rows and the rows being scored.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

from src.features import WORD_RE

BIGRAM_WEIGHT = 0.7
SMOOTHING = 0.5


def _tokens(text):
    return [w.lower() for w in WORD_RE.findall(text)]


class NgramModel:
    def __init__(self, texts):
        self.unigrams, self.bigrams = Counter(), Counter()
        for text in texts:
            words = _tokens(text)
            self.unigrams.update(words)
            self.bigrams.update(zip(words, words[1:]))
        self.total = sum(self.unigrams.values())
        self.vocab = len(self.unigrams) + 1

    def _unigram_prob(self, word):
        return (self.unigrams[word] + SMOOTHING) / (self.total + SMOOTHING * self.vocab)

    def _bigram_nll(self, prev, word):
        seen = self.unigrams[prev]
        conditional = self.bigrams[(prev, word)] / seen if seen else 0.0
        return -math.log(BIGRAM_WEIGHT * conditional + (1 - BIGRAM_WEIGHT) * self._unigram_prob(word))

    def features(self, text):
        words = _tokens(text)
        if len(words) < 2:
            return dict.fromkeys(("ngram_uni_nll", "ngram_bi_nll", "ngram_bi_std", "ngram_bi_q90", "ngram_gain", "ngram_oov"), 0.0)
        uni = np.array([-math.log(self._unigram_prob(w)) for w in words])
        bi = np.array([self._bigram_nll(p, w) for p, w in zip(words, words[1:])])
        return {
            "ngram_uni_nll": uni.mean(), "ngram_bi_nll": bi.mean(), "ngram_bi_std": bi.std(),
            "ngram_bi_q90": np.quantile(bi, 0.9), "ngram_gain": uni.mean() - bi.mean(),
            "ngram_oov": np.mean([w not in self.unigrams for w in words]),
        }


def ngram_features(fit_texts, texts):
    model = NgramModel(fit_texts)
    return pd.DataFrame([model.features(t) for t in texts])
