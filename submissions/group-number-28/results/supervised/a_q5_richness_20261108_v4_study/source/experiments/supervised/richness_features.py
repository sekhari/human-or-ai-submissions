"""Document-local length and lexical features for the frozen Q5 ablation.

These extractors do not learn from a corpus. Any scaling belongs in the fitted
training/calibration pipeline, after extraction. Tokens are lowercase Unicode
letter runs bounded as words; numbers, underscores and mixed alphanumeric
tokens are excluded by the frozen regular expression.
"""
from collections import Counter
import math
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


TOKEN_PATTERN = r"(?u)\b[^\W\d_]+\b"
_TOKENIZER = re.compile(TOKEN_PATTERN)


def tokenize(text):
    """Return the frozen document-local lowercase letter tokens."""
    if not isinstance(text, str):
        raise TypeError("Document features require string inputs")
    return _TOKENIZER.findall(text.lower())


class LengthFeatures(TransformerMixin, BaseEstimator):
    """Return log1p(letter-token count) and log1p(Unicode codepoint count)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [(math.log1p(len(tokenize(text))), math.log1p(len(text))) for text in X]
        return np.asarray(rows, dtype=np.float64).reshape(-1, 2)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["log1p_word_count", "log1p_character_count"], dtype=object)


class RichnessFeatures(TransformerMixin, BaseEstimator):
    """Return moving type diversity, normalized entropy and bigram repetition.

    MATTR uses every contiguous window of min(window_size, token_count) tokens.
    Short documents therefore have one window; empty documents return zero.
    Entropy is -sum(p*log(p))/log(token_count), with zero for fewer than two
    tokens. Bigram repetition is (number of bigrams - unique bigrams) divided
    by number of bigrams, with zero when no bigrams exist.
    """

    def __init__(self, window_size=50):
        self.window_size = window_size

    def _validate_window(self):
        if (isinstance(self.window_size, bool)
                or not isinstance(self.window_size, (int, np.integer))
                or self.window_size < 1):
            raise ValueError("window_size must be a positive integer")

    def fit(self, X, y=None):
        self._validate_window()
        return self

    def _document(self, text):
        tokens = tokenize(text)
        count = len(tokens)
        if count == 0:
            return 0.0, 0.0, 0.0
        width = min(self.window_size, count)
        window = Counter(tokens[:width])
        diversity_sum = len(window)
        for position in range(width, count):
            outgoing = tokens[position - width]
            window[outgoing] -= 1
            if window[outgoing] == 0:
                del window[outgoing]
            window[tokens[position]] += 1
            diversity_sum += len(window)
        mattr = diversity_sum / ((count - width + 1) * width)
        if count == 1:
            return mattr, 0.0, 0.0
        frequencies = Counter(tokens)
        entropy = -sum((frequency / count) * math.log(frequency / count)
                       for frequency in frequencies.values()) / math.log(count)
        # Bound harmless floating-point rounding at the theoretical endpoints.
        entropy = min(1.0, max(0.0, entropy))
        n_bigrams = count - 1
        repetition = (n_bigrams - len(set(zip(tokens, tokens[1:])))) / n_bigrams
        return mattr, entropy, repetition

    def transform(self, X):
        self._validate_window()
        return np.asarray([self._document(text) for text in X], dtype=np.float64).reshape(-1, 3)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"mattr_window_{self.window_size}",
                           "token_entropy_over_log_count", "repeated_bigram_fraction"], dtype=object)
