"""Hand-crafted stylometric features (length, punctuation, vocabulary, rhythm).

These describe *how* a passage is written rather than *which words* it uses,
so they give a model family that is very different from TF-IDF.
"""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

PUNCT = [",", ".", ";", ":", "!", "?", "-", "'", '"', "(", "—"]
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "it", "this", "that", "i", "you", "we",
    "they", "he", "she", "not", "as", "at", "by", "from",
}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z']+")

FEATURE_NAMES = (
    ["n_chars", "n_words", "avg_word_len", "std_word_len",
     "n_sentences", "avg_sent_len", "std_sent_len", "max_sent_len",
     "type_token_ratio", "hapax_ratio", "function_word_ratio",
     "uppercase_ratio", "digit_ratio", "newline_ratio", "long_word_ratio"]
    + [f"punct_{p}" for p in PUNCT]
)


def style_vector(text):
    words = WORD_RE.findall(text)
    lower = [w.lower() for w in words]
    n_chars = max(len(text), 1)
    n_words = max(len(words), 1)

    word_lens = np.array([len(w) for w in words]) if words else np.array([0])
    sents = [s for s in SENT_SPLIT.split(text.strip()) if s]
    sent_lens = np.array([len(WORD_RE.findall(s)) for s in sents]) if sents else np.array([0])

    counts = {}
    for w in lower:
        counts[w] = counts.get(w, 0) + 1

    feats = [
        len(text),
        len(words),
        word_lens.mean(),
        word_lens.std(),
        len(sents),
        sent_lens.mean(),
        sent_lens.std(),
        sent_lens.max(),
        len(counts) / n_words,
        sum(1 for c in counts.values() if c == 1) / n_words,
        sum(1 for w in lower if w in FUNCTION_WORDS) / n_words,
        sum(ch.isupper() for ch in text) / n_chars,
        sum(ch.isdigit() for ch in text) / n_chars,
        text.count("\n") / n_words,
        float((word_lens >= 8).mean()),
    ]
    feats += [text.count(p) / n_words for p in PUNCT]
    return feats


class StyleFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer: list of texts -> (n, n_features) array."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([style_vector(t) for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
