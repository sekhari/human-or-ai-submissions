"""Text normalisation, hand-crafted stylometric features and model factories."""

from __future__ import annotations

import re
import string
import unicodedata

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------------------
# Normalisation (used by the robustness improvement, Q5)
# --------------------------------------------------------------------------------------
ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
# Small confusables table: Cyrillic/Greek letters that look like Latin ones.
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l", "А": "A", "В": "B", "Е": "E", "К": "K",
    "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "ο": "o", "α": "a",
    "ν": "v", "ι": "i", "Ο": "O", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y",
})
MIXED_WORD = re.compile(r"\b\w*[A-Za-z]\w*\b")


def _fix_mixed_script(text: str) -> str:
    """Map look-alike letters to Latin only inside words that already contain Latin letters,
    so genuine Greek/Cyrillic text is left untouched."""
    def repl(m):
        w = m.group(0)
        return w.translate(HOMOGLYPHS) if not w.isascii() else w
    return MIXED_WORD.sub(repl, text)


def normalize(text: str) -> str:
    """NFKC, strip zero-width chars, undo homoglyphs inside Latin words, collapse spaces."""
    t = unicodedata.normalize("NFKC", text)
    t = ZERO_WIDTH.sub("", t)
    t = _fix_mixed_script(t)
    t = re.sub(r"[ \t]+", " ", t)
    return t


def normalize_all(texts) -> list[str]:
    return [normalize(t) for t in texts]


# --------------------------------------------------------------------------------------
# Stylometric features (Model C)
# --------------------------------------------------------------------------------------
STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been
being it its this that these those i you he she we they me him her us them my your his our their
not no so than too very can will just do does did have has had""".split())
WORD = re.compile(r"[A-Za-z']+|\w+")
SENT = re.compile(r"(?<=[.!?])\s+")
PUNCT = set(string.punctuation)

STYLE_NAMES = [
    "n_chars", "n_words", "log_words", "avg_word_len", "std_word_len", "type_token_ratio",
    "hapax_ratio", "stopword_ratio", "n_sents", "avg_sent_len", "std_sent_len", "max_sent_len",
    "punct_ratio", "comma_per_word", "period_per_word", "excl_per_word", "quest_per_word",
    "colon_semicolon_per_word", "quote_ratio", "paren_ratio", "dash_ratio", "upper_ratio",
    "digit_ratio", "newline_per_word", "blank_line_per_word", "double_space_per_char",
    "nonascii_ratio", "zero_width_count", "mixed_script_words", "rep_bigram_ratio",
    "rep_trigram_ratio", "max_word_repeat_frac", "starts_upper_ratio", "first_person_ratio",
    "contraction_ratio", "ellipsis_count", "curly_quote_ratio", "weird_case_words",
    "sent_start_repeat",
]


def style_vector(text: str) -> list[float]:
    n_chars = len(text)
    words = WORD.findall(text)
    nw = max(1, len(words))
    lw = [w.lower() for w in words]
    wl = np.array([len(w) for w in words]) if words else np.array([0])
    counts: dict[str, int] = {}
    for w in lw:
        counts[w] = counts.get(w, 0) + 1
    sents = [s for s in SENT.split(text.strip()) if s]
    sl = np.array([len(WORD.findall(s)) for s in sents]) if sents else np.array([0])
    bigr = list(zip(lw, lw[1:]))
    trigr = list(zip(lw, lw[1:], lw[2:]))
    rep2 = 1 - len(set(bigr)) / max(1, len(bigr))
    rep3 = 1 - len(set(trigr)) / max(1, len(trigr))
    alpha = sum(c.isalpha() for c in text) or 1
    first_tokens = [s.split()[0].lower() for s in sents if s.split()]
    return [
        n_chars, len(words), np.log1p(len(words)), wl.mean(), wl.std(),
        len(counts) / nw, sum(1 for c in counts.values() if c == 1) / nw,
        sum(1 for w in lw if w in STOP) / nw, len(sents), sl.mean(), sl.std(), sl.max(),
        sum(c in PUNCT for c in text) / max(1, n_chars), text.count(",") / nw,
        text.count(".") / nw, text.count("!") / nw, text.count("?") / nw,
        (text.count(":") + text.count(";")) / nw, (text.count('"') + text.count("'")) / nw,
        (text.count("(") + text.count(")")) / nw,
        (text.count("-") + text.count("–") + text.count("—")) / nw,
        sum(c.isupper() for c in text) / alpha, sum(c.isdigit() for c in text) / max(1, n_chars),
        text.count("\n") / nw, text.count("\n\n") / nw, text.count("  ") / max(1, n_chars),
        sum(ord(c) > 127 for c in text) / max(1, n_chars), len(ZERO_WIDTH.findall(text)),
        sum(1 for w in words if not w.isascii() and any(c.isascii() and c.isalpha() for c in w)),
        rep2, rep3, max(counts.values()) / nw if counts else 0,
        sum(1 for s in sents if s[:1].isupper()) / max(1, len(sents)),
        sum(1 for w in lw if w in {"i", "me", "my", "mine", "we", "our"}) / nw,
        sum(1 for w in lw if "'" in w) / nw, text.count("...") + text.count("…"),
        sum(text.count(q) for q in "“”‘’") / max(1, n_chars),
        sum(1 for w in words if re.search(r"[a-z][A-Z]", w)),
        1 - len(set(first_tokens)) / max(1, len(first_tokens)),
    ]


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Turns raw texts into a dense matrix of stylometric features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([style_vector(t) for t in X], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(STYLE_NAMES)


class Normalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return normalize_all(X)


def hstack_dense(blocks):
    return sparse.hstack(blocks).tocsr()
