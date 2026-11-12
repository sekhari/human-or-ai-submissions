"""The 23 stylometric features used by every Group 26 model.

The feature definitions below are copied verbatim from Kevin's original
implementation (`kevin_train.py` in our development repository) and are
deliberately left unchanged: the reported AUROC figures depend on these exact
definitions, so re-deriving or "tidying" them would risk silently moving the
numbers. The only additions in this module are `FEATURE_NAMES` and
`featurize()`, which are convenience wrappers around `style_features()` and do
not alter any value it returns.

Note the non-ASCII characters in `CONTRACTION` and `PUNCTUATION` (curly
apostrophe U+2019, curly quotes U+201C/U+201D, em dash U+2014, en dash
U+2013). They are part of the feature definition; this file must stay UTF-8.

Feature order (23 total):
    0  mean sentence length, in words
    1  population standard deviation of sentence length
    2  mean word length, in characters
    3  root type-token ratio, len(vocab) / sqrt(n_words)
    4  hapax rate: share of word types occurring exactly once
    5  contraction rate per word
    6  newline rate per word
    7  blank-line rate per word
    8  uppercase character rate
    9  digit character rate
    10-22 per-character rate for each mark in PUNCTUATION
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean, pstdev

import numpy as np


WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
CONTRACTION = re.compile(r"\b(?:\w+n['’]t|\w+['’](?:m|re|ve|ll|d))\b", re.I)
PUNCTUATION = ",;:!?()\"“”—–-"


def style_features(text):
    words = WORD.findall(text.lower())
    count = max(1, len(words))
    chars = max(1, len(text))
    sentence_lengths = [len(WORD.findall(part)) for part in SENTENCE_BREAK.split(text)]
    sentence_lengths = [length for length in sentence_lengths if length] or [0]
    word_counts = Counter(words)
    punctuation = Counter(text)
    return [
        mean(sentence_lengths),
        pstdev(sentence_lengths),
        mean(map(len, words)) if words else 0,
        # Root TTR is less directly tied to length than raw type-token ratio.
        len(word_counts) / count**0.5,
        sum(n == 1 for n in word_counts.values()) / count,
        len(CONTRACTION.findall(text)) / count,
        text.count("\n") / count,
        len(re.findall(r"\n\s*\n", text)) / count,
        sum(c.isupper() for c in text) / chars,
        sum(c.isdigit() for c in text) / chars,
        *(punctuation[mark] / chars for mark in PUNCTUATION),
    ]


#: Human-readable name per column of `style_features`, in the same order.
_PUNCTUATION_NAMES = {
    ",": "comma", ";": "semicolon", ":": "colon", "!": "exclamation",
    "?": "question", "(": "open_paren", ")": "close_paren",
    '"': "straight_quote", "\u201c": "left_curly_quote",
    "\u201d": "right_curly_quote", "\u2014": "em_dash",
    "\u2013": "en_dash", "-": "hyphen",
}

FEATURE_NAMES: tuple[str, ...] = (
    "sentence_length_mean",
    "sentence_length_pstdev",
    "word_length_mean",
    "root_type_token_ratio",
    "hapax_rate",
    "contraction_rate",
    "newline_rate",
    "blank_line_rate",
    "uppercase_rate",
    "digit_rate",
    *(f"punct_{_PUNCTUATION_NAMES[mark]}_rate" for mark in PUNCTUATION),
)

N_FEATURES = len(FEATURE_NAMES)

#: Column indices kept by the feature-ablation experiment in `team_knn.py`:
#: everything except the newline rate (6), the blank-line rate (7) and the 13
#: punctuation rates (10-22). What remains is sentence shape, word shape and
#: vocabulary statistics, with all layout and punctuation signal removed.
#: Defined by Kevin alongside the original `style_features.py`; kept here so
#: the feature definitions have a single home.
FEATURES_WITHOUT_LAYOUT_PUNCTUATION = (0, 1, 2, 3, 4, 5, 8, 9)

#: The 10 scalar features (columns 0-9), dropping the 13 per-character
#: punctuation rates (10-22). Used by the shift-robust submission
#: (`predict.py --feature-set scalar`). Those 13 rates are mostly zero (e.g. the
#: em dash is zero for 96-99% of passages) and their test distribution differs
#: from train, so after standardization a handful of rare marks dominate the
#: Euclidean distance. See improvement.md.
FEATURES_SCALAR_ONLY = tuple(range(10))


def featurize(texts) -> np.ndarray:
    """Stack `style_features` over an iterable of passages.

    Returns a float array of shape (n_texts, 23) in FEATURE_NAMES order.
    """
    matrix = np.asarray([style_features(text) for text in texts], dtype=float)
    if matrix.size and matrix.shape[1] != N_FEATURES:
        raise AssertionError(
            f"expected {N_FEATURES} features per passage, got {matrix.shape[1]}"
        )
    return matrix
