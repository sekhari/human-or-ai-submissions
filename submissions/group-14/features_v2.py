"""Blank-line-aware normalization (v2) and stylometric features.

`features.normalize_text` maps every run of line breaks to a single `¶`, so a
blank line ("\\n\\n") and a single line break become identical. In train.csv a
blank line appears in 66.3% of AI passages but only 0.4% of human ones, so v2
emits one `¶` per line break. Every other normalization step matches v1, and
`features.py` is left unchanged so the SVM results stay reproducible.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion

from features import NEWLINE_TOKEN, _INVISIBLE_RE, _NUMBER_RE, _SPACES_RE, build_vectorizer

_LINE_BREAK_RE = re.compile(r"[^\S\n]*\n[^\S\n]*")
_MULTI_SPACE_RE = re.compile(r" {2,}")


def normalize_text_v2(text: object, lowercase: bool = False) -> str:
    """Same as `features.normalize_text`, except "a\\n\\nb" -> "a ¶ ¶ b" instead of "a ¶ b"."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")
    text = _NUMBER_RE.sub("0", text)
    text = _SPACES_RE.sub(" ", text)
    text = _LINE_BREAK_RE.sub(NEWLINE_TOKEN, text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text.lower() if lowercase else text


def build_vectorizer_v2(**kwargs) -> FeatureUnion:
    """`features.build_vectorizer` (word 1-2 + char 2-5 TF-IDF) with the v2 normalizer."""
    preprocessor = partial(normalize_text_v2, lowercase=kwargs.get("lowercase", False))
    return build_vectorizer(**kwargs).set_params(word__preprocessor=preprocessor, char__preprocessor=preprocessor)


# --- Stylometric features ----------------------------------------------------

FUNCTION_WORDS = frozenset(
    "the a an of to in and or but is are was were be been it this that with for on as by at from not".split()
)
# NFKC rewrites "…" as "...", so the ellipsis character is not counted separately.
PUNCTUATION = {
    ",": "comma", ".": "period", ";": "semicolon", ":": "colon", "!": "exclaim", "?": "question",
    '"': "dquote", "'": "squote", "(": "lparen", ")": "rparen", "-": "hyphen",
    "—": "em_dash", "–": "en_dash", "“": "curly_dquote", "’": "curly_squote",
}
_WORD_RE = re.compile(r"\b\w+\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_BLANK_LINE_RE = re.compile(r"\n[^\S\n]*\n")


def _mean_std(values: list[int]) -> tuple[float, float]:
    arr = np.asarray(values or [0], dtype=float)
    return float(arr.mean()), float(arr.std())


def stylometric_row(text: object) -> dict[str, float]:
    """Length, layout, vocabulary and punctuation statistics for one passage."""
    raw = text if isinstance(text, str) else ""
    n_invisible = len(_INVISIBLE_RE.findall(raw))
    non_ascii_rate = sum(ord(ch) > 127 for ch in raw) / max(len(raw), 1)
    text = _INVISIBLE_RE.sub("", unicodedata.normalize("NFKC", raw))

    n_chars = max(len(text), 1)
    words = _WORD_RE.findall(text)
    n_words = max(len(words), 1)
    lower = [w.lower() for w in words]
    counts = Counter(lower)
    bigrams = list(zip(lower, lower[1:]))
    paragraphs = [p for p in text.split("\n") if p.strip()]
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.replace("\n", " ")) if s.strip()]

    sent_mean, sent_std = _mean_std([len(s.split()) for s in sentences])
    para_mean, para_std = _mean_std([len(p.split()) for p in paragraphs])
    word_mean, word_std = _mean_std([len(w) for w in words])

    row = {
        "n_chars": len(text),
        "n_words": len(words),
        "n_sentences": len(sentences),
        "n_paragraphs": len(paragraphs),
        "n_line_breaks": text.count("\n"),
        "n_blank_lines": len(_BLANK_LINE_RE.findall(text)),
        "sentence_len_mean": sent_mean,
        "sentence_len_std": sent_std,
        "paragraph_len_mean": para_mean,
        "paragraph_len_std": para_std,
        "word_len_mean": word_mean,
        "word_len_std": word_std,
        "type_token_ratio": len(counts) / n_words,
        "hapax_ratio": sum(1 for c in counts.values() if c == 1) / n_words,
        "function_word_rate": sum(counts[w] for w in FUNCTION_WORDS) / n_words,
        "repeated_bigram_rate": 1 - len(set(bigrams)) / max(len(bigrams), 1),
        "upper_char_rate": sum(ch.isupper() for ch in text) / n_chars,
        "digit_char_rate": sum(ch.isdigit() for ch in text) / n_chars,
        "capitalized_word_rate": sum(w[0].isupper() for w in words) / n_words,
        "n_invisible_chars": n_invisible,
        "non_ascii_rate": non_ascii_rate,
        "ends_with_punct": float(text.rstrip()[-1:] in ".!?"),
    }
    for mark, name in PUNCTUATION.items():
        row[f"per_1k_{name}"] = text.count(mark) / n_chars * 1000
    return row


def stylometric_features(texts) -> pd.DataFrame:
    return pd.DataFrame([stylometric_row(t) for t in texts])
