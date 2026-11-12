"""Text normalization and TF-IDF feature extraction for the Human-or-AI task."""

from __future__ import annotations

import re
import unicodedata
from functools import partial

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# Invisible characters used by RAID-style adversarial attacks (zero-width
# spaces/joiners, BOM, soft hyphen, word joiner).
_INVISIBLE_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_PARAGRAPH_RE = re.compile(r"\s*\n\s*")
_SPACES_RE = re.compile(r"[^\S\n]+")
_NUMBER_RE = re.compile(r"\d")
NEWLINE_TOKEN = " ¶ "


def normalize_text(text: object, lowercase: bool = False) -> str:
    """Light, attack-robust normalization shared by every feature view."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")
    text = _NUMBER_RE.sub("0", text)
    text = _SPACES_RE.sub(" ", text)
    text = _PARAGRAPH_RE.sub(NEWLINE_TOKEN, text).strip()
    return text.lower() if lowercase else text


def build_vectorizer(
    word_max_features: int = 100_000,
    char_max_features: int = 200_000,
    min_df: int = 2,
    lowercase: bool = False,
    char_analyzer: str = "char",
    char_ngrams: tuple[int, int] = (2, 5),
) -> FeatureUnion:
    """Word (1-2 gram) + character n-gram TF-IDF, concatenated."""
    preprocessor = partial(normalize_text, lowercase=lowercase)
    word = TfidfVectorizer(
        preprocessor=preprocessor,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.95,
        max_features=word_max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        preprocessor=preprocessor,
        analyzer=char_analyzer,
        ngram_range=char_ngrams,
        min_df=min_df,
        max_features=char_max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)])
