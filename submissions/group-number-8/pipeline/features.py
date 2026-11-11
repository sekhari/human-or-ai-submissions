"""Wording features (BPE TF-IDF) and layout features.

The wording path flattens whitespace, trains a byte-level BPE tokenizer from
scratch, and builds 1--3 token TF-IDF. The layout path never sees the words.
No pretrained weights are used. Labels are not used while fitting BPE or IDF.
"""

from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pipeline.config import (
    BPE_MIN_FREQUENCY,
    BPE_VOCAB_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)

FORMAT_COLUMNS = [
    "n_words",
    "newlines_per_100w",
    "blank_lines_per_100w",
    "has_blank_line",
    "mid_sentence_newlines_per_100w",
    "newline_after_punct_share",
    "mean_line_len",
    "line_len_std",
    "double_space_per_100w",
    "ends_sentence_punct",
    "ends_quote_or_bracket",
    "starts_quote",
    "starts_lowercase",
    "md_bullets_per_100w",
    "md_numbered_per_100w",
    "md_headers",
    "md_bold",
    "quote_lines",
    "non_ascii_per_100c",
    "curly_quotes_per_100w",
    "em_dash_per_100w",
]


def flatten_text(text: pd.Series) -> pd.Series:
    return text.fillna("").astype(str).map(lambda s: re.sub(r"\s+", " ", s).strip())


def _identity(tokens):
    return tokens


def train_bpe(texts: list[str]) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=BPE_VOCAB_SIZE,
        min_frequency=BPE_MIN_FREQUENCY,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=_identity,
        preprocessor=_identity,
        token_pattern=None,
        lowercase=False,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True,
        max_features=TFIDF_MAX_FEATURES,
        dtype=np.float32,
    )


def _encode(tokenizer: Tokenizer, docs: list[str]) -> list[list[str]]:
    if not docs:
        return []
    return [encoding.tokens for encoding in tokenizer.encode_batch(docs)]


def build_word_features(
    fit_docs: list[str], transform_docs: dict[str, list[str]]
) -> tuple[Tokenizer, TfidfVectorizer, dict, dict]:
    """Fit BPE and TF-IDF on fit_docs. Transform every list in transform_docs.

    fit_docs may contain duplicates and empty strings. BPE training uses the
    unique non-empty strings in first-seen order. TF-IDF is fit on every
    fit document, matching the way the submission vectorizer is built.
    """
    started = time.perf_counter()
    corpus = list(dict.fromkeys(doc for doc in fit_docs if doc))
    tokenizer = train_bpe(corpus)
    vectorizer = make_vectorizer()
    vectorizer.fit(_encode(tokenizer, fit_docs))
    matrices = {
        name: vectorizer.transform(_encode(tokenizer, docs))
        for name, docs in transform_docs.items()
    }
    n_features = int(next(iter(matrices.values())).shape[1]) if matrices else 0
    info = {
        "vocab_size": int(tokenizer.get_vocab_size()),
        "n_features": n_features,
        "n_fit_docs": int(len(fit_docs)),
        "n_bpe_docs": int(len(corpus)),
        "seconds": time.perf_counter() - started,
    }
    return tokenizer, vectorizer, matrices, info


def format_features(text: pd.Series) -> pd.DataFrame:
    """21 layout features from the raw string, mostly scaled by length."""
    n_chars = text.str.len().clip(lower=1)
    n_words = text.str.split().str.len().fillna(0).clip(lower=1)
    n_lines = text.str.count("\n") + 1
    frame = pd.DataFrame(index=text.index)
    frame["n_words"] = n_words
    frame["newlines_per_100w"] = 100 * text.str.count("\n") / n_words
    frame["blank_lines_per_100w"] = 100 * text.str.count(r"\n[ \t]*\n") / n_words
    frame["has_blank_line"] = text.str.contains(r"\n[ \t]*\n").astype(int)
    frame["mid_sentence_newlines_per_100w"] = (
        100 * text.str.count(r"[a-z,;]\n[a-z]") / n_words
    )
    frame["newline_after_punct_share"] = text.str.count(r"[.!?:\"']\n") / text.str.count(
        "\n"
    ).clip(lower=1)
    frame["mean_line_len"] = n_chars / n_lines
    frame["line_len_std"] = text.map(
        lambda s: np.std([len(line) for line in s.split("\n")]) if "\n" in s else 0.0
    )
    frame["double_space_per_100w"] = 100 * text.str.count("  ") / n_words
    last = text.str.rstrip().str[-1:]
    frame["ends_sentence_punct"] = last.isin(list(".!?")).astype(int)
    frame["ends_quote_or_bracket"] = last.isin(list("\"')]”’")).astype(int)
    frame["starts_quote"] = text.str.lstrip().str[:1].isin(list("\"“'")).astype(int)
    frame["starts_lowercase"] = (
        text.str.lstrip().str[:1].str.islower().fillna(False).astype(int)
    )
    frame["md_bullets_per_100w"] = 100 * text.str.count(r"(?m)^\s*[-*•] ") / n_words
    frame["md_numbered_per_100w"] = 100 * text.str.count(r"(?m)^\s*\d+[.)] ") / n_words
    frame["md_headers"] = text.str.count(r"(?m)^#+ ")
    frame["md_bold"] = text.str.count(r"\*\*")
    frame["quote_lines"] = text.str.count(r"(?m)^\s*>")
    frame["non_ascii_per_100c"] = (
        100 * text.map(lambda s: sum(ord(ch) > 127 for ch in s)) / n_chars
    )
    frame["curly_quotes_per_100w"] = 100 * text.str.count("[“”‘’]") / n_words
    frame["em_dash_per_100w"] = 100 * text.str.count("—") / n_words
    frame = frame.loc[:, FORMAT_COLUMNS].astype(float)
    return frame


def format_matrix(text: pd.Series) -> np.ndarray:
    return format_features(text).to_numpy(dtype=np.float64)
