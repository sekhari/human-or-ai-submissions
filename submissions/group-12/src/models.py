"""Model factories shared across experiments (ported from notebooks/02_init_model.ipynb).

Every factory returns an unfitted sklearn pipeline that takes a Series of raw texts and
exposes predict_proba, so the same object works for Q1-Q5 scripts.
"""
import re
from functools import lru_cache

import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

try:
    from nltk.corpus import stopwords
    FUNCTION_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    FUNCTION_WORDS = set(stopwords.words("english"))

TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|\n|[^\w\s]")


def mask_content(text):
    """Hide topic words: keep function words, '*' for content words, '#' for numbers, '^' for newlines."""
    masked = []
    for tok in TOKEN.findall(text):
        if tok == "\n":
            masked.append("^")
        elif tok[0].isdigit():
            masked.append("#")
        elif tok[0].isalpha():
            masked.append(tok.lower() if tok.lower() in FUNCTION_WORDS else "*")
        else:
            masked.append(tok)
    return " ".join(masked)


def mask_texts(texts):
    return texts.map(mask_content)


def flatten(texts):
    return texts.str.replace(r"\s+", " ", regex=True).str.strip()


def make_tfidf_lr(C=4.0):
    """Word 1-2gram + char 3-5gram TF-IDF, logistic regression (baseline from notebook 02)."""
    return make_pipeline(
        make_union(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000),
        ),
        LogisticRegression(C=C, max_iter=3000),
    )


def make_masked_lr(C=4.0, flat=False):
    """Masked 1-3gram TF-IDF + logistic regression. flat=True removes all linebreak information."""
    steps = [FunctionTransformer(flatten)] if flat else []
    steps += [
        FunctionTransformer(mask_texts),
        TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 3), min_df=3,
                        sublinear_tf=True, max_features=200_000),
        LogisticRegression(C=C, max_iter=3000),
    ]
    return make_pipeline(*steps)


PUNCTUATION = {
    "comma": r",", "semicolon": r";", "colon": r":", "exclamation": r"!", "question": r"\?",
    "em_dash": r"—", "en_dash": r"–", "hyphen": r"(?<=\w)-(?=\w)", "paren": r"\(",
    "straight_quote": r'"', "curly_quote": r"[“”‘’]", "apostrophe": r"'", "ellipsis": r"\.\.\.|…",
}


@lru_cache(maxsize=1)
def english_words():
    try:
        from nltk.corpus import words
        return frozenset(w.lower() for w in words.words())
    except LookupError:
        nltk.download("words", quiet=True)
        from nltk.corpus import words
        return frozenset(w.lower() for w in words.words())


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def style_features(texts):
    """24 hand-crafted style features per passage: word/sentence lengths, character ratios,
    newline and punctuation rates per 100 words, and the share of words not in an English word list."""
    n_words = texts.str.split().str.len()
    n_chars = texts.str.len()
    lower_words = texts.str.lower().str.findall(r"[a-z]+")
    sent_lens = texts.map(lambda t: [len(s.split()) for s in split_sentences(t)])

    feats = pd.DataFrame({
        "mean_word_length": lower_words.map(lambda ws: np.mean([len(w) for w in ws]) if ws else 0),
        "type_token_ratio": lower_words.map(lambda ws: len(set(ws)) / max(len(ws), 1)),
        "mean_sentence_length": sent_lens.map(np.mean),
        "sentence_length_std": sent_lens.map(np.std),
        "uppercase_ratio": texts.str.count(r"[A-Z]") / n_chars,
        "digit_ratio": texts.str.count(r"\d") / n_chars,
        "non_ascii_ratio": texts.str.count(r"[^\x00-\x7f]") / n_chars,
        "newlines_per_100w": texts.str.count("\n") / n_words * 100,
        "lowercase_i_per_100w": texts.str.count(r"\bi\b") / n_words * 100,
        "double_space_per_100w": texts.str.count(r"  ") / n_words * 100,
    })
    for name, pattern in PUNCTUATION.items():
        feats[f"{name}_per_100w"] = texts.str.count(pattern) / n_words * 100
    vocab = english_words()
    feats["oov_rate"] = lower_words.map(lambda ws: np.mean([w not in vocab for w in ws]) if ws else 0)
    return feats


def make_style_gb(learning_rate=0.1, max_leaf_nodes=31):
    """Style features + histogram gradient boosting (notebook 02)."""
    return make_pipeline(
        FunctionTransformer(style_features),
        HistGradientBoostingClassifier(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, random_state=77),
    )


MODELS = {
    "tfidf_lr": make_tfidf_lr,
    "masked_lr": make_masked_lr,
    "masked_flat_lr": lambda: make_masked_lr(flat=True),
    "style_gb": make_style_gb,
}
