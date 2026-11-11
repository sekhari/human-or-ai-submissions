"""Hand-crafted stylometric features (no learned or pretrained components).

Ported from the research repo's src/features.py (base + "extended" Ma et al. 2023 features), plus
line-structure and adversarial-character features added for this track:
  - line structure: median / max words per line, share of short lines (verse is ~15% of test,
    0.3% of train, and line length differs sharply by class in train)
  - character classes: zero-width characters, Cyrillic/Greek letters inside Latin words (homoglyphs),
    other non-ASCII letters, curly quotes, em/en dashes
Each feature is a number per passage; names are in FEATURE_NAMES (same order).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[^.!?\n]+[.!?]?")
PUNCT = list(",.;:!?\"'()-*#") + ["’", "“", "—", "–", "…"]
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our"}
CONTRACTION_RE = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.I)


def _base(t: str) -> list[float]:
    n_chars = max(len(t), 1)
    words = _WORD_RE.findall(t)
    n_words = max(len(words), 1)
    lw = [w.lower() for w in words]
    sents = [s.strip() for s in _SENT_RE.findall(t) if s.strip()]
    sent_lens = np.array([len(s.split()) for s in sents]) if sents else np.array([0])
    word_lens = np.array([len(w) for w in words]) if words else np.array([0])
    paras = [p for p in t.split("\n") if p.strip()]
    counts = Counter(lw)
    bigrams = list(zip(lw, lw[1:]))
    trigrams = list(zip(lw, lw[1:], lw[2:]))
    feats = [
        len(t), n_words, len(sents), len(paras),
        word_lens.mean(), word_lens.std(),
        sent_lens.mean(), sent_lens.std(), sent_lens.std() / (sent_lens.mean() + 1e-6),  # burstiness
        len(counts) / n_words,                                   # type-token ratio
        sum(1 for c in counts.values() if c == 1) / n_words,     # hapax ratio
        1 - len(set(bigrams)) / max(len(bigrams), 1),            # repeated bigrams
        1 - len(set(trigrams)) / max(len(trigrams), 1),          # repeated trigrams
        sum(1 for w in lw if w in ENGLISH_STOP_WORDS) / n_words,
        sum(1 for w in lw if w in FIRST_PERSON) / n_words,
        len(CONTRACTION_RE.findall(t)) / n_words,
        sum(c.isupper() for c in t) / n_chars,
        sum(c.isdigit() for c in t) / n_chars,
        sum(ord(c) > 127 for c in t) / n_chars,
        t.count("\n") / n_chars * 100, t.count("\n\n"), t.count("  "),
        float(t[:1].isupper()), float(t.rstrip()[-1:] in ".!?\"'"),
        sum(1 for s in sents if s[:1].isupper()) / max(len(sents), 1),
    ]
    feats += [t.count(p) / n_words for p in PUNCT]
    return feats


BASE_NAMES = (
    ["n_chars", "n_words", "n_sents", "n_paras", "word_len_mean", "word_len_std",
     "sent_len_mean", "sent_len_std", "burstiness", "ttr", "hapax_ratio",
     "rep_bigram", "rep_trigram", "stopword_ratio", "first_person", "contractions",
     "upper_ratio", "digit_ratio", "nonascii_ratio", "newline_rate", "n_blank_lines",
     "n_double_space", "starts_upper", "ends_punct", "sent_cap_ratio"]
    + [f"punct_{i}" for i in range(len(PUNCT))]
)

# ------------------------------------------------- extended features (Ma et al., 2023)
# POS tags are approximated with closed word lists, so nothing is downloaded.
FUNCTION_WORDS = ["the", "of", "and", "to", "a", "in", "that", "is", "was", "it", "for", "on",
                  "with", "as", "be", "by", "at", "this", "from", "or", "an", "but", "not", "are",
                  "which", "have", "has", "can", "will", "would", "also", "more", "such", "these"]
PRONOUNS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "you", "your", "yours",
            "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"}
CONJUNCTIONS = {"and", "but", "or", "nor", "yet", "so", "because", "although", "though", "while",
                "whereas", "since", "unless", "if", "however", "therefore", "moreover", "furthermore",
                "additionally", "thus", "hence"}
HEDGES = {"may", "might", "could", "perhaps", "possibly", "likely", "generally", "often",
          "typically", "usually", "potentially", "arguably"}
_VOWELS_RE = re.compile(r"[aeiouy]+")
_NUM_RE = re.compile(r"\b\d[\d,.]*\b")


def _syllables(w: str) -> int:
    n = len(_VOWELS_RE.findall(w))
    if w.endswith("e") and n > 1:
        n -= 1
    return max(n, 1)


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb + 1e-9)


def _extended(t: str) -> list[float]:
    words = _WORD_RE.findall(t)
    lw = [w.lower() for w in words]
    n = max(len(lw), 1)
    counts = Counter(lw)
    sents = [s.strip() for s in _SENT_RE.findall(t) if len(s.split()) >= 3]
    n_sent = max(len(sents), 1)
    syl = [_syllables(w) for w in lw] or [1]
    wps = n / n_sent
    spw = float(np.mean(syl))
    bags = [Counter(w for w in _WORD_RE.findall(s.lower()) if w not in ENGLISH_STOP_WORDS) for s in sents]
    sims = [_cos(bags[i], bags[i + 1]) for i in range(len(bags) - 1)]
    para_lens = [len(p.split()) for p in t.split("\n") if p.strip()] or [0]
    feats = [counts.get(w, 0) / n for w in FUNCTION_WORDS]
    feats += [
        sum(counts[w] for w in FUNCTION_WORDS) / n,
        sum(c for w, c in counts.items() if w in PRONOUNS) / n,
        sum(c for w, c in counts.items() if w in CONJUNCTIONS) / n,
        sum(c for w, c in counts.items() if w in HEDGES) / n,
        sum(c for w, c in counts.items() if w.endswith("ly") and len(w) > 4) / n,  # adverb rate (words ending in -ly)
        len(_NUM_RE.findall(t)) / n,
        sum(1 for w in lw if len(w) > 6) / n,
        spw,
        206.835 - 1.015 * wps - 84.6 * spw,                                         # Flesch reading ease
        0.39 * wps + 11.8 * spw - 15.59,                                            # Flesch-Kincaid grade
        float(np.mean(sims)) if sims else 0.0,                                      # adjacent-sentence coherence
        float(np.std(sims)) if sims else 0.0,
        float(np.std(para_lens)),
        sum(1 for s in sents if s.endswith("?")) / n_sent,
        sum(1 for s in sents if s.endswith("!")) / n_sent,
    ]
    return feats


EXT_NAMES = ([f"fw_{w}" for w in FUNCTION_WORDS]
             + ["function_word_share", "pronoun_rate", "conjunction_rate", "hedge_rate", "adverb_ly_rate",
                "number_rate", "long_word_rate", "syllables_per_word", "flesch_reading_ease", "fk_grade",
                "coherence_mean", "coherence_std", "para_len_std", "question_sent_ratio", "exclaim_sent_ratio"])

# ------------------------------------------------- added for this track
ZERO_WIDTH = "​‌‍⁠﻿"
_CONFUSABLE = re.compile(r"[Ͱ-ϿЀ-ӿ]")          # Greek + Cyrillic letters
_MIXED_WORD = re.compile(r"\w*[A-Za-z]\w*")


def _added(t: str) -> list[float]:
    n_chars = max(len(t), 1)
    lines = [ln for ln in t.splitlines() if ln.strip()]
    wpl = np.array([len(ln.split()) for ln in lines]) if lines else np.array([0])
    words = _MIXED_WORD.findall(t)
    homoglyph_words = sum(1 for w in words if _CONFUSABLE.search(w))
    return [
        float(np.median(wpl)), float(wpl.max()), float((wpl <= 10).mean()), len(lines),
        sum(t.count(z) for z in ZERO_WIDTH),
        homoglyph_words,
        sum(1 for c in t if ord(c) > 127 and c.isalpha() and not _CONFUSABLE.match(c)) / n_chars,
        sum(t.count(c) for c in "‘’“”") / n_chars * 100,
        sum(t.count(c) for c in "–—") / n_chars * 100,
    ]


ADDED_NAMES = ["words_per_line_median", "words_per_line_max", "short_line_share", "n_lines",
               "n_zero_width", "n_homoglyph_words", "other_nonascii_letter_rate", "curly_quote_rate", "dash_rate"]


def feature_names(extended: bool = True, added: bool = True) -> list[str]:
    return BASE_NAMES + (EXT_NAMES if extended else []) + (ADDED_NAMES if added else [])


_CACHE: dict[str, list[float]] = {}  # text -> all features; the inner-CV grid re-featurises the same texts


def stylometric_features(texts, extended: bool = True, added: bool = True) -> np.ndarray:
    nb, ne = len(BASE_NAMES), len(EXT_NAMES)
    cols = list(range(nb)) + (list(range(nb, nb + ne)) if extended else []) \
        + (list(range(nb + ne, nb + ne + len(ADDED_NAMES))) if added else [])
    rows = []
    for t in texts:
        if t not in _CACHE:
            _CACHE[t] = _base(t) + _extended(t) + _added(t)
        rows.append(_CACHE[t])
    return np.asarray(rows, dtype=np.float32)[:, cols]
