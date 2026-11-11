"""Reusable feature extractors: TF-IDF (word / char / both) and hand-crafted
stylometric features. Models in src/models/ import from here."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from tqdm.auto import tqdm


class TfidfFeaturizer:
    """Word and/or character TF-IDF, horizontally stacked.

    params example:
        word: {ngram_range: [1, 2], max_features: 200000, min_df: 2}
        char: {ngram_range: [3, 5], max_features: 300000, min_df: 3, analyzer: char_wb}
        sublinear_tf: true
        lowercase: true
    Set word: null or char: null to disable one of them.
    """

    def __init__(self, word=None, char=None, sublinear_tf=True, lowercase=True,
                 tokenizer_name=None):
        self.cfgs = {"word": word, "char": char}
        self.sublinear_tf = sublinear_tf
        self.lowercase = lowercase
        self.tokenizer_name = tokenizer_name  # optional HF subword tokenizer for the word part
        self.vectorizers: dict[str, TfidfVectorizer] = {}

    def _make(self, kind: str, c: dict) -> TfidfVectorizer:
        common = dict(sublinear_tf=self.sublinear_tf, lowercase=self.lowercase,
                      min_df=c.get("min_df", 2), max_df=c.get("max_df", 1.0),
                      max_features=c.get("max_features"), dtype=np.float32,
                      ngram_range=tuple(c.get("ngram_range", (1, 2))))
        if kind == "char":
            return TfidfVectorizer(analyzer=c.get("analyzer", "char_wb"), **common)
        if self.tokenizer_name:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
            return TfidfVectorizer(tokenizer=tok.tokenize, token_pattern=None, **common)
        return TfidfVectorizer(analyzer="word", token_pattern=c.get("token_pattern", r"(?u)\b\w+\b|[^\w\s]"), **common)

    def fit(self, texts):
        for kind, c in self.cfgs.items():
            if c:
                v = self._make(kind, c)
                v.fit(tqdm(texts, desc=f"tfidf-{kind} fit", leave=False))
                self.vectorizers[kind] = v
        if not self.vectorizers:
            raise ValueError("TfidfFeaturizer needs at least one of word / char")
        return self

    def transform(self, texts):
        mats = [v.transform(texts) for v in self.vectorizers.values()]
        return sp.hstack(mats).tocsr() if len(mats) > 1 else mats[0]

    @property
    def n_features(self):
        return sum(len(v.vocabulary_) for v in self.vectorizers.values())


# ---------------------------------------------------------------- stylometry
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[^.!?\n]+[.!?]?")
PUNCT = list(",.;:!?\"'()-*#") + ["’", "“", "—", "–", "…"]
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our"}
CONTRACTION_RE = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.I)


def _stylo_one(t: str) -> list[float]:
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
    + [f"punct_{p}" for p in PUNCT]
)

# ------------------------------------------------- extended features (Ma et al., 2023)
# Ma et al. find AI scientific text uses fewer adverbs, conjunctions, numbers and
# pronouns, fewer commas/uppercase, shorter sentences, and has higher
# sentence-to-sentence similarity (coherence). Function-word usage is what their
# LIME analysis shows fine-tuned detectors rely on. POS tags are approximated
# with closed word lists so no NLP model download is needed.
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


def _extended_one(t: str) -> list[float]:
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
        sum(counts[w] for w in FUNCTION_WORDS) / n,                       # function-word share
        sum(c for w, c in counts.items() if w in PRONOUNS) / n,
        sum(c for w, c in counts.items() if w in CONJUNCTIONS) / n,
        sum(c for w, c in counts.items() if w in HEDGES) / n,
        sum(c for w, c in counts.items() if w.endswith("ly") and len(w) > 4) / n,  # adverb rate (words ending in -ly)
        len(_NUM_RE.findall(t)) / n,
        sum(1 for w in lw if len(w) > 6) / n,                            # long words
        spw,
        206.835 - 1.015 * wps - 84.6 * spw,                               # Flesch reading ease
        0.39 * wps + 11.8 * spw - 15.59,                                  # Flesch-Kincaid grade
        float(np.mean(sims)) if sims else 0.0,                            # coherence (adjacent sentences)
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

STYLO_NAMES = BASE_NAMES  # kept for backwards compatibility


def stylometric_names(extended: bool = False) -> list[str]:
    return BASE_NAMES + (EXT_NAMES if extended else [])


def stylometric_features(texts, desc="stylometric", extended: bool = False) -> np.ndarray:
    rows = []
    for t in tqdm(texts, desc=desc, leave=False):
        f = _stylo_one(t)
        if extended:
            f += _extended_one(t)
        rows.append(f)
    return np.array(rows, dtype=np.float32)
