"""Hand-crafted stylometric features shared by the EDA and the style models."""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
ZERO_WIDTH_RE = re.compile("[​-‏⁠﻿]")
# A "word" mixing Latin with Cyrillic/Greek letters is a classic homoglyph attack.
HOMOGLYPH_RE = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*[Ͱ-ϿЀ-ӿ])\w+\b")
PUNCT = {
    "comma": ",", "semicolon": ";", "colon": ":", "exclam": "!", "question": "?",
    "dquote": '"', "squote": "'", "paren": "(", "hyphen": "-", "emdash": "—",
    "endash": "–", "ellipsis": "…", "curly_quote": "“”‘’",
}

# Features that only describe line layout. EDA: a blank line almost always means AI in train,
# which may be a collection artefact rather than a property of authorship.
LAYOUT_FEATURES = ["blank_lines", "n_lines", "avg_line_len", "markdown_marks"]
LINE_BREAK_RE = re.compile(r"\s*\n\s*")


def strip_layout(text):
    """Replace every line break (and blank line) with one space; used as a robustness check."""
    return LINE_BREAK_RE.sub(" ", text)


def style_features(text):
    words = WORD_RE.findall(text)
    n_words = max(len(words), 1)
    lower = [w.lower() for w in words]
    sents = [s for s in SENT_SPLIT_RE.split(text.strip()) if s]
    sent_lens = np.array([len(s.split()) for s in sents]) if sents else np.array([0])
    n_chars = max(len(text), 1)
    lines = text.split("\n")
    feats = {
        "n_chars": len(text),
        "n_words": len(text.split()),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "n_sentences": len(sents),
        "avg_sent_len": sent_lens.mean(),
        "std_sent_len": sent_lens.std(),
        "type_token_ratio": len(set(lower)) / n_words,
        "stopword_ratio": sum(w in ENGLISH_STOP_WORDS for w in lower) / n_words,
        "n_lines": len(lines),
        "blank_lines": sum(1 for l in lines if not l.strip()),
        "avg_line_len": np.mean([len(l) for l in lines]),
        "digit_ratio": sum(c.isdigit() for c in text) / n_chars,
        "upper_ratio": sum(c.isupper() for c in text) / n_chars,
        "non_ascii_ratio": sum(ord(c) > 127 for c in text) / n_chars,
        "first_person": sum(w in {"i", "me", "my", "mine", "we", "our", "us"} for w in lower) / n_words,
        "ends_terminal_punct": float(text.rstrip()[-1:] in ".!?\"')"),
        "starts_lowercase": float(text[:1].islower()),
        "trailing_ws": float(text != text.rstrip()),
        "double_space": text.count("  "),
        "markdown_marks": len(re.findall(r"^\s*(?:[#>*-]|\d+\.)\s", text, flags=re.M)),
        "zero_width": len(ZERO_WIDTH_RE.findall(text)),
        "homoglyph_words": len(HOMOGLYPH_RE.findall(text)),
    }
    for name, chars in PUNCT.items():
        feats[f"p_{name}"] = sum(text.count(c) for c in chars) / n_words
    return feats


def build_features(df):
    return pd.DataFrame([style_features(t) for t in df.text], index=df.index)


TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?|\d+|[^\w\s]")


def mask_content(text):
    """Keep function words and punctuation; replace content words by their shape.

    Topic words do not transfer to unseen domains, while function-word and punctuation
    patterns might. "The Eiffel tower, built in 1889" -> "the Xx x , x in 0".
    """
    out = []
    for tok in TOKEN_RE.findall(text):
        low = tok.lower()
        if low in ENGLISH_STOP_WORDS or not tok[0].isalnum():
            out.append(low)
        elif tok[0].isdigit():
            out.append("0")
        else:
            out.append("Xx" if tok[0].isupper() else "x")
    return " ".join(out)


TRANSITIONS = ("however", "moreover", "furthermore", "additionally", "overall", "therefore", "conclusion")
CONTRACTION_RE = re.compile(r"\b\w+['\u2019](?:t|s|re|ve|ll|d|m)\b", re.I)


def _repeat_rate(items):
    return 1 - len(set(items)) / len(items) if items else 0.0


def extended_features(text):
    """style_features plus finer line structure and repetition, which weak generators tend to show."""
    feats = style_features(text)
    lower = [w.lower() for w in WORD_RE.findall(text)]
    n_words = max(len(lower), 1)
    counts = {}
    for w in lower:
        counts[w] = counts.get(w, 0) + 1
    sents = [s.strip().lower() for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    starts = [s.split()[0] for s in sents if s.split()]
    sent_lens = np.array([len(s.split()) for s in sents]) if sents else np.array([0])
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    line_lens = np.array([len(l) for l in lines]) if lines else np.array([0])
    n_lines = max(len(lines), 1)
    n_breaks = text.count("\n")
    tail = text.rstrip()[-1:]
    feats.update({
        "bigram_repeat": _repeat_rate(list(zip(lower, lower[1:]))),
        "trigram_repeat": _repeat_rate(list(zip(lower, lower[1:], lower[2:]))),
        "sentence_repeat": _repeat_rate(sents),
        "sentence_start_repeat": _repeat_rate(starts),
        "hapax_ratio": sum(c == 1 for c in counts.values()) / n_words,
        "long_word_ratio": sum(len(w) >= 7 for w in lower) / n_words,
        "contraction_rate": len(CONTRACTION_RE.findall(text)) / n_words,
        "transition_rate": sum(w in TRANSITIONS for w in lower) / n_words,
        "sent_len_cv": sent_lens.std() / max(sent_lens.mean(), 1),
        "commas_per_sentence": text.count(",") / max(len(sents), 1),
        "n_paragraphs": len(re.split(r"\n\s*\n", text.strip())),
        "single_break_share": len(re.findall(r"(?<!\n)\n(?!\n)", text)) / max(n_breaks, 1),
        "line_len_std": line_lens.std(),
        "first_line_len": line_lens[0],
        "last_line_len": line_lens[-1],
        "lines_end_punct": sum(l[-1] in ".!?" for l in lines) / n_lines,
        "lines_end_colon": sum(l.endswith(":") for l in lines) / n_lines,
        "lines_start_upper": sum(l[0].isupper() for l in lines) / n_lines,
        "numbered_lines": len(re.findall(r"^\s*\d+[.)]\s", text, flags=re.M)),
        "bullet_lines": len(re.findall(r"^\s*[-*\u2022]\s", text, flags=re.M)),
        "bold_marks": text.count("**"),
        "ends_alpha": float(tail.isalpha()),
        "ends_comma": float(tail == ","),
    })
    return feats


def build_extended_features(df):
    return pd.DataFrame([extended_features(t) for t in df.text], index=df.index)
