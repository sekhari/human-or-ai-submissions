"""Hand-crafted style features: layout, burstiness, punctuation and vocabulary."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from layout import normalize_texts

_WORD = re.compile(r"\b\w+\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARA_SPLIT = re.compile(r"\n\s*\n")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s")

# Characters whose rate (count per character of text) becomes a feature.
_CHAR_RATES = {
    "comma": ",", "period": ".", "semicolon": ";", "colon": ":",
    "exclam": "!", "question": "?", "paren": "(", "dquote": '"',
    "apostrophe": "'", "hyphen": "-", "en_dash": "–", "em_dash": "—",
}


def _stats(values: list[int], prefix: str) -> dict[str, float]:
    """Mean, std, burstiness (coefficient of variation), min and max of lengths."""
    arr = np.asarray(values, dtype=float) if values else np.zeros(1)
    mean = arr.mean()
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": arr.std(),
        f"{prefix}_cv": arr.std() / mean if mean > 0 else 0.0,
        f"{prefix}_min": arr.min(),
        f"{prefix}_max": arr.max(),
    }


def _one(text: str) -> dict[str, float]:
    n_chars = max(len(text), 1)
    words = _WORD.findall(text)
    n_words = max(len(words), 1)
    lines = text.split("\n")
    sentences = [s for s in _SENT_SPLIT.split(text.strip()) if s]
    paragraphs = [p for p in _PARA_SPLIT.split(text.strip()) if p.strip()]
    lower_words = [w.lower() for w in words]
    counts = pd.Series(lower_words).value_counts() if words else pd.Series(dtype=int)

    f = {
        "n_chars": len(text),
        "n_words": len(words),
        "n_newlines": text.count("\n"),
        "n_double_newlines": text.count("\n\n"),
        "newline_rate": text.count("\n") / n_chars,
        "frac_empty_lines": sum(not l.strip() for l in lines) / len(lines),
        "frac_bullet_lines": sum(bool(_BULLET.match(l)) for l in lines) / len(lines),
        "n_heading_lines": sum(l.lstrip().startswith("#") for l in lines),
        "n_bold_markers": text.count("**"),
        "starts_with_gt": float(text.lstrip().startswith(">")),
        "n_trailing_space_lines": sum(l != l.rstrip(" \t") for l in lines),
        "n_double_spaces": text.count("  "),
        "ends_with_punct": float(text.rstrip()[-1:] in ".!?"),
        "n_sentences": len(sentences),
        "n_paragraphs": len(paragraphs),
        "type_token_ratio": len(counts) / n_words,
        "hapax_ratio": float((counts == 1).sum()) / n_words,
        "digit_rate": sum(c.isdigit() for c in text) / n_chars,
        "upper_rate": sum(c.isupper() for c in text) / n_chars,
        "space_rate": text.count(" ") / n_chars,
        "non_ascii_rate": sum(ord(c) > 127 for c in text) / n_chars,
    }
    for name, ch in _CHAR_RATES.items():
        f[f"{name}_rate"] = text.count(ch) / n_chars
    f.update(_stats([len(_WORD.findall(s)) for s in sentences], "sent_len"))
    f.update(_stats([len(l) for l in lines if l.strip()], "line_len"))
    f.update(_stats([len(p) for p in paragraphs], "para_len"))
    f.update(_stats([len(w) for w in words], "word_len"))
    return f


# Features that only describe layout (newlines, paragraphs, markup, spacing).
LAYOUT_FEATURES = [
    "n_newlines", "n_double_newlines", "newline_rate", "frac_empty_lines",
    "frac_bullet_lines", "n_heading_lines", "n_bold_markers", "starts_with_gt",
    "n_trailing_space_lines", "n_double_spaces", "n_paragraphs",
    *(f"{prefix}_{stat}" for prefix in ("line_len", "para_len")
      for stat in ("mean", "std", "cv", "min", "max")),
]


def style_features(texts, layout: bool = True) -> pd.DataFrame:
    """With layout=False, texts are layout-normalized first and the layout-only
    features are dropped."""
    if not layout:
        texts = normalize_texts(texts)
    features = pd.DataFrame([_one(t) for t in texts])
    return features if layout else features.drop(columns=LAYOUT_FEATURES)
