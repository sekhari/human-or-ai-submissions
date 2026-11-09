"""Step 2 - text normalization (deliberately NOT NFKC) and raw-text flags.

normalize(text): remove zero-width chars, map Cyrillic/Greek lookalikes to ASCII,
collapse whitespace. Other characters (e.g. "—", "…", curly quotes) are left untouched.
raw_flags(text): features computed on the RAW text, before normalization.
"""
import re

ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿", "᠎", "‎", "‏"}
_ZW_RE = re.compile("[" + "".join(sorted(ZERO_WIDTH)) + "]")

# Cyrillic / Greek letters visually identical (or near) to an ASCII letter.
HOMOGLYPHS = {
    # Cyrillic uppercase
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "У": "Y",
    # Cyrillic lowercase
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ԛ": "q", "ԝ": "w", "һ": "h",
    # Greek uppercase
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Greek lowercase
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "ι": "i", "κ": "k", "υ": "u", "χ": "x",
    "γ": "y", "ε": "e", "τ": "t",
}
_HOMOGLYPH_TABLE = str.maketrans(HOMOGLYPHS)
_WS_RE = re.compile(r"\s+")


def normalize(text):
    text = _ZW_RE.sub("", text)
    text = text.translate(_HOMOGLYPH_TABLE)
    return _WS_RE.sub(" ", text).strip()


def _is_lookalike(ch):
    return ch in HOMOGLYPHS


def _is_unusual_ws(ch):
    # Any whitespace other than a plain space or newline (tabs, NBSP, thin space, ...).
    return ch not in (" ", "\n") and ch.isspace()


def raw_flags(text):
    """Flags on RAW text: has_zero_width, has_lookalike, has_unusual_ws (0/1) plus
    n_em_dash and n_ellipsis counts of "—" and "…"."""
    return {
        "has_zero_width": int(any(ch in ZERO_WIDTH for ch in text)),
        "has_lookalike": int(any(_is_lookalike(ch) for ch in text)),
        "has_unusual_ws": int(any(_is_unusual_ws(ch) for ch in text)),
        "n_em_dash": text.count("—"),
        "n_ellipsis": text.count("…"),
    }
