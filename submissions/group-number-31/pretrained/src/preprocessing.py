"""Text preprocessing pipeline.

Every step is a pure function str -> str, switched on/off in the
`preprocessing:` block of the config. The same pipeline is applied to every
split so train and test always see identical cleaning.

Note: in this dataset, formatting (blank lines, double spaces, stripped
fraction slashes, ...) differs systematically between classes. Removing it
can lower validation AUC but may or may not help on unseen test domains --
treat each step as an experiment, not as an obvious improvement.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable

from tqdm.auto import tqdm

ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿­]")

# Common look-alike characters (Cyrillic / Greek) used in homoglyph attacks.
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "У": "Y",
    "ο": "o", "α": "a", "ν": "v", "Ο": "O", "Α": "A", "Β": "B", "Ε": "E",
    "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ρ": "P",
    "Τ": "T", "Χ": "X", "Υ": "Y",
}
_LATIN_RE = re.compile(r"[A-Za-z]")
_TOKEN_RE = re.compile(r"\S+")

QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"',
          "–": "-", "—": "-", "…": "..."}
_QUOTES_RE = re.compile("|".join(map(re.escape, QUOTES)))


def unicode_nfkc(t: str) -> str:
    return unicodedata.normalize("NFKC", t)


def remove_zero_width(t: str) -> str:
    return ZERO_WIDTH_RE.sub("", t)


def map_homoglyphs(t: str) -> str:
    """Replace look-alike chars only inside tokens that also contain Latin
    letters, so genuine Russian/Greek words are left alone."""
    def fix(m: re.Match) -> str:
        tok = m.group(0)
        if _LATIN_RE.search(tok):
            return "".join(HOMOGLYPHS.get(c, c) for c in tok)
        return tok
    return _TOKEN_RE.sub(fix, t)


def normalize_quotes(t: str) -> str:
    return _QUOTES_RE.sub(lambda m: QUOTES[m.group(0)], t)


def collapse_blank_lines(t: str) -> str:
    return re.sub(r"\n\s*\n+", "\n", t)


def join_lines(t: str) -> str:
    """Replace every line break with a space. Train humans are mostly long unwrapped lines and train AI
    short lines, so line structure is a class shortcut that flags human verse (~15% of test) as AI."""
    return re.sub(r"\s*\n\s*", " ", t)


def collapse_spaces(t: str) -> str:
    return re.sub(r"[ \t]+", " ", t)


def lowercase(t: str) -> str:
    return t.lower()


def strip(t: str) -> str:
    return t.strip()


# Order matters: steps run in this order when enabled.
STEPS: dict[str, Callable[[str], str]] = {
    "unicode_nfkc": unicode_nfkc,
    "remove_zero_width": remove_zero_width,
    "map_homoglyphs": map_homoglyphs,
    "normalize_quotes": normalize_quotes,
    "collapse_blank_lines": collapse_blank_lines,
    "join_lines": join_lines,
    "collapse_spaces": collapse_spaces,
    "lowercase": lowercase,
    "strip": strip,
}


def build_pipeline(pp_cfg: dict) -> list[Callable[[str], str]]:
    unknown = set(pp_cfg) - set(STEPS)
    if unknown:
        raise ValueError(f"Unknown preprocessing steps: {unknown}. Known: {list(STEPS)}")
    return [fn for name, fn in STEPS.items() if pp_cfg.get(name, False)]


def preprocess(texts: Iterable[str], pp_cfg: dict, desc: str = "preprocess") -> list[str]:
    steps = build_pipeline(pp_cfg)
    out = []
    for t in tqdm(list(texts), desc=desc, leave=False):
        t = "" if t is None else str(t)
        for fn in steps:
            t = fn(t)
        out.append(t)
    return out
