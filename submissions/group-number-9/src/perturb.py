"""Synthetic edits approximating the adversarially modified AI passages in the test set.

The real attacks are not disclosed, so results on these edits are only a rough stress test.
"""
import numpy as np

from src.features import WORD_RE, strip_layout

HOMOGLYPHS = str.maketrans({"a": "а", "e": "е", "o": "о"})  # Cyrillic look-alikes


def add_typos(text, rate, rng):
    """Swap two adjacent letters in about `rate` of the words with at least four letters."""
    def swap(match):
        w = match.group()
        if len(w) < 4 or rng.random() >= rate:
            return w
        i = rng.integers(1, len(w) - 2)
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return WORD_RE.sub(swap, text)


def add_homoglyphs(text, rate, rng):
    return "".join(c.translate(HOMOGLYPHS) if rng.random() < rate else c for c in text)


def attack(text, rng, typo_rate=0.05, homoglyph_rate=0.01):
    return add_homoglyphs(add_typos(strip_layout(text), typo_rate, rng), homoglyph_rate, rng)


def attack_ai_docs(df, seed=42):
    """Edit only the AI rows, as the test set does; human rows stay untouched."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    is_ai = out.label == 1
    out.loc[is_ai, "text"] = [attack(t, rng) for t in out.loc[is_ai, "text"]]
    return out
