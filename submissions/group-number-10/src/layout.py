"""Label-free layout changes: re-break a passage's lines without changing its words.

Used by the line-break stress test (experiments/format_stress.py) and, at training time, as data
augmentation (DistilBertConfig.layout_aug) so the classifier cannot use layout as a shortcut.
"""

from __future__ import annotations

import re
import textwrap

import numpy as np


def no_newlines(text: str) -> str:
    """Every line break replaced by a space (one paragraph)."""
    return re.sub(r"\s*\n\s*", " ", text).strip()


def unwrap(text: str) -> str:
    """Single line breaks joined, blank-line paragraph breaks kept (undoes hard wrapping)."""
    return re.sub(r"[ \t]*(?<!\n)\n(?!\n)[ \t]*", " ", text)


def hard_wrap(text: str, width: int = 70) -> str:
    """Every paragraph re-wrapped at `width` characters."""
    return "\n\n".join(textwrap.fill(" ".join(p.split()), width) for p in re.split(r"\n\s*\n", text) if p.strip())


def short_lines(text: str, rng: np.random.Generator) -> str:
    """All words re-broken into lines of 4-9 words (poem-like layout)."""
    words, lines, i = text.split(), [], 0
    while i < len(words):
        k = int(rng.integers(4, 10))
        lines.append(" ".join(words[i: i + k]))
        i += k
    return "\n".join(lines)


def random_layout(text: str, rng: np.random.Generator) -> str:
    """One of the four layouts above, chosen uniformly (hard-wrap width uniform in 40-90)."""
    choice = int(rng.integers(4))
    if choice == 0:
        return no_newlines(text)
    if choice == 1:
        return unwrap(text)
    if choice == 2:
        return hard_wrap(text, int(rng.integers(40, 91)))
    return short_lines(text, rng)


def augment_layouts(texts: list[str], prob: float, rng: np.random.Generator) -> list[str]:
    """Re-lay-out each passage with probability `prob`, independent of its label."""
    return [random_layout(t, rng) if rng.random() < prob else t for t in texts]
