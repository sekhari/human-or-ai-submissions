"""Strip layout so models can't use formatting shortcuts.

In train, no human text contains a blank line and markdown markers (leading
">", "#" headings, "**") appear only in AI texts, but test does not follow
the same pattern. Normalized text keeps words and punctuation only.
"""

from __future__ import annotations

import re

_QUOTE = re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE)
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")


def normalize_layout(text: str) -> str:
    text = _QUOTE.sub("", text)
    text = _HEADING.sub("", text)
    text = text.replace("**", "")
    return _WHITESPACE.sub(" ", text).strip()


def normalize_texts(texts) -> list[str]:
    return [normalize_layout(t) for t in texts]
