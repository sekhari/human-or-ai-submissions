"""Passage-level measurements used by the Group 26 style models.

Compatibility shim. These feature definitions were authored by Kevin alongside
`team_knn.py`. When the two Group 26 directories were merged, this module and
`features.py` turned out to hold identical definitions: the four regex/string
constants compare equal, and the 23-column feature matrix over all 10,000
labeled passages compares equal under `numpy.array_equal`.

Rather than keep two copies that could silently drift apart, the definitions
now live in `features.py` and this module re-exports them. Every name
`team_knn.py` imported before still resolves, and the values are unchanged.

Import either module; they return the same numbers.
"""

from features import (  # noqa: F401  (re-exported for backwards compatibility)
    CONTRACTION,
    FEATURES_WITHOUT_LAYOUT_PUNCTUATION,
    PUNCTUATION,
    SENTENCE_BREAK,
    WORD,
    style_features,
)

__all__ = [
    "WORD",
    "SENTENCE_BREAK",
    "CONTRACTION",
    "PUNCTUATION",
    "FEATURES_WITHOUT_LAYOUT_PUNCTUATION",
    "style_features",
]
