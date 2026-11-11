"""Model implementations. Each module is owned by one teammate.

Named ``classifiers`` (not ``models``) because the course repository's root
.gitignore ignores every directory called ``models/``.
"""

from src.classifiers.base import TextClassifier

__all__ = ["TextClassifier"]
