"""Stateless, importable quote canonicalization for reproducible pipelines."""
from sklearn.base import BaseEstimator, TransformerMixin

from .q5_sensitivity import apply_condition


class QuoteNormalizer(TransformerMixin, BaseEstimator):
    """Map the frozen curly-quote characters to ASCII, without corpus fitting."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [apply_condition(text, "quote_typography") for text in X]
