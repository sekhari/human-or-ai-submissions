"""Interpretable, lexical-identity-free raw-text style models.

The transformer is stateless: it never learns a vocabulary or corpus statistics.
All features describe the supplied text's structure; no source ID, document ID,
label, external model, or topic/author dictionary is used. The caller controls
the train/validation protocol and must fit each model on training rows only.
"""
from __future__ import annotations

from collections import Counter
import math
import re
import string
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline


TOKEN_RE = re.compile(r"\b\w+(?:['’]\w+)*\b", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"[.!?。！？]+(?:[\"'”’)]*)\s*|\n+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
PUNCTUATION_GROUPS = {
    'period': '.', 'comma': ',', 'semicolon': ';', 'colon': ':',
    'exclamation': '!', 'question': '?', 'apostrophe': "'’",
    'double_quote': '\"“”', 'hyphen': '-', 'long_dash': '—–',
    'parenthesis': '()', 'square_bracket': '[]', 'curly_bracket': '{}',
    'slash': '/\\', 'hash': '#', 'asterisk': '*', 'underscore': '_',
    'backtick': '`', 'equals': '=', 'angle_bracket': '<>',
    'ampersand': '&', 'percent': '%', 'at': '@', 'tilde': '~',
}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    return value if isinstance(value, str) else str(value)


def _entropy(counts: Iterable[int]) -> float:
    values = np.asarray(list(counts), dtype=np.float64)
    if values.size == 0 or values.sum() == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _summary(values: Iterable[int], prefix: str) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        array = np.zeros(1)
    return {
        f'{prefix}_mean': float(array.mean()),
        f'{prefix}_std': float(array.std()),
        f'{prefix}_min': float(array.min()),
        f'{prefix}_p25': float(np.percentile(array, 25)),
        f'{prefix}_median': float(np.median(array)),
        f'{prefix}_p75': float(np.percentile(array, 75)),
        f'{prefix}_p90': float(np.percentile(array, 90)),
        f'{prefix}_max': float(array.max()),
    }


def extract_style_features(value: object) -> dict[str, float]:
    """Return fixed-order scalar features for one original, uncleaned text.

    Tokens include Unicode word characters and internal apostrophes. Sentence
    units split on terminal punctuation or newlines, and empty units are
    dropped. Paragraphs split on blank lines. Rates use max(denominator, 1),
    so missing and empty strings have finite outputs.
    """
    text = _text(value)
    tokens = TOKEN_RE.findall(text)
    folded = [token.casefold() for token in tokens]
    token_counts = Counter(folded)
    char_counts = Counter(text)
    sentences = [part for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    paragraphs = [part for part in PARAGRAPH_SPLIT_RE.split(text) if part.strip()]
    lines = text.splitlines() if text else []
    n_chars, n_tokens = len(text), len(tokens)
    chars_denom, tokens_denom = max(n_chars, 1), max(n_tokens, 1)
    lines_denom = max(len(lines), 1)
    alphabetic_chars = sum(char.isalpha() for char in text)
    alphabetic_tokens = [token for token in tokens if any(c.isalpha() for c in token)]
    alpha_tokens_denom = max(len(alphabetic_tokens), 1)
    punctuation_count = sum(
        count for char, count in char_counts.items()
        if char in string.punctuation or char in '“”‘’—–…。！？、，；：'
    )
    token_entropy = _entropy(token_counts.values())

    result = {
        'char_count': float(n_chars),
        'token_count': float(n_tokens),
        'unique_token_count': float(len(token_counts)),
        'sentence_count': float(len(sentences)),
        'paragraph_count': float(len(paragraphs)),
        'line_count': float(len(lines)),
        'char_entropy': _entropy(char_counts.values()),
        'token_entropy': token_entropy,
        'normalized_token_entropy': token_entropy / max(math.log2(max(n_tokens, 2)), 1),
        'word_type_token_ratio': len(token_counts) / tokens_denom,
        'word_hapax_fraction': sum(count == 1 for count in token_counts.values()) / tokens_denom,
        'word_repeat_fraction': sum(count for count in token_counts.values() if count > 1) / tokens_denom,
        'max_word_frequency_fraction': max(token_counts.values(), default=0) / tokens_denom,
        'adjacent_word_repeat_rate': sum(a == b for a, b in zip(folded, folded[1:])) / max(n_tokens - 1, 1),
        'alphabetic_char_fraction': alphabetic_chars / chars_denom,
        'uppercase_char_fraction': sum(c.isupper() for c in text) / chars_denom,
        'uppercase_among_letters': sum(c.isupper() for c in text) / max(alphabetic_chars, 1),
        'lowercase_char_fraction': sum(c.islower() for c in text) / chars_denom,
        'digit_char_fraction': sum(c.isdigit() for c in text) / chars_denom,
        'nonascii_char_fraction': sum(ord(c) > 127 for c in text) / chars_denom,
        'punctuation_char_fraction': punctuation_count / chars_denom,
        'titlecase_token_fraction': sum(t.istitle() for t in alphabetic_tokens) / alpha_tokens_denom,
        'uppercase_token_fraction': sum(t.isupper() for t in alphabetic_tokens) / alpha_tokens_denom,
        'lowercase_token_fraction': sum(t.islower() for t in alphabetic_tokens) / alpha_tokens_denom,
        'numeric_token_fraction': sum(t.isdigit() for t in tokens) / tokens_denom,
        'mixed_alnum_token_fraction': sum(any(c.isdigit() for c in t) and any(c.isalpha() for c in t) for t in tokens) / tokens_denom,
        'apostrophe_token_fraction': sum("'" in t or '’' in t for t in tokens) / tokens_denom,
        'whitespace_char_fraction': sum(c.isspace() for c in text) / chars_denom,
        'newline_per_char': text.count('\n') / chars_denom,
        'tab_per_char': text.count('\t') / chars_denom,
        'carriage_return_per_char': text.count('\r') / chars_denom,
        'multiple_space_run_per_char': len(re.findall(r' {2,}', text)) / chars_denom,
        'blank_line_fraction': sum(not line.strip() for line in lines) / lines_denom,
        'indented_line_fraction': sum(bool(re.match(r'^[ \t]+\S', line)) for line in lines) / lines_denom,
        'trailing_whitespace_line_fraction': sum(bool(re.search(r'[ \t]+$', line)) for line in lines) / lines_denom,
        'markdown_heading_line_fraction': sum(bool(re.match(r'^\s{0,3}#{1,6}\s+', line)) for line in lines) / lines_denom,
        'markdown_list_line_fraction': sum(bool(re.match(r'^\s*(?:[-*+]\s+|\d+[.)]\s+)', line)) for line in lines) / lines_denom,
        'markdown_quote_line_fraction': sum(bool(re.match(r'^\s*>', line)) for line in lines) / lines_denom,
        'code_fence_count': float(len(re.findall(r'```|~~~', text))),
        'markdown_link_per_token': len(re.findall(r'\[[^\]\n]+\]\([^\)\n]+\)', text)) / tokens_denom,
        'code_operator_per_char': len(re.findall(r'==|!=|<=|>=|=>|:=|\+=|-=|&&|\|\|', text)) / chars_denom,
        'punctuation_run_per_char': len(re.findall(r'[!?.,;:]{2,}', text)) / chars_denom,
        'ellipsis_per_char': (text.count('...') + text.count('…')) / chars_denom,
        'sentence_terminal_per_token': len(re.findall(r'[.!?。！？]', text)) / tokens_denom,
    }
    for n in (2, 3):
        grams = [tuple(folded[i:i + n]) for i in range(max(n_tokens - n + 1, 0))]
        result[f'duplicate_{"bi" if n == 2 else "tri"}gram_rate'] = (len(grams) - len(set(grams))) / max(len(grams), 1)
    for name, characters in PUNCTUATION_GROUPS.items():
        count = sum(char_counts[character] for character in characters)
        result[f'{name}_count'] = float(count)
        result[f'{name}_per_char'] = count / chars_denom
    word_lengths = [len(token) for token in tokens]
    result.update(_summary(word_lengths, 'token_length'))
    for length in range(1, 9):
        result[f'token_length_{length}_fraction'] = sum(n == length for n in word_lengths) / tokens_denom
    result['token_length_9_to_12_fraction'] = sum(9 <= n <= 12 for n in word_lengths) / tokens_denom
    result['token_length_over_12_fraction'] = sum(n > 12 for n in word_lengths) / tokens_denom
    sentence_lengths = [len(TOKEN_RE.findall(part)) for part in sentences]
    result.update(_summary(sentence_lengths, 'sentence_tokens'))
    result['sentence_token_length_cv'] = result['sentence_tokens_std'] / max(result['sentence_tokens_mean'], 1)
    result.update(_summary([len(TOKEN_RE.findall(part)) for part in paragraphs], 'paragraph_tokens'))
    result.update(_summary([len(line) for line in lines], 'line_chars'))
    result.update(_summary([len(run) for run in re.findall(r'\s+', text)], 'whitespace_run'))
    return result


FEATURE_NAMES = tuple(extract_style_features('').keys())


class StyleFeatures(TransformerMixin, BaseEstimator):
    """Scikit-learn transformer with no corpus-dependent fitted parameters."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [list(extract_style_features(text).values()) for text in X]
        return np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)


def candidate_configs() -> list[dict]:
    """Five predeclared configurations; no validation-driven expansion."""
    return [
        {'name': 'style_hgb15', 'kind': 'hist_gradient_boosting', 'params': {
            'max_iter': 250, 'max_leaf_nodes': 15, 'learning_rate': 0.06,
            'min_samples_leaf': 30, 'l2_regularization': 5.0}},
        {'name': 'style_hgb31', 'kind': 'hist_gradient_boosting', 'params': {
            'max_iter': 250, 'max_leaf_nodes': 31, 'learning_rate': 0.06,
            'min_samples_leaf': 30, 'l2_regularization': 5.0}},
        {'name': 'style_hgb15_regularized', 'kind': 'hist_gradient_boosting', 'params': {
            'max_iter': 350, 'max_leaf_nodes': 15, 'learning_rate': 0.04,
            'min_samples_leaf': 45, 'l2_regularization': 15.0}},
        {'name': 'style_extra_leaf2', 'kind': 'extra_trees', 'params': {
            'n_estimators': 400, 'max_features': 0.8, 'min_samples_leaf': 2,
            'max_depth': None}},
        {'name': 'style_extra_leaf5', 'kind': 'extra_trees', 'params': {
            'n_estimators': 400, 'max_features': 0.8, 'min_samples_leaf': 5,
            'max_depth': 24}},
    ]


def build_pipeline(config: dict) -> Pipeline:
    """Build an unfitted style pipeline; caller enforces protocol boundaries."""
    params = dict(config['params'])
    if config['kind'] == 'hist_gradient_boosting':
        estimator = HistGradientBoostingClassifier(
            **params, early_stopping=False, random_state=12,
        )
    elif config['kind'] == 'extra_trees':
        estimator = ExtraTreesClassifier(**params, n_jobs=2, random_state=12)
    else:
        raise ValueError(f"Unknown style model kind: {config['kind']!r}")
    return Pipeline([('style_features', StyleFeatures()), ('classifier', estimator)])


if __name__ == '__main__':
    from models.text_generalization.common import run_candidates

    run_candidates('style', candidate_configs(), build_pipeline)
