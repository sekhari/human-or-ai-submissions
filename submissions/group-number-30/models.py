"""Models for the supplied 75–350-word passages; no test text is used to fit features."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

SEED = 20261023


def word_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                           sublinear_tf=True, dtype=np.float64)


def build_model(config, *, seed=SEED):
    """All learned text preprocessing is inside the returned estimator."""
    kind = config.get('kind')
    if kind == 'char_style_blend':
        if set(config) != {'kind', 'char_C', 'word_style_C', 'weights'}:
            raise ValueError('Blend config requires char_C, word_style_C and weights')
        weights = np.asarray(config['weights'], dtype=float)
        if weights.shape != (2,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('Blend needs two finite nonnegative weights with positive sum')
        return VotingClassifier(estimators=[
            ('character', build_model({'kind': 'char_lr', 'C': config['char_C']}, seed=seed)),
            ('word_style', build_model({'kind': 'word_style_lr', 'C': config['word_style_C']}, seed=seed))],
            voting='soft', weights=weights.tolist(), n_jobs=1)
    allowed = {'kind', 'alpha'} if kind == 'word_nb' else {'kind', 'C'}
    if set(config) - allowed:
        raise ValueError(f'Unknown configuration keys: {sorted(set(config) - allowed)}')
    c = float(config.get('C', 1.0))
    alpha = float(config.get('alpha', 1.0))
    if not np.isfinite(c) or c <= 0 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('C and alpha must be finite positive numbers')

    def logistic():
        return LogisticRegression(C=c, solver='liblinear', max_iter=2000, random_state=seed)

    if kind == 'word_lr':
        return Pipeline([('features', word_vectorizer()), ('classifier', logistic())])
    if kind == 'char_lr':
        return Pipeline([('features', TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                         min_df=3, max_features=80000, sublinear_tf=True, dtype=np.float64)),
                         ('classifier', logistic())])
    if kind == 'word_nb':
        return Pipeline([('features', word_vectorizer()), ('classifier', ComplementNB(alpha=alpha))])
    if kind == 'word_svm':
        classifier = CalibratedClassifierCV(
            LinearSVC(C=c, dual='auto', max_iter=10000, random_state=seed),
            method='sigmoid', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed), n_jobs=1)
        # Calibration uses only the outer training partition. Its internal folds
        # share this training-only vocabulary/IDF; validation never enters fit.
        return Pipeline([('features', word_vectorizer()), ('classifier', classifier)])

    # Core comparisons do not require a POS tagger or optional feature module.
    if kind not in {'style_lr', 'pos_lr', 'surprisal_lr', 'word_style_lr'}:
        raise ValueError(f'Unknown model kind: {kind}')
    from structure import StyleFeatures, POSTransformer, ClassConditionalPOSSurprisal

    def style():
        return Pipeline([('extract', StyleFeatures()), ('scale', StandardScaler())])

    if kind == 'style_lr':
        return Pipeline([('features', style()), ('classifier', logistic())])
    if kind == 'pos_lr':
        return Pipeline([('tags', POSTransformer()), ('features', TfidfVectorizer(
            lowercase=False, token_pattern=r'\S+', ngram_range=(1, 3), min_df=2,
            max_features=20000, sublinear_tf=True)), ('classifier', logistic())])
    if kind == 'surprisal_lr':
        return Pipeline([('features', ClassConditionalPOSSurprisal(order=3, alpha=.5, cv=3, random_state=seed)),
                         ('scale', StandardScaler()), ('classifier', logistic())])
    if kind == 'word_style_lr':
        return Pipeline([('features', FeatureUnion([('word', word_vectorizer()), ('style', style())],
                         transformer_weights={'word': 1., 'style': .1})), ('classifier', logistic())])
    raise ValueError(f'Unknown model kind: {kind}')
