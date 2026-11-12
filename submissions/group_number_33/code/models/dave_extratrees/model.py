"""Portable raw-text classifier for the retained ExtraTrees recipe."""
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.validation import check_is_fitted

from ..text_generalization.style_search import FEATURE_NAMES, StyleFeatures


CHAMPION_PARAMS = {
    'n_estimators': 800, 'max_features': 0.8, 'min_samples_leaf': 2,
    'max_depth': None, 'criterion': 'gini', 'random_state': 12, 'n_jobs': 2,
    'bootstrap': True, 'max_samples': 0.8, 'class_weight': 'balanced_subsample',
}


def _texts(values):
    if isinstance(values, str):
        raise ValueError('Pass a sequence of passages, not a single string')
    texts=list(values)
    if not texts or not all(isinstance(text,str) for text in texts):
        raise ValueError('Expected a nonempty sequence of raw text strings')
    return texts


class DaveExtraTreesClassifier(ClassifierMixin, BaseEstimator):
    """145 fixed statistics -> ExtraTrees -> clipped positive-class probability.

    The feature extractor does not fit a vocabulary or corpus statistics.
    Its definition is shared with the already-published style method; this
    class does not import experimental search modules or generated artifacts.
    """
    def __init__(self,params=None):
        self.params=params

    def fit(self,X,y):
        texts=_texts(X)
        labels=np.asarray(y)
        if labels.ndim!=1 or len(labels)!=len(texts) or set(np.unique(labels))!={0,1}:
            raise ValueError('Expected one binary label per passage, including both classes')
        self.features_=StyleFeatures()
        features=self.features_.fit_transform(texts)
        if features.shape!=(len(texts),145) or not np.isfinite(features).all():
            raise ValueError('Expected 145 finite statistical features per passage')
        self.classifier_=ExtraTreesClassifier(**deepcopy(CHAMPION_PARAMS if self.params is None else self.params))
        self.classifier_.fit(features,labels)
        np.testing.assert_array_equal(self.classifier_.classes_,[0,1])
        self.classes_=self.classifier_.classes_.copy()
        self.n_training_rows_=len(texts)
        self.n_features_in_=len(FEATURE_NAMES)
        return self

    def predict_proba(self,X):
        check_is_fitted(self,'classifier_')
        texts=_texts(X)
        features=self.features_.transform(texts)
        if features.shape!=(len(texts),145) or not np.isfinite(features).all():
            raise ValueError('Expected 145 finite statistical features per passage')
        positive=int(np.flatnonzero(self.classes_==1).item())
        p=np.asarray(self.classifier_.predict_proba(features)[:,positive],dtype=float)
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
            raise ValueError('Invalid forest probabilities')
        p=np.clip(p,1e-7,1-1e-7)
        return np.column_stack([1-p,p])

    def predict(self,X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
