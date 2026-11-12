"""Reusable ensemble of frozen text-model recipes; no shared-code changes."""
import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .common import decision_scores


def build_component(family, config):
    if family == 'style':
        from .style_search import build_pipeline
    elif family == 'word':
        from .word_search import build_pipeline
    elif family == 'char':
        from .char_search import build_pipeline
    elif family == 'style_ablation':
        from .style_ablation import build_pipeline
    elif family == 'group_robust':
        from .group_robust import build_pipeline
    else:
        raise ValueError(f'Unsupported frozen model family: {family}')
    return build_pipeline(config)


class WeightedTextEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(self, components, aggregation='probability'):
        self.components = components
        self.aggregation = aggregation

    def fit(self, X, y):
        if self.aggregation not in {'probability', 'scaled_logit'}:
            raise ValueError('Unknown ensemble aggregation')
        weights = np.asarray([item['weight'] for item in self.components], dtype=float)
        if len(weights)==0 or not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1):
            raise ValueError('Require finite nonnegative weights summing to one')
        if set(np.unique(y)) != {0,1}:
            raise ValueError('Require human=0 and AI=1 training labels')
        self.fitted_components_=[]
        for item in self.components:
            model=build_component(item['family'],item['config']).fit(X,y)
            np.testing.assert_array_equal(model.classes_,[0,1])
            mean, std=0.,1.
            if self.aggregation=='scaled_logit':
                scores=decision_scores(model,X)
                mean,std=float(scores.mean()),float(scores.std())
                if not np.isfinite(std) or std<=0:raise ValueError('Constant training scores')
            self.fitted_components_.append(dict(model=model,weight=item['weight'],mean=mean,std=std))
        self.classes_=np.array([0,1])
        self.n_training_rows_=len(X)
        return self

    def predict_proba(self, X):
        check_is_fitted(self,'fitted_components_')
        score=np.zeros(len(X),dtype=float)
        for item in self.fitted_components_:
            model=item['model']
            values=(expit(decision_scores(model,X)) if self.aggregation=='probability'
                    else (decision_scores(model,X)-item['mean'])/item['std'])
            score+=item['weight']*values
        if self.aggregation=='scaled_logit':score=expit(score)
        if not np.isfinite(score).all() or (score<0).any() or (score>1).any():
            raise ValueError('Invalid final probabilities')
        return np.column_stack([1-score,score])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
