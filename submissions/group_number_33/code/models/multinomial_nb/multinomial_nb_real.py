"""Custom binary SAMME.R-style boosting of fixed-feature MultinomialNB learners.

This implements the real-boost update from historical sklearn 1.5; it does not
request the removed SAMME.R option from installed sklearn. ``real_log_odds``
uses the canonical equal-coefficient log-odds pool. ``equal_probability_average``
is a separate custom prediction rule over the same real-boosted NB learners.

Reference: https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.X/sklearn/ensemble/_weight_boosting.py

Run selected settings: python3 -m models.multinomial_nb.multinomial_nb_real
Verify the selected settings on inner CV first: python3 -m models.multinomial_nb.multinomial_nb_real --verify-cv
"""

import argparse
import json
from numbers import Integral
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted, check_non_negative, check_X_y

from .multinomial_nb_ensemble import MeanWeightNB
from .multinomial_nb import ROOT, REFERENCE_DIR, SEED, run_experiment
from preprocess import split_validation_set, tfidf_data


EPSILON = np.finfo(np.float64).eps

# CLI recipe selected using training-only five-fold CV. Constructor defaults
# retain the previous 320-learner baseline for explicit comparisons.
SELECTED_CONFIG = dict(alpha=1500., learning_rate=1., n_estimators=1600,
                       aggregation="equal_probability_average", random_state=SEED)
SELECTION_HISTORY = REFERENCE_DIR / "high_smoothing" / "metadata.json"


def clipped_log_odds(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Expected two columns of binary probabilities")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and between zero and one")
    clipped = np.maximum(probabilities, EPSILON)
    return np.log(clipped[:, 1]) - np.log(clipped[:, 0])


def update_weights(weights, labels, probabilities, learning_rate):
    """Stable normalized w*exp(-lr*y_signed*log(p1/p0)/2), preserving zeros."""
    weights = np.asarray(weights, dtype=float)
    labels = np.asarray(labels)
    log_odds = clipped_log_odds(probabilities)
    if weights.ndim != 1 or weights.shape != labels.shape or weights.shape != log_odds.shape:
        raise ValueError("Weights, labels and probability rows must match")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Weights must be finite and nonnegative with positive support")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("Require positive finite learning rate and binary labels")
    positive = weights > 0
    log_weights = np.log(weights[positive]) - .5*learning_rate*(2*labels[positive]-1)*log_odds[positive]
    result = np.zeros_like(weights)
    result[positive] = np.exp(log_weights-logsumexp(log_weights))
    if not np.isfinite(result).all() or not np.isclose(result.sum(), 1.):
        raise ValueError("Real-boost weight update is not finite and normalized")
    return result


class RealNBEnsemble(ClassifierMixin, BaseEstimator):
    """Clone-compatible binary real-boosted MNB with unchanged input columns.

    Each learner uses scalar smoothing, learned class priors, and incoming
    sample weights rescaled to nonzero-row mean one by MeanWeightNB. Initially
    zero-weight rows stay zero. Unlike discrete SAMME, weighted error >= .5
    does not stop fitting. Perfect weighted classification stops early.

    ``random_state`` is retained for the shared experiment interface; MNB and
    this update are deterministic and do not sample training rows or columns.
    """

    coefficient_power = 0.
    temperature = 1.
    training_algorithm = "SAMME.R-style"

    def __init__(self, alpha=300., learning_rate=1., n_estimators=320,
                 aggregation="equal_probability_average", random_state=12):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.aggregation = aggregation
        self.random_state = random_state

    def _check_parameters(self):
        if not np.isfinite([self.alpha, self.learning_rate]).all() or self.alpha <= 0 or self.learning_rate <= 0:
            raise ValueError("Alpha and learning rate must be positive and finite")
        if not isinstance(self.n_estimators, Integral) or isinstance(self.n_estimators, bool) or self.n_estimators <= 0:
            raise ValueError("Number of estimators must be a positive integer")
        if self.aggregation not in {"real_log_odds", "equal_probability_average"}:
            raise ValueError("Aggregation must be real_log_odds or equal_probability_average")

    def fit(self, X, y, sample_weight=None):
        self._check_parameters()
        X, labels = check_X_y(X, y, accept_sparse="csr", dtype=np.float64)
        check_non_negative(X, "RealNBEnsemble")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Require human=0 and AI=1 labels with both classes present")
        weights = (np.ones(X.shape[0]) if sample_weight is None
                   else np.array(sample_weight, dtype=float, copy=True))
        if weights.shape != (X.shape[0],) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Sample weights must be matching, finite, nonnegative values")
        nonzero = np.asarray(X.sum(axis=1)).ravel() > 0
        weights[~nonzero] = 0.
        if not np.isfinite(weights.sum()) or weights.sum() <= 0:
            raise ValueError("Nonzero-feature rows must have positive finite total weight")
        if any(weights[labels == label].sum() <= 0 for label in [0, 1]):
            raise ValueError("Each class must have positive weight on nonzero-feature rows")
        zero_mask = weights == 0
        weights /= weights.sum()
        learners, errors = [], []
        for stage in range(1, self.n_estimators+1):
            weights = np.maximum(weights, EPSILON)
            weights[zero_mask] = 0.
            learner = MeanWeightNB(alpha=self.alpha, prior_power=0., fit_prior=True).fit(X, labels, sample_weight=weights)
            train_probability = learner.predict_proba(X)
            incorrect = learner.classes_[train_probability.argmax(axis=1)] != labels
            error = float(np.average(incorrect, weights=weights))
            learners.append(learner)
            errors.append(error)
            if error <= 0:
                break
            if stage < self.n_estimators:
                weights = update_weights(weights, labels, train_probability, self.learning_rate)
        self.boost_ = SimpleNamespace(estimators_=learners, estimator_weights_=np.ones(len(learners)),
                                      estimator_errors_=np.asarray(errors))
        self.classes_ = learners[0].classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["boost_", "classes_", "n_features_in_"])
        self._check_parameters()
        X = check_array(X, accept_sparse="csr", dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, received {X.shape[1]}")
        learners = self.boost_.estimators_
        if not learners:
            raise ValueError("At least one fitted learner is required")
        score = np.zeros(X.shape[0])
        for learner in learners:
            probability = learner.predict_proba(X)
            if self.aggregation == "real_log_odds":
                score += clipped_log_odds(probability)
            else:
                score += probability[:, 1]
        score /= len(learners)
        positive = expit(score) if self.aggregation == "real_log_odds" else score
        return np.column_stack([1-positive, positive])

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-cv", action="store_true", help="verify previously selected parameters on training-only five-fold CV")
    args = parser.parse_args()
    estimator = RealNBEnsemble(**SELECTED_CONFIG)
    tuning = None
    if args.verify_cv:
        start = perf_counter()
        train, _ = split_validation_set(ROOT / "data/train.csv", random_state=SEED)
        scores = []
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(train.text, train.label), 1):
            X, V, _ = tfidf_data(train.text.iloc[tr], train.text.iloc[va], train.text.iloc[va])
            model = RealNBEnsemble(**estimator.get_params()).fit(X, train.label.iloc[tr])
            score = float(roc_auc_score(train.label.iloc[va], model.predict_proba(V)[:, 1]))
            scores.append(score)
            print(f"Selected real-boost inner fold {fold}/5: AUROC={score:.8f}", flush=True)
        record = {**estimator.get_params(), "mean_cv_auroc": float(np.mean(scores)),
                  "std_cv_auroc": float(np.std(scores, ddof=1)),
                  **{f"fold_{i}_auroc": value for i, value in enumerate(scores, 1)}}
        tuning = dict(selection_metric="Previously selected by mean inner-CV AUROC; this run verifies that fixed configuration",
                      selected_config=estimator.get_params(), selected_mean_cv_auroc=record["mean_cv_auroc"],
                      inner_folds=5, inner_seed=SEED, training_rows_used_for_search=len(train),
                      search_seconds=0., verification_seconds=perf_counter()-start,
                      protocol="Shared TF-IDF refit independently on each inner training fold; no outer validation used",
                      candidates=[record])
        history_path = SELECTION_HISTORY
        if history_path.is_file():
            history = json.loads(history_path.read_text())
            keys = ["alpha", "learning_rate", "n_estimators", "aggregation"]
            matches = [candidate for candidate in history["candidates"]
                       if all(candidate[key] == estimator.get_params()[key] for key in keys)]
            if len(matches) != 1:
                raise ValueError("Historical selection does not uniquely match the current configuration")
            best = matches[0]
            if best["mean_cv_auroc"] < max(candidate["mean_cv_auroc"] for candidate in history["candidates"]) - 1e-12:
                raise ValueError("Selected configuration is not a highest-scoring historical candidate")
            np.testing.assert_allclose(scores, [best[f"fold_{i}_auroc"] for i in range(1, 6)], rtol=0, atol=1e-12)
            tuning["candidates"] = history["candidates"]
            tuning["search_seconds"] = history["search_seconds"]
            tuning["historical_selection_record"] = str(history_path.relative_to(ROOT))
    run_experiment(estimator=estimator, tuning=tuning)


if __name__ == "__main__":
    main()
