"""Boosted MultinomialNB learners over the unchanged shared TF-IDF features.

``samme`` uses sklearn's standard AdaBoost prediction rule. The other modes
are custom weighted probability or log-odds averages over the same learners
trained by SAMME; they are not standard AdaBoost probabilities or SAMME.R.

Run selected settings: python3 -m models.multinomial_nb.multinomial_nb_ensemble
Repeat training-only parameter selection: python3 -m models.multinomial_nb.multinomial_nb_ensemble --tune
"""

import argparse
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import (
    check_array, check_is_fitted, check_non_negative, check_X_y,
)

from preprocess import tfidf_data, split_validation_set
from .multinomial_nb import ROOT, SEED, run_experiment


def _training_data(X, y):
    X, y = check_X_y(X, y, accept_sparse="csr", dtype=np.float64)
    check_non_negative(X, "MultinomialNB ensemble")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Require human=0 and AI=1 labels, with both classes present")
    return X, y


def _weights(X, y, sample_weight):
    weights = (np.ones(X.shape[0]) if sample_weight is None
               else np.array(sample_weight, dtype=float, copy=True))
    if weights.shape != (X.shape[0],) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Sample weights must be matching, finite, nonnegative values")
    nonzero = np.asarray(X.sum(axis=1)).ravel() > 0
    weights[~nonzero] = 0.
    if not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("Nonzero-feature rows must have positive finite total weight")
    if any(weights[y == label].sum() <= 0 for label in [0, 1]):
        raise ValueError("Each class must have positive weight on nonzero-feature rows")
    return weights, nonzero


def initial_weights(X, weighted=False):
    """Optional inverse-row-mass weights, capped before final normalization."""
    X = check_array(X, accept_sparse="csr", dtype=np.float64)
    check_non_negative(X, "MultinomialNB ensemble")
    mass = np.asarray(X.sum(axis=1)).ravel()
    nonzero = mass > 0
    if not nonzero.any():
        raise ValueError("At least one training row must have nonzero features")
    weights = np.zeros(X.shape[0])
    weights[nonzero] = 1.
    if weighted:
        weights[nonzero] = (mass[nonzero] / mass[nonzero].mean()) ** -1.
        weights[nonzero] /= weights[nonzero].mean()
        weights[nonzero] = np.minimum(weights[nonzero], 2.)
        weights[nonzero] /= weights[nonzero].mean()
    return weights


class MeanWeightNB(ClassifierMixin, BaseEstimator):
    """MNB whose incoming sample weights have mean one on nonzero rows.

    Rescaling preserves relative AdaBoost weights while keeping alpha on the
    ordinary training-count scale. Empirical alpha statistics use training X
    only. This estimator never alters, selects, or rescales feature columns.
    """

    def __init__(self, alpha=1., prior_power=0., fit_prior=True):
        self.alpha = alpha
        self.prior_power = prior_power
        self.fit_prior = fit_prior

    def fit(self, X, y, sample_weight=None):
        X, y = _training_data(X, y)
        if not np.isfinite([self.alpha, self.prior_power]).all() or self.alpha <= 0:
            raise ValueError("Alpha must be positive and finite; prior power must be finite")
        mass = np.asarray(X.sum(axis=0)).ravel()
        if self.prior_power != 0 and np.any(mass <= 0):
            raise ValueError("Empirical smoothing requires positive training mass in every feature")
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                prior = mass ** self.prior_power
                smoothing = self.alpha * prior / prior.mean()
        except FloatingPointError as error:
            raise ValueError("Smoothing parameters produce invalid pseudo-counts") from error
        if not np.isfinite(smoothing).all() or np.any(smoothing <= 0):
            raise ValueError("Smoothing pseudo-counts must be positive and finite")
        weights, nonzero = _weights(X, y, sample_weight)
        weights[nonzero] *= nonzero.sum() / weights[nonzero].sum()
        self.model_ = MultinomialNB(alpha=smoothing, force_alpha=True,
                                   fit_prior=self.fit_prior).fit(X, y, sample_weight=weights)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)

    def predict_log_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_log_proba(X)


class NBEnsemble(ClassifierMixin, BaseEstimator):
    """SAMME-trained MNB ensemble with an explicit prediction aggregation.

    Constructor defaults preserve the original ensemble; the CLI uses the newer
    SELECTED_CONFIG recipe. Probability averaging
    is an arithmetic pool; log-odds averaging is a geometric pool. Both are
    custom inference rules using the fitted positive SAMME learner weights.
    """

    def __init__(self, alpha=30., prior_power=0., fit_prior=True,
                 initial_weighting=False, learning_rate=.25, n_estimators=40,
                 aggregation="log_odds_average", random_state=12,
                 coefficient_power=1., temperature=1.):
        self.alpha = alpha
        self.prior_power = prior_power
        self.fit_prior = fit_prior
        self.initial_weighting = initial_weighting
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.aggregation = aggregation
        self.random_state = random_state
        self.coefficient_power = coefficient_power
        self.temperature = temperature

    def _check_aggregation(self):
        if self.aggregation not in {"samme", "probability_average", "log_odds_average"}:
            raise ValueError("Aggregation must be samme, probability_average, or log_odds_average")
        if not np.isfinite(self.coefficient_power) or not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Require finite coefficient power and positive finite temperature")
        if self.aggregation != "probability_average" and self.temperature != 1.:
            raise ValueError("Temperature shaping is supported only before probability averaging")
        if self.aggregation == "samme" and self.coefficient_power != 1.:
            raise ValueError("Standard SAMME uses its original learner coefficients")

    def fit(self, X, y, sample_weight=None):
        self._check_aggregation()
        X, y = _training_data(X, y)
        external, _ = _weights(X, y, sample_weight)
        weights = initial_weights(X, self.initial_weighting) * external
        weights, _ = _weights(X, y, weights)
        base = MeanWeightNB(alpha=self.alpha, prior_power=self.prior_power,
                            fit_prior=self.fit_prior)
        boost = AdaBoostClassifier(estimator=base, n_estimators=self.n_estimators,
                                   learning_rate=self.learning_rate, random_state=self.random_state)
        # sklearn preserves initially zero weights, then logs them during SAMME
        # updates. Suppress only that expected warning from its specific module.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="^divide by zero encountered in log$",
                                    category=RuntimeWarning,
                                    module=r"^sklearn\.ensemble\._weight_boosting$")
            boost.fit(X, y, sample_weight=weights)
        self.boost_ = boost
        self.classes_ = boost.classes_
        self.n_features_in_ = boost.n_features_in_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["boost_", "classes_", "n_features_in_"])
        self._check_aggregation()
        X = check_array(X, accept_sparse="csr", dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, received {X.shape[1]}")
        if self.aggregation == "samme":
            return self.boost_.predict_proba(X)
        learners = self.boost_.estimators_
        weights = self.boost_.estimator_weights_[:len(learners)]
        if not len(learners) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Fitted learner weights must be finite, nonnegative, and have positive sum")
        if self.coefficient_power != 1.:
            # A common multiplier cancels in the weighted mean. Work in log
            # space so negative powers do not overflow on tiny coefficients.
            positive = weights > 0
            logs = np.log(weights[positive]) * self.coefficient_power
            transformed = np.zeros_like(weights)
            transformed[positive] = np.exp(logs - logs.max())
            weights = transformed
        score = np.zeros(X.shape[0])
        for learner, weight in zip(learners, weights):
            log_proba = learner.predict_log_proba(X)
            if self.aggregation == "probability_average":
                probabilities = (np.exp(log_proba[:, 1]) if self.temperature == 1.
                                 else expit((log_proba[:, 1] - log_proba[:, 0]) / self.temperature))
                score += weight * probabilities
            else:
                score += weight * (log_proba[:, 1] - log_proba[:, 0])
        score /= weights.sum()
        probability = score if self.aggregation == "probability_average" else expit(score)
        return np.column_stack([1-probability, probability])

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]

SELECTED_CONFIG = dict(alpha=300.0, prior_power=0.0, fit_prior=True,
                       initial_weighting=False, learning_rate=0.25,
                       n_estimators=160, aggregation="probability_average", random_state=SEED,
                       coefficient_power=0.0, temperature=0.03)


def select_ensemble_by_cv(texts, labels, fit_configs=None, stages=None, aggregations=None, n_splits=5):
    """Tune NB ensemble fitting and aggregation on outer-training rows only.

    Fit each boosting sequence once per fold and score its saved stages. Every
    feature extractor and initial sample-weight computation uses that fold's
    training texts only. The held-out 2,000 rows are never passed here.
    """
    if fit_configs is None:
        fit_configs = [dict(alpha=a, prior_power=0.0, fit_prior=True,
                            initial_weighting=False, learning_rate=lr)
                       for a in [3., 10., 30., 100., 300., 1000.]
                       for lr in [0.25, 0.5, 0.75, 1.0]]
    stages = [3, 5, 10, 20, 40] if stages is None else sorted(set(stages))
    aggregations = ["samme", "probability_average", "log_odds_average"] if aggregations is None else list(aggregations)
    if not fit_configs or not stages or not aggregations or len(texts) != len(labels):
        raise ValueError("Need fitting configurations, stages, aggregations and matching rows")
    if any(not isinstance(n, int) or n <= 0 for n in stages):
        raise ValueError("Stage counts must be positive integers")
    if not set(aggregations) <= {"samme", "probability_average", "log_odds_average"}:
        raise ValueError("Unsupported ensemble aggregation")
    started = perf_counter()
    scores = {(i, n, mode): [] for i in range(len(fit_configs)) for n in stages for mode in aggregations}
    stopping_counts = {i: [] for i in range(len(fit_configs))}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(cv.split(texts, labels), 1):
        X, V, _ = tfidf_data(texts.iloc[tr], texts.iloc[va], texts.iloc[va])
        y, v = labels.iloc[tr], labels.iloc[va]
        for i, config in enumerate(fit_configs):
            ensemble = NBEnsemble(**config, n_estimators=max(stages), aggregation="samme", random_state=SEED).fit(X, y)
            fitted = len(ensemble.boost_.estimators_)
            stopping_counts[i].append(fitted)
            vote = np.zeros(V.shape[0]); probability = np.zeros(V.shape[0]); log_odds = np.zeros(V.shape[0])
            total = 0.0; snapshots = {}
            for n, (estimator, weight) in enumerate(zip(ensemble.boost_.estimators_, ensemble.boost_.estimator_weights_), 1):
                lp = estimator.predict_log_proba(V)
                diff = lp[:, 1] - lp[:, 0]
                vote += weight * (2 * (diff > 0).astype(float) - 1)
                probability += weight * np.exp(lp[:, 1])
                log_odds += weight * diff
                total += weight
                if n in stages or n == fitted:
                    snapshots[n] = {"samme": expit(2 * vote / total),
                                    "probability_average": probability / total,
                                    "log_odds_average": expit(log_odds / total)}
            np.testing.assert_allclose(snapshots[fitted]["samme"], ensemble.boost_.predict_proba(V)[:, 1], atol=1e-12)
            for n in stages:
                for mode in aggregations:
                    scores[(i, n, mode)].append(float(roc_auc_score(v, snapshots[min(n, fitted)][mode])))
        print(f"Ensemble inner fold {fold}/{n_splits}: {len(fit_configs)} fits, {len(scores)} candidates", flush=True)
    records = [{**fit_configs[i], "n_estimators": n, "aggregation": mode,
                "mean_cv_auroc": float(np.mean(values)), "std_cv_auroc": float(np.std(values, ddof=1)),
                "fitted_estimators_by_fold": [min(n, count) for count in stopping_counts[i]],
                **{f"fold_{j}_auroc": value for j, value in enumerate(values, 1)}}
               for (i, n, mode), values in scores.items()]
    best = max(records, key=lambda r: r["mean_cv_auroc"])
    config_keys = list(fit_configs[0]) + ["n_estimators", "aggregation"]
    return dict(selection_metric="mean inner-CV AUROC",
                selected_config={**{k: best[k] for k in config_keys}, "random_state": SEED},
                selected_mean_cv_auroc=best["mean_cv_auroc"], inner_folds=n_splits, inner_seed=SEED,
                training_rows_used_for_search=len(texts), search_seconds=perf_counter()-started,
                protocol="Shared TF-IDF refit per inner training fold; boosting and smoothing use only fold-training rows",
                candidates=records)


def select_ensemble_inference_by_cv(texts, labels, fit_configs=None, stages=None,
                                  inference_configs=None, n_splits=5):
    """Select training and prediction parameters without the outer validation.

    Cache individual learners' predictions to compare stage counts, coefficient
    powers and confidence shaping efficiently. Temperature changes each base
    probability before averaging; it is not a fitted calibration model.
    """
    if fit_configs is None:
        fit_configs = [dict(alpha=a, prior_power=0., fit_prior=True,
                            initial_weighting=False, learning_rate=.25) for a in [300., 1000.]]
    stages = [80, 120, 160, 200, 240, 320] if stages is None else sorted(set(stages))
    if inference_configs is None:
        inference_configs = [dict(aggregation="probability_average", coefficient_power=q, temperature=t)
                             for q in [-.5, 0., 1.] for t in [.01, .03, .1, .3, 1., 3.]]
        inference_configs += [dict(aggregation="log_odds_average", coefficient_power=q, temperature=1.)
                              for q in [-.5, 0., 1.]]
    if not fit_configs or not stages or not inference_configs or len(texts) != len(labels):
        raise ValueError("Need configurations, stages and matching text/label rows")
    if any(not isinstance(n, int) or n <= 0 for n in stages):
        raise ValueError("Stage counts must be positive integers")
    for inference in inference_configs:
        NBEnsemble(**inference)._check_aggregation()
        if inference["aggregation"] == "samme":
            raise ValueError("Use select_ensemble_by_cv for standard SAMME")
    start = perf_counter()
    scores = {(i, n, j): [] for i in range(len(fit_configs)) for n in stages
              for j in range(len(inference_configs))}
    actual_counts = {i: [] for i in range(len(fit_configs))}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(cv.split(texts, labels), 1):
        X, V, _ = tfidf_data(texts.iloc[tr], texts.iloc[va], texts.iloc[va])
        y, v = labels.iloc[tr], labels.iloc[va]
        for i, config in enumerate(fit_configs):
            model = NBEnsemble(**config, n_estimators=max(stages), random_state=SEED).fit(X, y)
            learners = model.boost_.estimators_
            fitted = len(learners)
            actual_counts[i].append(fitted)
            lp = np.stack([learner.predict_log_proba(V) for learner in learners])
            logits = lp[:, :, 1] - lp[:, :, 0]
            probabilities = np.exp(lp[:, :, 1])
            weights = model.boost_.estimator_weights_[:fitted]
            for n in stages:
                used = min(n, fitted)
                for j, inference in enumerate(inference_configs):
                    positive = weights[:used] > 0
                    logs = np.log(weights[:used][positive]) * inference["coefficient_power"]
                    coefficients = np.zeros(used)
                    coefficients[positive] = np.exp(logs - logs.max())
                    coefficients /= coefficients.sum()
                    if inference["aggregation"] == "log_odds_average":
                        predicted = expit(np.sum(coefficients[:, None] * logits[:used], axis=0))
                    else:
                        values = (probabilities[:used] if inference["temperature"] == 1.
                                  else expit(logits[:used] / inference["temperature"]))
                        predicted = np.sum(coefficients[:, None] * values, axis=0)
                    scores[(i, n, j)].append(float(roc_auc_score(v, predicted)))
        print(f"Inference inner fold {fold}/{n_splits}: {len(fit_configs)} fits, {len(scores)} candidates", flush=True)
    records = [{**fit_configs[i], **inference_configs[j], "n_estimators": n,
                "mean_cv_auroc": float(np.mean(s)), "std_cv_auroc": float(np.std(s, ddof=1)),
                "fitted_estimators_by_fold": [min(n, count) for count in actual_counts[i]],
                **{f"fold_{k}_auroc": value for k, value in enumerate(s, 1)}}
               for (i, n, j), s in scores.items()]
    best = max(records, key=lambda r: r["mean_cv_auroc"])
    keys = list(fit_configs[0]) + list(inference_configs[0]) + ["n_estimators"]
    return dict(selection_metric="mean inner-CV AUROC",
                selected_config={**{k: best[k] for k in keys}, "random_state": SEED},
                selected_mean_cv_auroc=best["mean_cv_auroc"], inner_folds=n_splits, inner_seed=SEED,
                training_rows_used_for_search=len(texts), search_seconds=perf_counter()-start,
                protocol="Shared TF-IDF and boosting fit only on each inner training fold; inference settings chosen using inner validation only",
                candidates=records)


def main():
    parser = argparse.ArgumentParser(description="Fixed-feature MultinomialNB ensemble experiment")
    parser.add_argument("--tune", action="store_true", help="repeat training-only five-fold selection before held-out evaluation")
    args = parser.parse_args()
    tuning = None
    config = SELECTED_CONFIG
    if args.tune:
        training_rows, _ = split_validation_set(ROOT / "data/train.csv", random_state=SEED)
        tuning = select_ensemble_inference_by_cv(training_rows.text, training_rows.label)
        config = tuning["selected_config"]
        print(f"Selected {config}; mean inner-CV AUROC={tuning['selected_mean_cv_auroc']:.6f}", flush=True)
    run_experiment(estimator=NBEnsemble(**config), tuning=tuning)


if __name__ == "__main__":
    main()
