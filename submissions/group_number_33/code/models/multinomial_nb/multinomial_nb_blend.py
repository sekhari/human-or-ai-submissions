"""Blend two fixed-feature MultinomialNB components using training-only scales.

Run selected settings: python3 -m models.multinomial_nb.multinomial_nb_blend
Verify five-fold training-CV reproduction first:
    python3 -m models.multinomial_nb.multinomial_nb_blend --verify-cv

The main component averages real-boosted NB probabilities. The complementary
component's smoothing and sample-weight recipe were selected jointly with the
blend weight by training-only cross-validation.
Both consume the same unchanged input matrix. The retained default uses
clipped-probability logits for both components. Scores are divided by their
respective training population SD, without centering, then blended. Raw alternate
log odds with an optional symmetric cap remain available for reproduction.
"""

import argparse
import json
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted, check_non_negative, check_X_y
from threadpoolctl import threadpool_limits

from .multinomial_nb import ROOT, REFERENCE_DIR, SEED, fit_weighted_nb, run_experiment, sha256
from .multinomial_nb_real import EPSILON, RealNBEnsemble
from preprocess import split_validation_set, tfidf_data


ALT_CONFIG = dict(alpha=.3, sample_power=1., prior_power=-96., ai_weight=1., weight_cap=2.)
SELECTION_HISTORY = REFERENCE_DIR / "selected_blend" / "candidate_results.csv"
SELECTION_METADATA = SELECTION_HISTORY.with_name("metadata.json")


def probability_logits(probabilities):
    """Match the selected experiment's clipped positive-class logit transform."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0 or not np.isfinite(probabilities).all():
        raise ValueError("Require nonempty finite one-dimensional probabilities")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between zero and one")
    clipped = np.clip(probabilities, EPSILON, 1-EPSILON)
    return np.log(clipped)-np.log1p(-clipped)


def training_logit_std(training_probabilities):
    """Estimate the population SD only from the fitting rows' predictions."""
    scale = float(probability_logits(training_probabilities).std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training logits must have positive finite standard deviation")
    return scale


def blend_probability(main_logits, alternative_logits, weight):
    """Combine already scaled logits with the selected convex mixture."""
    main_logits = np.asarray(main_logits, dtype=float)
    alternative_logits = np.asarray(alternative_logits, dtype=float)
    if main_logits.ndim != 1 or alternative_logits.shape != main_logits.shape:
        raise ValueError("Blend inputs must have matching one-dimensional shapes")
    if not np.isfinite(main_logits).all() or not np.isfinite(alternative_logits).all():
        raise ValueError("Blend logits must be finite")
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Blend weight must be finite and between zero and one")
    return expit((1-weight)*main_logits+weight*alternative_logits)


class NBScoreBlend(ClassifierMixin, BaseEstimator):
    """Clone-compatible blend of two NB components with training-only scaling."""

    aggregation = "training_scaled_logit_blend"
    training_algorithm = "SAMME.R-style MultinomialNB plus weighted MultinomialNB score blend"
    extra_source_files = ("models/multinomial_nb/multinomial_nb_blend.py",)

    def __init__(self, alpha=1500., learning_rate=1., n_estimators=1600,
                 blend_weight=.25, random_state=12, alternate_score_mode="probability_logit",
                 alternate_logit_cap=None):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.blend_weight = blend_weight
        self.random_state = random_state
        self.alternate_score_mode = alternate_score_mode
        self.alternate_logit_cap = alternate_logit_cap

    def _check_parameters(self):
        if not np.isfinite([self.alpha, self.learning_rate, self.blend_weight]).all():
            raise ValueError("Alpha, learning rate and blend weight must be finite")
        if self.alpha <= 0 or self.learning_rate <= 0 or not 0 <= self.blend_weight <= 1:
            raise ValueError("Require positive alpha/learning rate and blend weight between zero and one")
        if not isinstance(self.n_estimators, Integral) or isinstance(self.n_estimators, bool) or self.n_estimators <= 0:
            raise ValueError("Number of estimators must be a positive integer")
        if self.alternate_score_mode not in {"raw_log_odds", "probability_logit"}:
            raise ValueError("Alternate score mode must be raw_log_odds or probability_logit")
        if self.alternate_logit_cap is not None and (not np.isfinite(self.alternate_logit_cap) or self.alternate_logit_cap <= 0):
            raise ValueError("Alternate logit cap must be positive and finite, or None")
        if self.alternate_score_mode == "probability_logit" and self.alternate_logit_cap is not None:
            raise ValueError("Legacy probability-logit scores require alternate_logit_cap=None")

    def alternate_scores(self, component, X):
        """Apply the same alternate score definition during fitting and prediction."""
        self._check_parameters()
        if self.alternate_score_mode == "probability_logit":
            return probability_logits(component.predict_proba(X)[:, 1])
        log_probabilities = component.predict_log_proba(X)
        scores = log_probabilities[:, 1]-log_probabilities[:, 0]
        if self.alternate_logit_cap is not None:
            scores = np.clip(scores, -self.alternate_logit_cap, self.alternate_logit_cap)
        if not np.isfinite(scores).all():
            raise ValueError("Alternate scores must be finite")
        return scores

    def fit(self, X, y):
        self._check_parameters()
        X, labels = check_X_y(X, y, accept_sparse="csr", dtype=np.float64)
        check_non_negative(X, "NBScoreBlend")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Require human=0 and AI=1 labels with both classes present")
        main = RealNBEnsemble(alpha=self.alpha, learning_rate=self.learning_rate,
                              n_estimators=self.n_estimators,
                              aggregation="equal_probability_average",
                              random_state=self.random_state).fit(X, labels)
        alternate = fit_weighted_nb(X, labels, **ALT_CONFIG)
        components = dict(main=main, weighted_single=alternate)
        # No validation/test argument exists here: these scales use fitting rows only.
        alternate_scale = float(self.alternate_scores(alternate, X).std(ddof=0))
        if not np.isfinite(alternate_scale) or alternate_scale <= 0:
            raise ValueError("Alternate training scores must have positive finite standard deviation")
        scales = dict(main=training_logit_std(main.predict_proba(X)[:, 1]),
                      weighted_single=alternate_scale)
        self.components_ = components
        self.logit_scales_ = scales
        self.classes_ = main.classes_.copy()
        self.n_features_in_ = X.shape[1]
        self.n_training_rows_ = X.shape[0]
        self.alternate_parameters_ = ALT_CONFIG.copy()
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["components_", "logit_scales_", "classes_", "n_features_in_"])
        self._check_parameters()
        X = check_array(X, accept_sparse="csr", dtype=np.float64)
        check_non_negative(X, "NBScoreBlend")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, received {X.shape[1]}")
        main_score = probability_logits(self.components_["main"].predict_proba(X)[:, 1])/self.logit_scales_["main"]
        alternate_score = self.alternate_scores(self.components_["weighted_single"], X)/self.logit_scales_["weighted_single"]
        positive = blend_probability(main_score, alternate_score, self.blend_weight)
        return np.column_stack([1-positive, positive])

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]

    def get_experiment_details(self, run_dir):
        """Persist the actual smoothing vector and report nested component provenance."""
        check_is_fitted(self, ["components_", "logit_scales_", "n_training_rows_"])
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        vector_file = "weighted_single_smoothing.npy"
        alternate = self.components_["weighted_single"]
        np.save(run_dir / vector_file, np.asarray(alternate.alpha, dtype=float))
        main = self.components_["main"]
        main_count = len(main.boost_.estimators_)
        probability_clip = dict(main=[EPSILON, 1-EPSILON],
                                weighted_single=([EPSILON, 1-EPSILON] if self.alternate_score_mode == "probability_logit" else None))
        alternate_definition = ("logit(clip(p_AI, epsilon, 1-epsilon))" if self.alternate_score_mode == "probability_logit"
                                else "predict_log_proba[:,1] - predict_log_proba[:,0], symmetrically clipped only when raw_logit_cap is not None")
        components = {
            "main": dict(estimator_class=type(main).__name__, parameters=main.get_params(),
                         n_estimators_fitted=main_count, n_features=self.n_features_in_,
                         score_mode="probability_logit", score_definition="logit(clip(mean NB p_AI, epsilon, 1-epsilon)); no extra logit cap",
                         probability_clip=probability_clip["main"], raw_logit_cap=None,
                         training_score_std=self.logit_scales_["main"],
                         prediction_weighting="Equal probability average over actual fitted NB learners",
                         training_weighting="SAMME.R-style exp(-0.5*learning_rate*y_signed*log(p1/p0)); zero rows excluded; each learner rescales positive-support sample weights to nonzero-row mean one",
                         smoothing="Scalar alpha from main parameters; learned class priors"),
            "weighted_single": dict(estimator_class=type(alternate).__name__,
                                    parameters=self.alternate_parameters_.copy(),
                                    n_estimators_fitted=1, n_features=self.n_features_in_,
                                    score_mode=self.alternate_score_mode, score_definition=alternate_definition,
                                    probability_clip=probability_clip["weighted_single"],
                                    raw_logit_cap=self.alternate_logit_cap,
                                    training_score_std=self.logit_scales_["weighted_single"],
                                    fit_prior=False, smoothing_vector_file=vector_file,
                                    smoothing_vector_sha256=sha256(run_dir / vector_file),
                                    smoothing_formula="alpha_j = parameters.alpha * feature_mass_j**parameters.prior_power / mean(feature_mass**parameters.prior_power); feature_mass_j = sum_i X_ij on fitting rows",
                                    sample_weight_formula="For nonzero rows: (row_mass / mean_nonzero(row_mass))**parameters.sample_power, normalize to mean one; when parameters.weight_cap is not None, cap at that value and renormalize to mean one; multiply AI weights by parameters.ai_weight; zero-feature rows receive zero weight"),
        }
        return dict(ensemble_fitted_estimators=main_count+1, n_nb_learners_total=main_count+1,
                    ensemble_components=components, aggregation=self.aggregation,
                    training_algorithm=self.training_algorithm,
                    blend_weights=dict(main=1-self.blend_weight, weighted_single=self.blend_weight),
                    logit_scaling=dict(training_logit_stds=self.logit_scales_.copy(),
                                       rows=self.n_training_rows_, source="fitting-row predictions only",
                                       ddof=0, centered=False, probability_clip=probability_clip),
                    blend_formula="sigmoid((1-w)*main_score/sd_main + w*alternate_score/sd_alternate); score definitions are recorded per component",
                    notes=["Two NB components share the unchanged TF-IDF matrix; logit scales use fitting rows only. This score scaling is not fitted probability calibration."])


def load_selection_history(estimator):
    """Require a matching recorded winning recipe before expensive CV verification."""
    estimator._check_parameters()
    for path in [SELECTION_HISTORY, SELECTION_METADATA]:
        if not path.is_file():
            raise FileNotFoundError(f"Historical selection reference is required for --verify-cv: {path}")
    history = json.loads(SELECTION_METADATA.read_text())
    timing_field = next((field for field in ["elapsed_seconds", "elapsed_scoring_seconds_reconstructed"]
                         if field in history), None)
    if timing_field is None:
        raise ValueError("Historical search timing is missing; cannot report a verified run duration")
    try:
        search_seconds = float(history[timing_field])
    except (TypeError, ValueError) as error:
        raise ValueError("Historical search timing must be a finite nonnegative number") from error
    if not np.isfinite(search_seconds) or search_seconds < 0:
        raise ValueError("Historical search timing must be a finite nonnegative number")
    reconstructed = timing_field == "elapsed_scoring_seconds_reconstructed"
    history["verified_search_timing"] = dict(
        seconds=search_seconds, source_field=timing_field, reconstructed=reconstructed,
        note=("Reconstructed elapsed scoring duration from recorded timestamps; not a directly recorded timer value."
              if reconstructed else "Elapsed search duration recorded by the original experiment timer."),
        recovery_note=history.get("recovery_note"))
    frame = pd.read_csv(SELECTION_HISTORY)
    params = estimator.get_params()
    expected_main = dict(alpha=params["alpha"], learning_rate=params["learning_rate"],
                         n_estimators=params["n_estimators"], aggregation="equal_probability_average",
                         random_state=params["random_state"])
    if history["main_config"] != expected_main or history["weighted_single_config"] != ALT_CONFIG:
        raise ValueError("Historical component configuration does not match the selected blend")
    score_config = dict(mode=params["alternate_score_mode"], logit_cap=params["alternate_logit_cap"])
    historical_score_config = history.get("alternate_score_config", dict(mode="probability_logit", logit_cap=None))
    if historical_score_config != score_config:
        raise ValueError("Historical alternate score configuration does not match the selected blend")
    if history["seed"] != SEED or history["n_folds"] != 5 or params["random_state"] != SEED:
        raise ValueError("Historical split/fold protocol does not match the selected blend")
    matching_rows = ((frame.component == "weighted_single") & (frame.role == "blend") &
                     (frame.alternate_weight == params["blend_weight"]))
    # Newer searches compare several alternate recipes at each blend weight.
    # Older records omit these columns and identify the single recipe in metadata.
    for name, expected in ALT_CONFIG.items():
        column = f"alternate_{name}"
        if column in frame.columns:
            matching_rows &= frame[column] == expected
    for name in ["alpha", "learning_rate", "n_estimators"]:
        column = f"main_{name}"
        if column in frame.columns:
            matching_rows &= frame[column] == params[name]
    if params["alternate_score_mode"] != "probability_logit" and not {"alternate_score_mode", "alternate_logit_cap"} <= set(frame.columns):
        raise ValueError("Raw-score history requires explicit alternate score mode and cap columns")
    if "alternate_score_mode" in frame.columns:
        matching_rows &= frame.alternate_score_mode == params["alternate_score_mode"]
    if "alternate_logit_cap" in frame.columns:
        matching_rows &= (frame.alternate_logit_cap.isna() if params["alternate_logit_cap"] is None
                          else frame.alternate_logit_cap == params["alternate_logit_cap"])
    matches = frame[matching_rows]
    if len(matches) != 1:
        raise ValueError("Historical selection must uniquely match the selected blend")
    selected = matches.astype(object).where(pd.notna(matches), None).iloc[0].to_dict()
    score_columns = ["mean_cv_auroc"]+[f"fold_{fold}_auroc" for fold in range(1, 6)]
    if not np.isfinite(frame[score_columns].to_numpy()).all():
        raise ValueError("Historical CV scores must all be finite")
    if selected["mean_cv_auroc"] < frame.loc[frame.role != "component_reference", "mean_cv_auroc"].max()-1e-12:
        raise ValueError("Selected blend is not a highest-scoring historical candidate")
    np.testing.assert_allclose(np.mean([selected[f"fold_{fold}_auroc"] for fold in range(1, 6)]),
                               selected["mean_cv_auroc"], rtol=0, atol=1e-12)
    candidates = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    return history, candidates, selected


def verify_selected_cv(estimator):
    history, candidates, selected = load_selection_history(estimator)
    start = perf_counter()
    train, outer = split_validation_set(ROOT / "data/train.csv", random_state=SEED)
    del outer
    scores = []
    for fold, (training_indices, validation_indices) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=SEED).split(train.text, train.label), 1):
        X, V, duplicate = tfidf_data(train.text.iloc[training_indices],
                                     train.text.iloc[validation_indices], train.text.iloc[validation_indices])
        del duplicate
        fitted = NBScoreBlend(**estimator.get_params()).fit(X, train.label.iloc[training_indices])
        score = float(roc_auc_score(train.label.iloc[validation_indices], fitted.predict_proba(V)[:, 1]))
        np.testing.assert_allclose(score, selected[f"fold_{fold}_auroc"], rtol=0, atol=1e-12)
        scores.append(score)
        print(f"Selected NB blend inner fold {fold}/5: AUROC={score:.10f}; historical parity verified", flush=True)
    return dict(selection_metric="Fixed-recipe inner-CV reproduction; recipe retained using development holdout comparisons",
                selection_context="This recipe was originally proposed by training-only inner cross-validation, then retained after comparing the development holdout with later method variants. It is not the globally highest inner-CV candidate across those later variants. The development holdout is not an untouched final test.",
                retained_using_development_holdout=True,
                selected_config=estimator.get_params(), alternate_config=ALT_CONFIG.copy(),
                selected_mean_cv_auroc=float(np.mean(scores)), inner_folds=5, inner_seed=SEED,
                training_rows_used_for_search=len(train),
                search_seconds=history["verified_search_timing"]["seconds"],
                search_timing=history["verified_search_timing"],
                verification_seconds=perf_counter()-start,
                protocol="Shared TF-IDF refit per inner fold; both NB models and logit SDs fitted only on inner training rows; all five held-out fold scores match the recorded winner to 1e-12",
                historical_selection_record=str(SELECTION_HISTORY.relative_to(ROOT)),
                historical_selection_sha256=sha256(SELECTION_HISTORY),
                historical_metadata_sha256=sha256(SELECTION_METADATA), candidates=candidates,
                verified_fold_scores=scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-cv", action="store_true",
                        help="require and reproduce the recorded winning five-fold training-CV result before the final run")
    args = parser.parse_args()
    estimator = NBScoreBlend()
    with threadpool_limits(limits=1):
        tuning = verify_selected_cv(estimator) if args.verify_cv else None
        run_experiment(estimator=estimator, tuning=tuning)


if __name__ == "__main__":
    main()
