"""Run MultinomialNB using the team's unchanged preprocessing.

Run from the repository root: python3 -m models.multinomial_nb.multinomial_nb
Tune alpha on training-only inner CV: python3 -m models.multinomial_nb.multinomial_nb --tune
Tune smoothing and sample weights: python3 -m models.multinomial_nb.multinomial_nb --tune-weighted
Results are saved under models/multinomial_nb/outputs/experiments/<UTC run timestamp>/.
The shared function also transforms test.csv, but this experiment only evaluates
the labeled training and validation splits. No Kaggle submission is generated.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold

from preprocess import preprocess_data, split_validation_set, tfidf_data


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODEL_DIR / "outputs"
REFERENCE_DIR = MODEL_DIR / "references"
SEED = 12
THRESHOLD = 0.5


def calculate_metrics(labels, probabilities):
    """Score probabilities for AI (label 1); threshold only accuracy/F1."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape:
        raise ValueError("Labels and probabilities must be matching 1-D arrays")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both human (0) and AI (1) labels")
    predictions = (probabilities >= THRESHOLD).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_weighted_nb(X, labels, alpha, sample_power, prior_power, ai_weight=1.0, weight_cap=None):
    """Tune MNB smoothing and sample weights without changing X.

    Smoothing alpha_j is proportional to pooled training feature mass raised
    to prior_power, normalized to mean alpha. Nonzero training rows receive
    weights proportional to their TF-IDF L1 mass raised to sample_power,
    normalized to mean one before the optional AI class multiplier.
    An optional cap clips those relative weights, followed by renormalization.
    Zero-feature rows contribute no counts and do not affect normalization.
    All these statistics use only X passed to fit, never evaluation data.
    """
    if not np.isfinite([alpha, sample_power, prior_power, ai_weight]).all() or min(alpha, ai_weight) <= 0:
        raise ValueError("Require positive finite alpha/AI weight and finite powers")
    if weight_cap is not None and (not np.isfinite(weight_cap) or weight_cap <= 0):
        raise ValueError("Weight cap must be finite and positive")
    labels = np.asarray(labels)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Weighted fitting requires human=0 and AI=1 labels")
    column_mass = np.asarray(X.sum(axis=0)).ravel()
    row_mass = np.asarray(X.sum(axis=1)).ravel()
    if np.any(column_mass <= 0):
        raise ValueError("Training features must each have positive mass")
    nonzero = row_mass > 0
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        prior = column_mass ** prior_power
        smoothing = alpha * prior / prior.mean()
        weights = np.zeros(len(row_mass))
        weights[nonzero] = (row_mass[nonzero] / row_mass[nonzero].mean()) ** sample_power
        weights[nonzero] /= weights[nonzero].mean()
        if weight_cap is not None:
            weights[nonzero] = np.minimum(weights[nonzero], weight_cap)
            weights[nonzero] /= weights[nonzero].mean()
        weights[labels == 1] *= ai_weight
    if not np.isfinite(smoothing).all() or np.any(smoothing <= 0) or not np.isfinite(weights).all():
        raise ValueError("Parameters produced invalid smoothing or sample weights")
    model = MultinomialNB(alpha=smoothing, force_alpha=True, fit_prior=False)
    return model.fit(X, labels, sample_weight=weights)


def fit_class_smoothed_nb(X, labels, alpha_human, alpha_ai):
    """Fit MNB with class-specific effective smoothing, keeping features fixed.

    Weighting every class-c row by 1/alpha_c with model alpha=1 gives
    (N_cj + alpha_c) / (sum_j N_cj + n_features * alpha_c).
    Equal priors prevent those weights from also changing class prevalence.
    """
    strengths = np.asarray([alpha_human, alpha_ai], dtype=float)
    if not np.isfinite(strengths).all() or np.any(strengths <= 0):
        raise ValueError("Class smoothing values must be finite and positive")
    labels = np.asarray(labels)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Class smoothing requires human=0 and AI=1 labels")
    weights = 1.0 / strengths[labels.astype(int)]
    model = MultinomialNB(alpha=1.0, force_alpha=True, fit_prior=False)
    return model.fit(X, labels, sample_weight=weights)


def select_weighted_parameters_by_cv(texts, labels, candidates=None, n_splits=5):
    """Select MNB fitting parameters with shared features refitted in each fold."""
    if candidates is None:
        # Refined using earlier training-only searches saved in parameter_search/.
        # Neither the feature extractor nor the held-out split is tuned here.
        candidates = [dict(alpha=a, sample_power=p, prior_power=b, ai_weight=1.0, weight_cap=cap)
                      for a in [0.03, 0.3, 1.0, 3.0, 30.0] for p in [-1.0, -0.75, -0.5]
                      for b in [-20.0, -16.0, -12.0] for cap in [1.0, 1.5, 2.0, None]]
    if not candidates or len(texts) != len(labels):
        raise ValueError("Need candidates and matching text/label rows")
    start = perf_counter()
    scores, log_scores = [[] for _ in candidates], [[] for _ in candidates]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(cv.split(texts, labels), 1):
        X, V, _ = tfidf_data(texts.iloc[tr], texts.iloc[va], texts.iloc[va])
        y, v = labels.iloc[tr], labels.iloc[va]
        for i, config in enumerate(candidates):
            model = fit_weighted_nb(X, y, **config)
            log_p = model.predict_log_proba(V)
            scores[i].append(float(roc_auc_score(v, np.exp(log_p[:, 1]))))
            log_scores[i].append(float(roc_auc_score(v, log_p[:, 1] - log_p[:, 0])))
        print(f"Inner CV fold {fold}/{n_splits}: evaluated {len(candidates)} weighted configurations", flush=True)
    records = [{
        **config, "mean_cv_auroc": float(np.mean(s)),
        "std_cv_auroc": float(np.std(s, ddof=1)),
        "mean_cv_log_odds_auroc_diagnostic": float(np.mean(ls)),
        **{f"fold_{i}_auroc": value for i, value in enumerate(s, 1)},
    } for config, s, ls in zip(candidates, scores, log_scores)]
    best_index = max(range(len(records)), key=lambda i: records[i]["mean_cv_auroc"])
    return {
        "selection_metric": "mean inner-CV AUROC",
        "selected_config": dict(candidates[best_index]),
        "selected_mean_cv_auroc": records[best_index]["mean_cv_auroc"],
        "inner_folds": n_splits, "inner_seed": SEED,
        "training_rows_used_for_search": len(texts),
        "search_seconds": perf_counter() - start,
        "protocol": "Shared TF-IDF refitted per inner training fold; all weights and smoothing computed from fold training only",
        "candidates": records,
    }


def select_class_smoothing_by_cv(texts, labels, alphas=None, n_splits=5):
    """Search pairs of effective smoothing strengths using training-only folds."""
    if alphas is None:
        alphas = [1e-30, 1e-24, 1e-18, 1e-15, 1e-12, 10**-11.5,
                  1e-9, 1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0, 1000.0]
    alphas = sorted(set(float(value) for value in alphas))
    if not alphas or not np.isfinite(alphas).all() or min(alphas) <= 0:
        raise ValueError("Candidates must be finite positive alpha values")
    if len(texts) != len(labels):
        raise ValueError("Text and label row counts must match")
    start = perf_counter()
    scores = {(human, ai): [] for human in alphas for ai in alphas}
    log_odds_scores = {pair: [] for pair in scores}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (training_positions, validation_positions) in enumerate(cv.split(texts, labels), 1):
        inner_train, inner_val = texts.iloc[training_positions], texts.iloc[validation_positions]
        X_train, X_val, _ = tfidf_data(inner_train, inner_val, inner_val)
        y_train, y_val = labels.iloc[training_positions], labels.iloc[validation_positions]
        for pair in scores:
            model = fit_class_smoothed_nb(X_train, y_train, *pair)
            log_p = model.predict_log_proba(X_val)
            scores[pair].append(float(roc_auc_score(y_val, np.exp(log_p[:, 1]))))
            log_odds_scores[pair].append(float(roc_auc_score(y_val, log_p[:, 1] - log_p[:, 0])))
        print(f"Inner CV fold {fold}/{n_splits}: evaluated {len(scores)} class-smoothing pairs", flush=True)
    records = [{
        "alpha_human": pair[0], "alpha_ai": pair[1],
        "mean_cv_auroc": float(np.mean(fold_scores)),
        "std_cv_auroc": float(np.std(fold_scores, ddof=1)),
        "mean_cv_log_odds_auroc_diagnostic": float(np.mean(log_odds_scores[pair])),
        **{f"fold_{i}_auroc": value for i, value in enumerate(fold_scores, 1)},
    } for pair, fold_scores in scores.items()]
    best = max(records, key=lambda r: (r["mean_cv_auroc"], r["alpha_human"], r["alpha_ai"]))
    return {
        "selection_metric": "mean inner-CV AUROC",
        "method": "class-specific smoothing via class-constant sample weights; equal class priors",
        "selected_alpha_human": best["alpha_human"],
        "selected_alpha_ai": best["alpha_ai"],
        "selected_mean_cv_auroc": best["mean_cv_auroc"],
        "inner_folds": n_splits, "inner_seed": SEED,
        "training_rows_used_for_search": len(texts),
        "search_seconds": perf_counter() - start,
        "protocol": "Stratified folds within the 8,000 training rows; shared TF-IDF refit independently in each fold",
        "candidates": records,
    }


def select_alpha_by_cv(texts, labels, alphas=None, n_splits=5):
    """Choose smoothing using only the outer training data, refitting TF-IDF per fold."""
    if alphas is None:
        # The initial 1e-5..1e3 search selected its lower boundary using
        # inner-CV scores, so extend only that boundary. All values remain
        # strictly positive; force_alpha preserves these small candidates.
        alphas = np.unique(np.concatenate((np.logspace(-30, -5, 51), np.logspace(-5, 3, 33))))
    alphas = sorted(set(float(value) for value in alphas))
    if not alphas or not np.isfinite(alphas).all() or min(alphas) <= 0:
        raise ValueError("Candidates must be finite positive alpha values")
    if len(texts) != len(labels):
        raise ValueError("Text and label row counts must match")
    start = perf_counter()
    scores = {alpha: [] for alpha in alphas}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (training_positions, validation_positions) in enumerate(cv.split(texts, labels), 1):
        # The shared API needs three text batches. Reuse the inner validation
        # texts as its third argument and discard that duplicate transform.
        # No outer validation or Kaggle test texts are passed into this search.
        inner_train = texts.iloc[training_positions]
        inner_val = texts.iloc[validation_positions]
        X_train, X_val, _ = tfidf_data(inner_train, inner_val, inner_val)
        y_train, y_val = labels.iloc[training_positions], labels.iloc[validation_positions]
        for alpha in alphas:
            model = MultinomialNB(alpha=alpha, force_alpha=True, fit_prior=True)
            model.fit(X_train, y_train)
            ai_column = int(np.flatnonzero(model.classes_ == 1).item())
            probabilities = model.predict_proba(X_val)[:, ai_column]
            scores[alpha].append(float(roc_auc_score(y_val, probabilities)))
        print(f"Inner CV fold {fold}/{n_splits}: evaluated {len(alphas)} alpha values", flush=True)
    records = []
    for alpha, fold_scores in scores.items():
        records.append({
            "alpha": alpha,
            "mean_cv_auroc": float(np.mean(fold_scores)),
            "std_cv_auroc": float(np.std(fold_scores, ddof=1)),
            **{f"fold_{i}_auroc": score for i, score in enumerate(fold_scores, 1)},
        })
    # Prefer stronger smoothing only in the case of an exact mean-score tie.
    best = max(records, key=lambda record: (record["mean_cv_auroc"], record["alpha"]))
    return {
        "selection_metric": "mean inner-CV AUROC",
        "selected_alpha": best["alpha"],
        "selected_mean_cv_auroc": best["mean_cv_auroc"],
        "inner_folds": n_splits,
        "inner_seed": SEED,
        "training_rows_used_for_search": len(texts),
        "search_seconds": perf_counter() - start,
        "protocol": "Stratified folds within the 8,000 training rows; shared TF-IDF refit independently in each fold",
        "candidates": records,
    }


def run_experiment(alpha=1.0, tuning=None, weighted_config=None, estimator=None):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Use a finite positive alpha for this baseline")
    if estimator is not None and weighted_config is not None:
        raise ValueError("Choose a supplied estimator or single-NB weighting, not both")

    started = datetime.now(timezone.utc)
    start = perf_counter()
    X_train, y_train, X_val, y_val, X_test = preprocess_data(n=8000, random_state=SEED)
    preprocessing_seconds = perf_counter() - start

    # Shared preprocessing preserves the original row indices in the labels.
    # Recover IDs from those indices rather than guessing the shuffled order.
    raw = pd.read_csv(ROOT / "data" / "train.csv")
    train_rows = raw.loc[y_train.index]
    val_rows = raw.loc[y_val.index]
    if (len(train_rows), len(val_rows), len(raw)) != (8000, 2000, 10000):
        raise ValueError("Expected the shared 8,000/2,000 split of 10,000 rows")
    if train_rows.label.value_counts().to_dict() != {0: 4000, 1: 4000}:
        raise ValueError("Training labels are not the expected stratified split")
    if val_rows.label.value_counts().to_dict() != {0: 1000, 1: 1000}:
        raise ValueError("Validation labels are not the expected stratified split")
    if not raw.id.is_unique or set(train_rows.id) & set(val_rows.id):
        raise ValueError("IDs must be unique and disjoint between splits")
    if not np.array_equal(train_rows.label.to_numpy(), y_train.to_numpy()):
        raise ValueError("Training labels do not match recovered rows")
    if not np.array_equal(val_rows.label.to_numpy(), y_val.to_numpy()):
        raise ValueError("Validation labels do not match recovered rows")

    start = perf_counter()
    if estimator is not None:
        model = clone(estimator).fit(X_train, y_train)
    elif weighted_config is None:
        model = MultinomialNB(alpha=alpha, force_alpha=True, fit_prior=True, class_prior=None)
        model.fit(X_train, y_train)
    else:
        model = fit_weighted_nb(X_train, y_train, **weighted_config)
    fit_seconds = perf_counter() - start

    positive_column = int(np.flatnonzero(model.classes_ == 1).item())
    start = perf_counter()
    train_probabilities = model.predict_proba(X_train)[:, positive_column]
    val_probabilities = model.predict_proba(X_val)[:, positive_column]
    train_metrics = calculate_metrics(y_train, train_probabilities)
    val_metrics = calculate_metrics(y_val, val_probabilities)
    evaluation_seconds = perf_counter() - start

    run_dir = OUTPUT_DIR / "experiments" / started.strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    provenance_files = ["models/multinomial_nb/multinomial_nb.py", "preprocess.py", "data/train.csv", "data/test.csv"]
    if estimator is not None:
        provenance_files.append("models/multinomial_nb/multinomial_nb_ensemble.py")
        provenance_files.extend(getattr(model, "extra_source_files", ()))
        if getattr(model, "training_algorithm", "SAMME") != "SAMME":
            provenance_files.append("models/multinomial_nb/multinomial_nb_real.py")
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    description = "MultinomialNB selected by training-only inner CV" if tuning else "One supervised MultinomialNB baseline; no tuning"
    if estimator is not None:
        description = getattr(model, "training_algorithm", "SAMME") + "-trained MultinomialNB ensemble; aggregation=" + model.aggregation
    model_parameters = model.get_params()
    if isinstance(model_parameters["alpha"], np.ndarray):
        np.save(run_dir / "smoothing_alpha.npy", model_parameters["alpha"])
        model_parameters["alpha"] = "Per-feature smoothing vector saved in smoothing_alpha.npy"
    results = {
        "experiment": description,
        "started_at_utc": started.isoformat(),
        "base_git_commit": base_commit,
        "file_sha256": {name: sha256(ROOT / name) for name in provenance_files},
        "model": type(model).__name__,
        "model_parameters": model_parameters,
        "training_weight_parameters": weighted_config,
        "preprocessing": "Unchanged preprocess.preprocess_data; default word TF-IDF",
        "split_seed": SEED,
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "test_shape_transformed_only": list(X_test.shape),
        "train_label_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "validation_label_counts": {str(k): int(v) for k, v in y_val.value_counts().sort_index().items()},
        "classification_threshold": THRESHOLD,
        "confusion_matrix_order": "rows=true, columns=predicted; classes=[human=0, AI=1]",
        "train": train_metrics,
        "validation": val_metrics,
        "timing_seconds": {
            "preprocessing_including_test_transform": preprocessing_seconds,
            "model_fit_only": fit_seconds,
            "train_and_validation_evaluation": evaluation_seconds,
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "notes": [
            "Training metrics use the same data used to fit the model.",
            "Validation data are not used to fit the model or the shared TF-IDF.",
            "Accuracy and F1 use probability >= 0.5; AUROC uses unthresholded probabilities.",
            "Test data are transformed by the shared function but not predicted or scored here.",
            "No classifier artifact is saved because shared preprocessing does not return the fitted vectorizer.",
        ],
    }
    if tuning is not None:
        results["tuning"] = tuning
        filename = "parameter_cv_results.csv" if weighted_config or estimator is not None else "alpha_cv_results.csv"
        pd.DataFrame(tuning["candidates"]).to_csv(run_dir / filename, index=False)
    if weighted_config is not None:
        results["notes"].extend([
            "Shared text cleaning, vocabulary selection, TF-IDF values and matrix shapes are unchanged.",
            "Only MNB smoothing and training sample weights change; equal class priors are fixed.",
            "Zero-feature training rows receive weight zero and are excluded from weight normalization.",
            "The fixed validation set has also been inspected in earlier experiments; it is development validation, not an untouched test set.",
        ])
    if estimator is not None:
        results["ensemble_training_algorithm"] = getattr(model, "training_algorithm", "SAMME")
        if hasattr(model, "get_experiment_details"):
            # Nested score blends need component-level coefficients and fitted
            # scales, rather than misleading flattened base-learner weights.
            details = model.get_experiment_details(run_dir)
            results["notes"].extend(details.pop("notes", []))
            results.update(details)
            results["notes"].extend([
                "Every classifier is MultinomialNB; all components use the same unchanged input features.",
                "The fixed validation set has been inspected across earlier rounds; it is development validation, not an untouched final test set.",
            ])
        else:
            results["ensemble_fitted_estimators"] = len(model.boost_.estimators_)
            results["ensemble_learner_weights"] = model.boost_.estimator_weights_[:len(model.boost_.estimators_)].tolist()
            raw_weights = np.asarray(results["ensemble_learner_weights"])
            positive = raw_weights > 0
            log_weights = np.log(raw_weights[positive]) * model.coefficient_power
            prediction_weights = np.zeros_like(raw_weights)
            prediction_weights[positive] = np.exp(log_weights - log_weights.max())
            results["ensemble_aggregation_weights"] = (prediction_weights / prediction_weights.sum()).tolist()
            results["notes"].extend([
                "Every base learner is MultinomialNB; no input feature values or columns are changed.",
                "Base-learner sample weights are rescaled to mean one on nonzero training rows so alpha retains its scale.",
                ("Probability averaging and log-odds averaging are custom prediction rules over SAMME-trained learners, not standard AdaBoost predict_proba or SAMME.R."
                 if results["ensemble_training_algorithm"] == "SAMME" else
                 "Custom binary real-boost implementation following the historical SAMME.R probability-based weight update. Equal-probability pooling is a separate inference rule; it is not the canonical mean-log-odds pooling rule."),
                "Prediction coefficients can differ from SAMME training coefficients; both are recorded. Temperature, when used, shapes each base probability before averaging and is not fitted probability calibration.",
                "The fixed validation set has been inspected across earlier rounds; it is development validation, not an untouched final test set.",
            ])
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    pd.DataFrame({
        "id": val_rows.id.to_numpy(),
        "true_label": y_val.to_numpy(),
        "probability_ai": val_probabilities,
        "predicted_label": (val_probabilities >= THRESHOLD).astype(int),
    }).to_csv(run_dir / "validation_predictions.csv", index=False)
    pd.concat([
        pd.DataFrame({"id": train_rows.id.to_numpy(), "split": "train"}),
        pd.DataFrame({"id": val_rows.id.to_numpy(), "split": "validation"}),
    ], ignore_index=True).to_csv(run_dir / "split_ids.csv", index=False)

    summary = (
        f"{description}\n"
        f"Parameters: {model_parameters if estimator is not None else weighted_config if weighted_config else {'alpha': alpha, 'fit_prior': True}}; seed={SEED}\n"
        f"Training: {len(y_train)} rows; validation: {len(y_val)} rows; features: {X_train.shape[1]}\n\n"
        "Metric                Train       Validation\n"
        f"AUROC                 {train_metrics['auroc']:.6f}    {val_metrics['auroc']:.6f}\n"
        f"Accuracy (t=0.5)      {train_metrics['accuracy']:.6f}    {val_metrics['accuracy']:.6f}\n"
        f"F1 (AI positive)      {train_metrics['f1']:.6f}    {val_metrics['f1']:.6f}\n\n"
        f"Preprocessing, including test transform: {preprocessing_seconds:.4f} seconds\n"
        f"Model fitting only: {fit_seconds:.4f} seconds\n"
        f"Training + validation evaluation: {evaluation_seconds:.4f} seconds\n"
        f"Validation confusion matrix (rows=true, columns=predicted; 0,1): {val_metrics['confusion_matrix']}\n"
        "No test predictions or Kaggle submissions were made.\n"
    )
    if tuning is not None:
        summary += (
            f"Parameter candidates: {len(tuning['candidates'])}; inner folds: {tuning['inner_folds']}\n"
            f"Selected mean inner-CV AUROC: {tuning['selected_mean_cv_auroc']:.6f}\n"
            f"Search time (separate from final model fitting): {tuning['search_seconds']:.4f} seconds\n"
            "Candidate parameters were proposed using training-only inner CV; the fixed 2,000-row holdout is reused development validation.\n"
        )
    (run_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"Results saved to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=1.0, help="NB smoothing parameter (default: 1.0)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tune", action="store_true", help="choose alpha by 5-fold CV within training data; overrides --alpha")
    modes.add_argument("--tune-weighted", action="store_true", help="choose MNB smoothing and sample weights by training-only CV")
    args = parser.parse_args()
    if args.tune_weighted:
        training_rows, _ = split_validation_set(ROOT / "data" / "train.csv", random_state=SEED)
        tuning = select_weighted_parameters_by_cv(training_rows["text"], training_rows["label"])
        print(f"Selected parameters: {tuning['selected_config']}; mean CV AUROC={tuning['selected_mean_cv_auroc']:.6f}", flush=True)
        run_experiment(tuning=tuning, weighted_config=tuning["selected_config"])
    elif args.tune:
        training_rows, _ = split_validation_set(ROOT / "data" / "train.csv", random_state=SEED)
        tuning = select_alpha_by_cv(training_rows["text"], training_rows["label"])
        print(f"Selected alpha={tuning['selected_alpha']:.8g} using mean CV AUROC={tuning['selected_mean_cv_auroc']:.6f}", flush=True)
        run_experiment(alpha=tuning["selected_alpha"], tuning=tuning)
    else:
        run_experiment(alpha=args.alpha)
