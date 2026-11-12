r"""Linear SVM versus logistic regression on the fixed CS5785 validation split.

Run from this folder:
    .\.venv\Scripts\python.exe svm_experiments.py

This script reads the existing word-only and punctuation logistic-regression
results, verifies them, and preserves their files. Outputs go to results_svm/.
The shared baseline script supplies data loading, splitting, and bootstrapping.

EXPERIMENT DESIGN
* Use the SAME 8,000 labeled training passages and 2,000 validation passages.
  Each partition is balanced; seed=42. Test/unlabeled data are never read.
* Fit 18 linear SVMs: 2 tokenizers x 3 n-gram settings x 3 C values.
  Tokenizers: words only; words plus punctuation/non-word symbols.
  Features: unigrams (1,1), bigrams only (2,2), both (1,2).
  C values: 0.1, 1, 10. Smaller C means stronger regularization.
* Copy each original TF-IDF configuration exactly. Fit vocabulary and IDF on
  the 8,000 training passages only; reuse the matrix across C values.
* LinearSVC uses L2 regularization and squared hinge loss. Its score is w.x+b.
  Positive scores favor AI; negative scores favor human. Scores are NOT
  probabilities. AUROC uses these continuous scores, without thresholding.
* Squared hinge loss penalizes max(0, 1-y*score)^2, with y=-1/+1 internally.
  This differs from logistic regression's log loss. Equal C values do not
  imply equal effective regularization across these different objectives.
* dual=True solves the dual optimization problem, appropriate here because
  all feature spaces have more columns than training rows. This does NOT mean
  a nonlinear kernel. All models are linear in the same TF-IDF feature space.
* Numerical-library threads=1; fixed random seed; tolerance=1e-4. Convergence
  warnings stop the run rather than silently accepting unfinished fits.

FAIR COMPARISON AND UNCERTAINTY
* Compare matched feature/C settings and the best tested model per family.
* Save all training/validation AUROCs, gaps, timings, and decision scores.
* Paired, class-stratified bootstrap samples use identical validation rows for
  every classifier. Intervals condition on trained models and this split. They
  do not account for training variability, model selection, multiple comparisons,
  or the domain/generator shift in the competition test set.
* Timing excludes prediction and calibration. 'fit_seconds' is classifier-only;
  'train_total_seconds' adds TF-IDF training construction as if fit independently.
  Logistic-regression times came from earlier runs, so these are not rigorous
  repeated benchmarks of relative runtime.

PROBABILITIES FOR THE SELECTED SVM
* Select the SVM using raw-score validation AUROC.
* Then use 3-fold stratified out-of-fold sigmoid calibration ONLY inside the
  8,000 training rows. The full TF-IDF + SVM pipeline is inside each fold, so even
  its vocabulary and IDF exclude that fold's calibration examples during fitting.
* ensemble=False fits one sigmoid to out-of-fold scores, then refits the base
  pipeline on all 8,000 training examples. This yields one SVM plus one sigmoid,
  rather than averaging several different SVMs at inference.
* A strictly increasing sigmoid preserves ranking/AUROC; verify that property
  and verify that calibration's refitted base model matches the selected SVM.
  Calibration estimates probabilities; it does not guarantee perfect reliability
  on unfamiliar text. No probability conversion is fitted on validation labels.

OUTPUTS
    metrics.csv                       all 18 SVMs and conditional AUROC intervals
    svm_vs_logistic.csv                18 matched classifier comparisons
    punctuation_effect.csv            punctuation effects within SVMs
    best_by_representation.csv         best C for each feature/tokenizer setting
    best_model_comparisons.csv         selected SVM versus selected LR models
    validation_decision_scores.csv    all 18 SVM raw scores, with IDs and labels
    validation_calibrated_predictions.csv  selected SVM probabilities
    calibration_training_folds.csv    fold IDs for the 8,000 training rows
    split_assignments.csv             all original train/validation assignments
    best_margin_pipeline.joblib       selected TF-IDF + raw-score SVM
    best_calibrated_model.joblib       probability-capable selected model
    best_model_features.csv            20 largest positive/negative coefficients
    comparison.png                    both classifiers across all grid settings
    validation_roc.png                ROC curves for selected classifier variants
    metadata.json                     configuration, provenance, and checks
    experiment_report.txt             full explanation and measured results

The saved models are trained on 8,000 rows. This does not refit on all 10,000
labels, generate test predictions, or submit anything to Kaggle.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import warnings

import logistic_regression_experiments as baseline

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits


SVM_SETTINGS = dict(penalty="l2", loss="squared_hinge", dual=True,
                    tol=1e-4, max_iter=20000, fit_intercept=True,
                    intercept_scaling=1.0, class_weight=None)
TOKENIZER_NAMES = {"words": "Words only", "punctuation": "Words + punctuation"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=baseline.PROJECT_DIR.parent / "cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip")
    parser.add_argument("--word-results", type=Path, default=baseline.PROJECT_DIR / "results")
    parser.add_argument("--punctuation-results", type=Path, default=baseline.PROJECT_DIR / "results_punctuation")
    parser.add_argument("--output-dir", type=Path, default=baseline.PROJECT_DIR / "results_svm")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        parser.error("Use at least 100 bootstrap resamples.")
    for reference in (args.word_results, args.punctuation_results):
        if args.output_dir.resolve().is_relative_to(reference.resolve()):
            parser.error("Outputs must be outside the existing logistic-regression results.")
    return args


def load_references(args):
    refs = {}
    digest = baseline.sha256(args.zip)
    seed = json.loads((args.word_results / "metadata.json").read_text())["seed"]
    train, validation, assignments = baseline.load_and_split(args.zip, seed)
    for tokenizer, directory in [("words", args.word_results), ("punctuation", args.punctuation_results)]:
        metadata = json.loads((directory / "metadata.json").read_text())
        assert metadata["seed"] == seed and metadata["source_zip_sha256"] == digest
        assert metadata["C_values"] == list(baseline.C_VALUES)
        assert metadata["representations"] == {name: list(value) for name, value in baseline.REPRESENTATIONS.items()}
        pd.testing.assert_frame_equal(assignments, pd.read_csv(directory / "split_assignments.csv"))
        probabilities = pd.read_csv(directory / "validation_predictions.csv").set_index("id").loc[validation["id"]]
        assert np.array_equal(probabilities["label"].to_numpy(), validation["label"].to_numpy())
        metrics = pd.read_csv(directory / "metrics.csv")
        assert len(metrics) == 9
        for row in metrics.itertuples():
            assert np.isclose(roc_auc_score(validation["label"], probabilities[row.experiment]), row.validation_auc, atol=1e-12, rtol=0)
        refs[tokenizer] = dict(directory=directory, metadata=metadata, probabilities=probabilities, metrics=metrics,
                               hashes={file.name: baseline.sha256(file) for file in directory.iterdir() if file.is_file()})
    return refs, seed, train, validation, assignments


def paired_difference(draws, first, second):
    low, high = np.quantile(draws[:, first] - draws[:, second], [0.025, 0.975])
    return float(low), float(high)


def plot_results(metrics, comparison, refs, validation, scores, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), sharex=True, sharey=True)
    for row, tokenizer in enumerate(TOKENIZER_NAMES):
        for column, representation in enumerate(baseline.REPRESENTATIONS):
            axis = axes[row, column]
            part = comparison.loc[comparison["tokenizer"].eq(tokenizer) & comparison["representation"].eq(representation)].sort_values("C")
            axis.plot(part["C"], part["logistic_validation_auc"], "o--", color="#777777", label="Logistic regression")
            axis.plot(part["C"], part["svm_validation_auc"], "o-", color="#0072B2", label="Linear SVM")
            axis.fill_between(part["C"], part["svm_auc_ci_low"], part["svm_auc_ci_high"], color="#0072B2", alpha=0.12)
            axis.set_xscale("log")
            axis.set_xticks(baseline.C_VALUES, ["0.1", "1", "10"])
            axis.set_title(f"{TOKENIZER_NAMES[tokenizer]} | {baseline.DISPLAY_NAMES[representation]}", fontsize=11)
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel("Validation AUROC")
            if row == 1:
                axis.set_xlabel("C (larger = weaker regularization)")
    axes[0, 0].set_ylim(min(comparison["logistic_validation_auc"].min(), comparison["svm_auc_ci_low"].min()) - 0.01,
                        min(1, comparison["svm_auc_ci_high"].max() + 0.008))
    fig.suptitle("Linear SVM versus logistic regression", fontsize=16, y=0.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.text(0.5, 0.006, "Same 8,000 / 2,000 split and TF-IDF features. Shading: conditional 95% bootstrap intervals. Equal C need not mean equal effective regularization.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.085, 1, 0.95))
    fig.savefig(output / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.6, 6))
    for tokenizer, color in [("words", "#D55E00"), ("punctuation", "#0072B2")]:
        svm_best = metrics.loc[metrics["tokenizer"].eq(tokenizer)].sort_values("validation_auc", ascending=False).iloc[0]
        lr_best = refs[tokenizer]["metrics"].sort_values("validation_auc", ascending=False).iloc[0]
        for family, values, auc, style in [
            ("SVM", scores[svm_best["experiment"]], svm_best["validation_auc"], "-"),
            ("LR", refs[tokenizer]["probabilities"][lr_best["experiment"]], lr_best["validation_auc"], "--"),
        ]:
            fpr, tpr, _ = roc_curve(validation["label"], values)
            axis.plot(fpr, tpr, style, color=color, label=f"{TOKENIZER_NAMES[tokenizer]}, {family}: {auc:.4f}")
    axis.plot([0, 1], [0, 1], ":", color="#aaaaaa")
    axis.set(xlabel="False positive rate (human predicted AI)", ylabel="True positive rate (AI detected)",
             title="Validation ROC: best tested model in each family", xlim=(0, 1), ylim=(0, 1.01))
    axis.legend(loc="lower right", fontsize=9)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "validation_roc.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    refs, seed, train, validation, assignments = load_references(args)
    assignments.to_csv(output / "split_assignments.csv", index=False)
    y_train, y_val = train["label"].to_numpy(), validation["label"].to_numpy()
    scores = validation[["id", "label"]].reset_index(drop=True)
    records, best_pipeline, best_score, best_experiment = [], None, -np.inf, None
    print("Verified the original balanced split and all 18 saved logistic-regression AUROCs.", flush=True)
    with threadpool_limits(limits=1):
        for tokenizer, reference in refs.items():
            settings = reference["metadata"]["vectorizer_settings"]
            for representation, ngrams in baseline.REPRESENTATIONS.items():
                vectorizer = TfidfVectorizer(ngram_range=ngrams, **settings)
                started = time.perf_counter()
                x_train = vectorizer.fit_transform(train["text"])
                prep_seconds = time.perf_counter() - started
                x_val = vectorizer.transform(validation["text"])
                expected_features = reference["metrics"].loc[reference["metrics"]["representation"].eq(representation), "n_features"].unique()
                assert expected_features.tolist() == [x_train.shape[1]]
                assert x_train.shape[1] > x_train.shape[0], "Review the explicit dual solver choice for this feature space."
                for c in baseline.C_VALUES:
                    old_name = f"{representation}_C{c:g}"
                    experiment = f"{tokenizer}__{old_name}"
                    model = LinearSVC(C=c, random_state=seed, **SVM_SETTINGS)
                    started = time.perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(x_train, y_train)
                    fit_seconds = time.perf_counter() - started
                    assert model.classes_.tolist() == [0, 1]
                    train_scores = model.decision_function(x_train)
                    val_scores = model.decision_function(x_val)
                    assert np.isfinite(val_scores).all()
                    train_auc, val_auc = roc_auc_score(y_train, train_scores), roc_auc_score(y_val, val_scores)
                    records.append(dict(experiment=experiment, tokenizer=tokenizer, representation=representation,
                                        C=c, reference_experiment=old_name, n_features=x_train.shape[1],
                                        train_auc=train_auc, validation_auc=val_auc,
                                        train_validation_gap=train_auc-val_auc, fit_seconds=fit_seconds,
                                        vectorizer_fit_seconds=prep_seconds, train_total_seconds=prep_seconds+fit_seconds,
                                        solver_iterations=int(np.max(model.n_iter_)), converged=True,
                                        empty_validation_vectors=int((np.diff(x_val.indptr) == 0).sum())))
                    scores[experiment] = val_scores
                    if val_auc > best_score:
                        best_pipeline = Pipeline([("tfidf", vectorizer), ("classifier", model)])
                        best_score, best_experiment = val_auc, experiment
                    print(f"{experiment}: train={train_auc:.6f}, validation={val_auc:.6f}, fit={fit_seconds:.3f}s, iterations={model.n_iter_}", flush=True)
                    pd.DataFrame(records).to_csv(output / "metrics.csv", index=False)
                    scores.to_csv(output / "validation_decision_scores.csv", index=False)

    metrics = pd.DataFrame(records)
    assert len(metrics) == 18 and metrics["experiment"].is_unique
    svm_columns = metrics["experiment"].tolist()
    lr_matrix = np.column_stack([refs[row.tokenizer]["probabilities"][row.reference_experiment].to_numpy() for row in metrics.itertuples()])
    matrix = np.column_stack([scores[svm_columns].to_numpy(), lr_matrix])
    print(f"\nComputing {args.bootstrap_repetitions:,} paired bootstrap resamples for 36 classifiers...", flush=True)
    draws = baseline.paired_bootstrap(y_val, matrix, args.bootstrap_repetitions, seed + 1)
    metrics["validation_auc_ci_low"] = np.quantile(draws[:, :18], 0.025, axis=0)
    metrics["validation_auc_ci_high"] = np.quantile(draws[:, :18], 0.975, axis=0)
    comparisons = []
    for i, row in metrics.iterrows():
        old = refs[row["tokenizer"]]["metrics"].set_index("experiment").loc[row["reference_experiment"]]
        low, high = paired_difference(draws, i, 18 + i)
        comparisons.append(dict(tokenizer=row["tokenizer"], representation=row["representation"], C=row["C"],
                                svm_validation_auc=row["validation_auc"], logistic_validation_auc=old["validation_auc"],
                                difference_svm_minus_logistic=row["validation_auc"]-old["validation_auc"],
                                difference_ci_low=low, difference_ci_high=high,
                                svm_auc_ci_low=row["validation_auc_ci_low"], svm_auc_ci_high=row["validation_auc_ci_high"],
                                svm_fit_seconds=row["fit_seconds"], previous_logistic_fit_seconds=old["fit_seconds"]))
    comparison = pd.DataFrame(comparisons)
    metrics.to_csv(output / "metrics.csv", index=False)
    comparison.to_csv(output / "svm_vs_logistic.csv", index=False)

    best_by_representation = metrics.loc[metrics.groupby(["tokenizer", "representation"])["validation_auc"].idxmax()]
    best_by_representation.to_csv(output / "best_by_representation.csv", index=False)
    punctuation_effect = []
    lookup = {(row.tokenizer, row.representation, row.C): i for i, row in metrics.iterrows()}
    for representation in baseline.REPRESENTATIONS:
        for c in baseline.C_VALUES:
            a, b = lookup[("punctuation", representation, c)], lookup[("words", representation, c)]
            low, high = paired_difference(draws, a, b)
            punctuation_effect.append(dict(representation=representation, C=c,
                                           punctuation_auc=metrics.loc[a, "validation_auc"], words_auc=metrics.loc[b, "validation_auc"],
                                           difference=metrics.loc[a, "validation_auc"]-metrics.loc[b, "validation_auc"],
                                           difference_ci_low=low, difference_ci_high=high))
    pd.DataFrame(punctuation_effect).to_csv(output / "punctuation_effect.csv", index=False)

    best_index = int(metrics["validation_auc"].idxmax())
    best_comparisons = []
    for tokenizer in TOKENIZER_NAMES:
        best_lr = refs[tokenizer]["metrics"].sort_values("validation_auc", ascending=False).iloc[0]
        lr_index = metrics.index[metrics["tokenizer"].eq(tokenizer) & metrics["reference_experiment"].eq(best_lr["experiment"])][0]
        low, high = paired_difference(draws, best_index, 18 + lr_index)
        best_comparisons.append(dict(best_svm=best_experiment, logistic_tokenizer=tokenizer,
                                     best_logistic=best_lr["experiment"], svm_auc=best_score,
                                     logistic_auc=best_lr["validation_auc"], difference=best_score-best_lr["validation_auc"],
                                     difference_ci_low=low, difference_ci_high=high))
    pd.DataFrame(best_comparisons).to_csv(output / "best_model_comparisons.csv", index=False)
    plot_results(metrics, comparison, refs, validation, scores, output)

    # Each calibration fold fits a fresh TF-IDF and SVM. Validation texts and
    # labels are never supplied to the calibration fit or internal CV splitter.
    print(f"\nSelected {best_experiment}; fitting probability calibration inside the training partition...", flush=True)
    fold_seed = seed + 2
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_seed)
    fold_ids = np.full(len(train), -1, dtype=int)
    for fold, (internal_train, internal_holdout) in enumerate(cv.split(train["text"], y_train)):
        assert set(internal_train).isdisjoint(internal_holdout)
        fold_ids[internal_holdout] = fold
    assert (fold_ids >= 0).all() and set(train["id"]).isdisjoint(validation["id"])
    pd.DataFrame({"id": train["id"].to_numpy(), "label": y_train, "calibration_fold": fold_ids}).to_csv(output / "calibration_training_folds.csv", index=False)
    calibrated = CalibratedClassifierCV(estimator=clone(best_pipeline), method="sigmoid", cv=cv, ensemble=False, n_jobs=1)
    started = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        calibrated.fit(train["text"].tolist(), y_train)
    calibration_seconds = time.perf_counter() - started
    assert calibrated.classes_.tolist() == [0, 1]
    calibrated_p = calibrated.predict_proba(validation["text"].tolist())[:, 1]
    refitted_scores = calibrated.calibrated_classifiers_[0].estimator.decision_function(validation["text"])
    assert np.allclose(refitted_scores, scores[best_experiment], atol=1e-10, rtol=0)
    assert np.isfinite(calibrated_p).all() and ((0 <= calibrated_p) & (calibrated_p <= 1)).all()
    calibrated_auc = roc_auc_score(y_val, calibrated_p)
    assert np.isclose(calibrated_auc, best_score, atol=1e-12, rtol=0)
    pd.DataFrame({"id": validation["id"].to_numpy(), "label": y_val,
                  "decision_score": scores[best_experiment], "probability_AI": calibrated_p}).to_csv(output / "validation_calibrated_predictions.csv", index=False)
    calibration_info = dict(method="sigmoid", internal_folds=3, internal_fold_seed=fold_seed,
                            ensemble=False, training_rows=8000, fit_seconds=calibration_seconds,
                            raw_validation_auc=best_score, calibrated_validation_auc=calibrated_auc,
                            validation_log_loss=log_loss(y_val, calibrated_p),
                            validation_brier_score=brier_score_loss(y_val, calibrated_p),
                            sigmoid_a=float(calibrated.calibrated_classifiers_[0].calibrators[0].a_),
                            sigmoid_b=float(calibrated.calibrated_classifiers_[0].calibrators[0].b_))

    joblib.dump(best_pipeline, output / "best_margin_pipeline.joblib", compress=3)
    joblib.dump(calibrated, output / "best_calibrated_model.joblib", compress=3)
    restored_margin = joblib.load(output / "best_margin_pipeline.joblib")
    restored_calibrated = joblib.load(output / "best_calibrated_model.joblib")
    assert np.allclose(restored_margin.decision_function(validation["text"]), scores[best_experiment], atol=1e-12, rtol=0)
    assert np.allclose(restored_calibrated.predict_proba(validation["text"])[:, 1], calibrated_p, atol=1e-12, rtol=0)
    for row in metrics.itertuples():
        assert np.isclose(roc_auc_score(y_val, scores[row.experiment]), row.validation_auc, atol=1e-12, rtol=0)
    for reference in refs.values():
        for name, digest in reference["hashes"].items():
            assert baseline.sha256(reference["directory"] / name) == digest

    features = best_pipeline.named_steps["tfidf"].get_feature_names_out()
    weights = best_pipeline.named_steps["classifier"].coef_[0]
    order = np.argsort(weights)
    chosen = np.concatenate([order[:20], order[-20:][::-1]])
    feature_table = pd.DataFrame({"feature": features[chosen], "coefficient": weights[chosen],
                                  "direction": np.where(weights[chosen] > 0, "AI", "human")})
    feature_table.to_csv(output / "best_model_features.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "bootstrap_seed": seed + 1,
        "bootstrap_repetitions": args.bootstrap_repetitions, "source_zip_sha256": baseline.sha256(args.zip),
        "script_sha256": baseline.sha256(Path(__file__)), "shared_script_sha256": baseline.sha256(Path(baseline.__file__)),
        "reference_output_hashes": {key: value["hashes"] for key, value in refs.items()},
        "train_rows": len(train), "validation_rows": len(validation), "class_counts": {"train": {"human": 4000, "AI": 4000}, "validation": {"human": 1000, "AI": 1000}},
        "svm_settings": SVM_SETTINGS, "C_values": baseline.C_VALUES, "representations": baseline.REPRESENTATIONS,
        "vectorizer_settings": {key: value["metadata"]["vectorizer_settings"] for key, value in refs.items()},
        "python_version": platform.python_version(), "numerical_library_threads": 1,
        "versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"]},
        "best_experiment": best_experiment, "best_validation_auc": best_score,
        "best_comparisons": best_comparisons, "calibration": calibration_info,
        "checks": {"same_saved_split": True, "reference_scores_recomputed": True, "matching_feature_counts": True,
                   "all_18_svm_fits_converged": True, "calibration_fits_converged": True,
                   "calibration_uses_training_rows_only": True, "refitted_base_scores_match": True,
                   "calibration_preserves_validation_auc": True, "saved_models_reproduce_predictions": True,
                   "all_validation_auc_values_recomputed": True, "original_outputs_unchanged": True},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    findings = []
    for tokenizer in TOKENIZER_NAMES:
        for representation in baseline.REPRESENTATIONS:
            part = metrics.loc[metrics["tokenizer"].eq(tokenizer) & metrics["representation"].eq(representation)].sort_values("C")
            best = part.loc[part["validation_auc"].idxmax()]
            findings.append(f"{TOKENIZER_NAMES[tokenizer]} / {baseline.DISPLAY_NAMES[representation]}: validation AUROCs at C=0.1,1,10 are " + ", ".join(f"{value:.6f}" for value in part["validation_auc"]) + f". Best tested C={best['C']:g}, training AUROC={best['train_auc']:.6f}, gap={best['train_validation_gap']:.6f}.")
    for item in best_comparisons:
        uncertain = item["difference_ci_low"] <= 0 <= item["difference_ci_high"]
        findings.append(f"Selected SVM minus best {item['logistic_tokenizer']} logistic regression: {item['difference']:+.6f}; paired 95% interval [{item['difference_ci_low']:+.6f}, {item['difference_ci_high']:+.6f}]. " + ("The direction is not clearly resolved by this interval." if uncertain else "This conditional comparison supports the observed direction; it is not a guarantee under distribution shift."))
    report = [
        "CS5785: Linear SVM comparison", "", __doc__, "",
        "ALL 18 EXPERIMENTS", metrics.to_string(index=False, float_format=lambda v: f"{v:.6f}"), "",
        "SVM VS LOGISTIC REGRESSION AT MATCHED SETTINGS", comparison.to_string(index=False, float_format=lambda v: f"{v:.6f}"), "",
        "BEST C BY TOKENIZER AND REPRESENTATION", best_by_representation.to_string(index=False, float_format=lambda v: f"{v:.6f}"), "",
        "INTERPRETATION", *findings, "",
        "PUNCTUATION EFFECT WITHIN SVMS", pd.DataFrame(punctuation_effect).to_string(index=False, float_format=lambda v: f"{v:.6f}"), "",
        f"Selected SVM: {best_experiment}, validation AUROC={best_score:.6f}.",
        "A near-perfect training AUROC is a ranking result, not necessarily perfect classification at score zero. Evaluate generalization using validation rather than training performance.",
        "Compare each classifier's best tested C as well as matched C settings: log loss and squared hinge loss have different scales, so their C values are not interchangeable.",
        "The fixed validation set has now guided multiple rounds of development. This may make the selected score optimistic; bootstrap intervals do not remove that effect.", "",
        "TRAINING-ONLY PROBABILITY CALIBRATION", json.dumps(calibration_info, indent=2),
        "Raw SVM scores are not probabilities and must not be uploaded as the competition's probability column. The saved calibrated model supplies predict_proba; these are validation predictions, not a test submission.", "",
        "FEATURE ASSOCIATIONS", feature_table.to_string(index=False),
        "Weights depend on the TF-IDF scaling and other correlated features. A word or punctuation mark's sign is not a universal rule about authorship.", "",
        "VERIFICATION", json.dumps(metadata["checks"], indent=2), "",
        "ENVIRONMENT", json.dumps(metadata["versions"], indent=2), "",
        "AI ASSISTANCE: Code and analysis prepared with Codex assistance. Review the implementation and evidence before using them in the course report.",
    ]
    (output / "experiment_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + metrics[["tokenizer", "representation", "C", "train_auc", "validation_auc", "fit_seconds", "solver_iterations"]].to_string(index=False), flush=True)
    print("\nBEST COMPARISONS " + json.dumps(best_comparisons), flush=True)
    print("CALIBRATION " + json.dumps(calibration_info), flush=True)
    print(f"\nSelected: {best_experiment}; AUROC={best_score:.6f}. All checks passed.\nOutputs: {output}", flush=True)


if __name__ == "__main__":
    main()
