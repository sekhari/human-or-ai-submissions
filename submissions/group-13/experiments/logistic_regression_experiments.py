r"""Controlled word-TF-IDF / logistic-regression experiments for CS5785.

Run from the kaggle folder:
    .\.venv\Scripts\python.exe logistic_regression_experiments.py

Or, in another Python 3.12 environment:
    python -m pip install -r requirements.txt
    python logistic_regression_experiments.py

WHAT THIS SCRIPT DOES
1. Reads only train.csv directly from the original competition zip. The starter
   file is already a subset of train.csv and must not be appended. Neither test
   nor unlabeled passages are used in this supervised baseline experiment.
2. Makes a reproducible, stratified 8,000 / 2,000 train/validation split. Each
   partition is balanced. split_assignments.csv records every ID's assignment.
3. Learns three TF-IDF representations ONLY from the training partition:
       unigrams: one-word features, ngram_range=(1, 1)
       bigrams: two-word features ONLY, ngram_range=(2, 2)
       unigrams_bigrams: both feature types, ngram_range=(1, 2)
   All representations lowercase text, retain stop words and single-letter
   words, keep digits, discard punctuation, and exclude terms appearing in
   fewer than two TRAINING documents. No stemming or accent stripping is used.
   This is a word baseline, not a punctuation-preserving neural tokenizer.
   Adjacent tokens can form a bigram across punctuation or sentence boundaries.
   Raw term counts are multiplied by smoothed IDF and L2-normalized per passage.
   Validation text is transformed with the already-fitted training vocabulary
   and IDF. Words/phrases unseen in training have no feature in this baseline.
4. For each representation, fits independent L2 logistic regressions with
   C=0.1, 1, 10. Smaller C means stronger regularization. A linear score is
   mapped through a sigmoid to estimate P(AI). The labels are human=0, AI=1.
   The sparse-data liblinear solver, tolerance, and iteration limit are fixed.
   A convergence warning stops the experiment rather than silently reporting
   an unfinished fit. One numerical-library thread makes timing less variable.
5. Records training/validation AUROC, their difference, vocabulary size, solver
   iterations, and runtime. AUROC uses probabilities, never thresholded labels.
   Vocabulary fitting is reused across C values. 'fit_seconds' times only the
   classifier; 'train_total_seconds' adds training TF-IDF construction time.

ADDITIONAL INTERPRETATION AND CHECKS
* Paired, stratified bootstrap resampling estimates validation AUROC uncertainty
  and selected differences. Every model sees the SAME resampled validation IDs.
  The intervals describe uncertainty conditional on this split and fitted
  models; they do not cover retraining variability or model-selection optimism.
* The model with the highest validation AUROC is saved, still trained on 8,000
  rows. It is NOT a final model retrained on all 10,000 rows or a Kaggle submission.
* Reloading the saved pipeline must reproduce its saved validation probabilities.
* Top coefficients are associations in this fitted model, not causal indicators
  or universal characteristics of AI writing. Correlated features affect them.

OUTPUTS (under results/ by default)
    metrics.csv                  all nine experiments, timing, and AUROC intervals
    split_assignments.csv        reproducible partition by original passage ID
    validation_predictions.csv   labels and all nine models' probabilities
    paired_comparisons.csv       paired AUROC differences and bootstrap intervals
    comparison.png              training AUROC, validation AUROC, and fit times
    validation_roc.png           ROC curves for the best C of each representation
    best_pipeline.joblib         fitted TF-IDF + selected logistic regression
    best_model_features.csv      20 most positive and 20 most negative weights
    metadata.json                seeds, parameters, versions, hashes, checks
    experiment_report.txt        readable settings, results, and limitations

The validation set is used for hyperparameter/model selection, so its best score
is a development estimate, not an unbiased final-test estimate. The hidden test
includes domain/generator shifts this random split cannot fully reproduce.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time
import warnings
import zipfile

# Keep matplotlib's writable font cache inside this project, not a user-global
# settings directory. The noninteractive backend writes figures without windows.
PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


REPRESENTATIONS = {
    "unigrams": (1, 1),
    "bigrams": (2, 2),
    "unigrams_bigrams": (1, 2),
}
DISPLAY_NAMES = {
    "unigrams": "Unigrams",
    "bigrams": "Bigrams only",
    "unigrams_bigrams": "Unigrams + bigrams",
}
C_VALUES = (0.1, 1.0, 10.0)
COLORS = ("#0072B2", "#D55E00", "#009E73")
VECTORIZER_SETTINGS = {
    "lowercase": True,
    "strip_accents": None,
    "analyzer": "word",
    "token_pattern": r"(?u)\b\w+\b",
    "stop_words": None,
    "min_df": 2,
    "max_df": 1.0,
    "max_features": None,
    "binary": False,
    "use_idf": True,
    "smooth_idf": True,
    "sublinear_tf": False,
    "norm": "l2",
}
MODEL_SETTINGS = {
    "solver": "liblinear",
    "penalty": "l2",
    "tol": 1e-4,
    "max_iter": 2000,
    "class_weight": None,
    "fit_intercept": True,
    "intercept_scaling": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=PROJECT_DIR.parent / "cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        parser.error("Use at least 100 bootstrap repetitions (default: 2000).")
    return args


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_and_split(zip_path: Path, seed: int):
    with zipfile.ZipFile(zip_path) as archive:
        # Selecting this exact member prevents accidental use of the starter,
        # test, or unlabeled files even though they are in the same archive.
        with archive.open("train.csv") as source:
            data = pd.read_csv(source)
    assert list(data.columns) == ["id", "text", "label"], "Unexpected training schema."
    assert len(data) == 10000 and data["id"].is_unique
    assert data.notna().all().all(), "Missing input data requires investigation."
    assert data["text"].str.strip().ne("").all()
    assert data["label"].value_counts().to_dict() == {0: 5000, 1: 5000}
    normalized = data["text"].str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    assert not normalized.duplicated().any(), "Group duplicate texts before splitting."
    train_idx, val_idx = train_test_split(
        np.arange(len(data)), test_size=2000, stratify=data["label"], random_state=seed
    )
    train = data.iloc[train_idx].copy()
    validation = data.iloc[val_idx].copy()
    assert set(train["id"]).isdisjoint(validation["id"])
    assert train["label"].value_counts().to_dict() == {0: 4000, 1: 4000}
    assert validation["label"].value_counts().to_dict() == {0: 1000, 1: 1000}
    assignments = data[["id", "label"]].copy()
    assignments["split"] = "train"
    assignments.loc[val_idx, "split"] = "validation"
    return train, validation, assignments


def paired_bootstrap(labels: np.ndarray, predictions: np.ndarray, repetitions: int, seed: int) -> np.ndarray:
    """Return [resample, model] AUROCs, with ties handled by average ranks.

    Rank-sum AUROC is mathematically equivalent to roc_auc_score for this binary
    case. Using a matrix lets all nine models share each validation resample.
    Both classes are resampled independently, retaining 1,000 examples each.
    """
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    n_negative, n_positive = len(negative), len(positive)
    draws = np.empty((repetitions, predictions.shape[1]))
    for repeat in range(repetitions):
        indices = np.concatenate((rng.choice(negative, n_negative), rng.choice(positive, n_positive)))
        scores = predictions[indices]
        ranks = rankdata(scores, method="average", axis=0)
        draws[repeat] = (ranks[n_negative:].sum(axis=0) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
        if repeat == 0:
            for column in range(predictions.shape[1]):
                assert np.isclose(draws[repeat, column], roc_auc_score(labels[indices], scores[:, column]), atol=1e-12)
    return draws


def make_comparisons(metrics: pd.DataFrame, draws: np.ndarray) -> pd.DataFrame:
    """Planned contrasts: feature combinations at each C; C=10 versus C=0.1."""
    pairs = []
    lookup = {(row.representation, row.C): i for i, row in metrics.iterrows()}
    for c in C_VALUES:
        for reference in ("unigrams", "bigrams"):
            pairs.append((lookup[("unigrams_bigrams", c)], lookup[(reference, c)]))
    for name in REPRESENTATIONS:
        pairs.append((lookup[(name, 10.0)], lookup[(name, 0.1)]))
    rows = []
    for a, b in pairs:
        interval = np.quantile(draws[:, a] - draws[:, b], [0.025, 0.975])
        rows.append({
            "model_a": metrics.loc[a, "experiment"],
            "model_b": metrics.loc[b, "experiment"],
            "validation_auc_difference_a_minus_b": metrics.loc[a, "validation_auc"] - metrics.loc[b, "validation_auc"],
            "difference_ci_low": interval[0],
            "difference_ci_high": interval[1],
        })
    return pd.DataFrame(rows)


def save_figures(metrics: pd.DataFrame, validation: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for (name, _), color in zip(REPRESENTATIONS.items(), COLORS):
        subset = metrics.loc[metrics["representation"] == name].sort_values("C")
        axes[0].plot(subset["C"], subset["train_auc"], "o-", color=color, label=DISPLAY_NAMES[name])
        axes[1].plot(subset["C"], subset["validation_auc"], "o-", color=color)
        axes[1].fill_between(subset["C"], subset["validation_auc_ci_low"], subset["validation_auc_ci_high"], color=color, alpha=0.12)
        axes[2].plot(subset["C"], subset["fit_seconds"], "o-", color=color)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(C_VALUES, ["0.1", "1", "10"])
        axis.set_xlabel("C (larger = weaker regularization)")
        axis.grid(alpha=0.2)
    axes[0].set(title="Training AUROC", ylabel="AUROC")
    axes[0].set_ylim(max(0.5, metrics["train_auc"].min() - 0.01), 1.002)
    axes[1].set(title="Validation AUROC", ylabel="AUROC")
    axes[1].set_ylim(max(0.5, metrics["validation_auc_ci_low"].min() - 0.005), min(1, metrics["validation_auc_ci_high"].max() + 0.005))
    axes[2].set(title="Classifier training time", ylabel="Seconds (excludes TF-IDF fitting)")
    fig.suptitle("Logistic regression: words, word pairs, and regularization", fontsize=15, y=0.99)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.text(0.5, 0.008, "Same 8,000 / 2,000 split in every run. Shading: 95% validation bootstrap intervals. AUROC panels use different, zoomed scales.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.115, 1, 0.94))
    fig.savefig(output_dir / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.8, 5.5))
    for (name, _), color in zip(REPRESENTATIONS.items(), COLORS):
        best = metrics.loc[metrics["representation"] == name].sort_values("validation_auc", ascending=False).iloc[0]
        false_positive, true_positive, _ = roc_curve(validation["label"], predictions[best["experiment"]])
        axis.plot(false_positive, true_positive, color=color, label=f"{DISPLAY_NAMES[name]}, C={best['C']:g}: {best['validation_auc']:.4f}")
    axis.plot([0, 1], [0, 1], "--", color="#999999", label="Random ranking")
    axis.set(xlabel="False positive rate (human labeled AI)", ylabel="True positive rate (AI detected)", title="Validation ROC: best C for each representation", xlim=(0, 1), ylim=(0, 1.01))
    axis.legend(loc="lower right", fontsize=9)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_roc.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    train, validation, assignments = load_and_split(args.zip, args.seed)
    assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    predictions = validation[["id", "label"]].reset_index(drop=True)
    y_train = train["label"].to_numpy()
    y_validation = validation["label"].to_numpy()
    results = []
    empty_vector_ids = {}
    best_auc = -np.inf
    best_pipeline = None
    best_experiment = None
    print(f"Fixed split, seed={args.seed}: training=8,000 (4,000/class); validation=2,000 (1,000/class).", flush=True)

    with threadpool_limits(limits=1):
        for representation, ngram_range in REPRESENTATIONS.items():
            vectorizer = TfidfVectorizer(ngram_range=ngram_range, **VECTORIZER_SETTINGS)
            started = time.perf_counter()
            x_train = vectorizer.fit_transform(train["text"])
            vectorizer_fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            x_validation = vectorizer.transform(validation["text"])
            validation_transform_seconds = time.perf_counter() - started
            zero_rows = int((np.diff(x_validation.indptr) == 0).sum())
            empty_vector_ids[representation] = validation.loc[np.diff(x_validation.indptr) == 0, "id"].tolist()
            print(f"\n{representation}: {x_train.shape[1]:,} features; TF-IDF fit={vectorizer_fit_seconds:.2f}s; empty validation vectors={zero_rows}", flush=True)
            for c in C_VALUES:
                experiment = f"{representation}_C{c:g}"
                model = LogisticRegression(C=c, random_state=args.seed, **MODEL_SETTINGS)
                started = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(x_train, y_train)
                fit_seconds = time.perf_counter() - started
                assert list(model.classes_) == [0, 1]
                started = time.perf_counter()
                train_probabilities = model.predict_proba(x_train)[:, 1]
                val_probabilities = model.predict_proba(x_validation)[:, 1]
                assert np.isfinite(val_probabilities).all()
                assert ((val_probabilities >= 0) & (val_probabilities <= 1)).all()
                train_auc = roc_auc_score(y_train, train_probabilities)
                val_auc = roc_auc_score(y_validation, val_probabilities)
                evaluation_seconds = time.perf_counter() - started
                predictions[experiment] = val_probabilities
                results.append({
                    "experiment": experiment, "representation": representation, "C": c,
                    "train_rows": len(train), "validation_rows": len(validation),
                    "n_features": x_train.shape[1], "train_auc": train_auc,
                    "validation_auc": val_auc, "train_validation_gap": train_auc - val_auc,
                    "vectorizer_fit_seconds": vectorizer_fit_seconds,
                    "validation_transform_seconds": validation_transform_seconds,
                    "fit_seconds": fit_seconds,
                    "train_total_seconds": vectorizer_fit_seconds + fit_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "solver_iterations": int(model.n_iter_.max()), "converged": True,
                    "empty_validation_vectors": zero_rows,
                })
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_experiment = experiment
                    best_pipeline = Pipeline([("tfidf", vectorizer), ("classifier", model)])
                print(f"  C={c:4g}  train AUROC={train_auc:.6f}  validation AUROC={val_auc:.6f}  gap={train_auc-val_auc:.6f}  fit={fit_seconds:.3f}s  iterations={model.n_iter_.max()}", flush=True)
                # Preserve completed fits even if a later stage is interrupted.
                pd.DataFrame(results).to_csv(output_dir / "metrics.csv", index=False)
                predictions.to_csv(output_dir / "validation_predictions.csv", index=False)

    metrics = pd.DataFrame(results)
    assert len(metrics) == 9 and metrics["experiment"].is_unique
    print(f"\nEstimating uncertainty with {args.bootstrap_repetitions:,} paired stratified bootstrap resamples...", flush=True)
    score_matrix = predictions[metrics["experiment"].tolist()].to_numpy()
    draws = paired_bootstrap(y_validation, score_matrix, args.bootstrap_repetitions, args.seed + 1)
    metrics["validation_auc_ci_low"] = np.quantile(draws, 0.025, axis=0)
    metrics["validation_auc_ci_high"] = np.quantile(draws, 0.975, axis=0)
    comparisons = make_comparisons(metrics, draws)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)

    # Save the selected training-only pipeline, then check serialization and raw
    # text prediction against the exact probabilities used to rank the models.
    assert best_pipeline is not None
    joblib.dump(best_pipeline, output_dir / "best_pipeline.joblib", compress=3)
    reloaded = joblib.load(output_dir / "best_pipeline.joblib")
    restored_probabilities = reloaded.predict_proba(validation["text"])[:, 1]
    assert np.allclose(restored_probabilities, predictions[best_experiment], atol=1e-12, rtol=0)
    for row in metrics.itertuples():
        assert np.isclose(roc_auc_score(y_validation, predictions[row.experiment]), row.validation_auc, atol=1e-12)

    names = best_pipeline.named_steps["tfidf"].get_feature_names_out()
    coefficients = best_pipeline.named_steps["classifier"].coef_[0]
    indices = np.argsort(coefficients)
    selected = np.concatenate((indices[:20], indices[-20:][::-1]))
    feature_weights = pd.DataFrame({"feature": names[selected], "coefficient": coefficients[selected], "direction": ["human" if coefficients[i] < 0 else "AI" for i in selected]})
    feature_weights.to_csv(output_dir / "best_model_features.csv", index=False)
    save_figures(metrics, validation, predictions, output_dir)

    # The literal HTML break marker is one plausible source-formatting signal.
    # Count it without changing preprocessing or training any additional models.
    markup_counts = []
    for split_name, frame in [("train", train), ("validation", validation)]:
        has_break = frame["text"].str.contains(r"<br\s*/?\s*>", case=False, regex=True)
        for label in (0, 1):
            markup_counts.append({"split": split_name, "label": label, "passages_with_html_break": int((has_break & frame["label"].eq(label)).sum())})

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_zip": str(args.zip.resolve()), "source_zip_sha256": sha256(args.zip),
        "script_sha256": sha256(Path(__file__)), "seed": args.seed,
        "bootstrap_seed": args.seed + 1, "bootstrap_repetitions": args.bootstrap_repetitions,
        "validation_rows": 2000, "train_class_counts": {"human": 4000, "AI": 4000},
        "validation_class_counts": {"human": 1000, "AI": 1000},
        "representations": REPRESENTATIONS, "C_values": C_VALUES,
        "vectorizer_settings": VECTORIZER_SETTINGS, "model_settings": MODEL_SETTINGS,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "logical_cpus": os.cpu_count(), "numerical_library_threads": 1,
        "versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"]},
        "best_experiment": best_experiment, "best_validation_auc": best_auc,
        "best_intercept": float(best_pipeline.named_steps["classifier"].intercept_[0]),
        "empty_validation_vector_ids": empty_vector_ids,
        "html_break_counts": markup_counts,
        "checks": {"balanced_disjoint_split": True, "no_normalized_exact_duplicate_training_texts": True, "all_fits_converged": True, "all_auc_values_recomputed": True, "saved_model_predictions_match": True, "bootstrap_auc_matches_sklearn": True},
        "limitations": ["One random validation split; no source/domain groups supplied.", "Near duplicates were not exhaustively checked.", "Best validation score is selected on this same validation set.", "Bootstrap intervals condition on fixed fitted models and are not adjusted for multiple comparisons.", "Timings are single runs on this computer, not repeated benchmarks."],
        "wall_seconds": time.perf_counter() - run_started,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    best_by_representation = metrics.loc[metrics.groupby("representation")["validation_auc"].idxmax()]
    findings = []
    for name in REPRESENTATIONS:
        subset = metrics.loc[metrics["representation"].eq(name)].sort_values("C")
        low, high = subset.iloc[0], subset.iloc[-1]
        findings.append(f"{DISPLAY_NAMES[name]}: validation AUROC changes from {low['validation_auc']:.6f} at C=0.1 to {high['validation_auc']:.6f} at C=10 (difference {high['validation_auc']-low['validation_auc']:+.6f}). Training AUROC at C=10 is {high['train_auc']:.6f}, with a training-validation gap of {high['train_validation_gap']:.6f}.")
    for reference in ("unigrams", "bigrams"):
        contrast = comparisons.loc[comparisons["model_a"].eq("unigrams_bigrams_C10") & comparisons["model_b"].eq(f"{reference}_C10")].iloc[0]
        includes_zero = contrast["difference_ci_low"] <= 0 <= contrast["difference_ci_high"]
        findings.append(f"At C=10, combined features minus {DISPLAY_NAMES[reference]}: {contrast['validation_auc_difference_a_minus_b']:+.6f} AUROC, paired 95% interval [{contrast['difference_ci_low']:+.6f}, {contrast['difference_ci_high']:+.6f}]. " + ("The interval includes zero, so this analysis does not clearly establish a difference." if includes_zero else "The interval excludes zero; the direction is supported by this conditional resampling comparison."))
    findings.extend([
        "The largest C is the edge of the tested grid. These results cannot establish that C=10 is the overall optimum.",
        "Near-perfect training ranking with lower validation ranking shows a generalization gap; it does not by itself imply that a lower-C model is preferable. Select using held-out evidence.",
        "Perfect training AUROC means perfect ranking, not necessarily perfect classification at probability threshold 0.5.",
        "The top coefficients below include topic, function-word, and formatting associations. For example, literal <br> markup becomes the word feature 'br' under this tokenizer. The counted markup prevalence helps identify possible source-formatting shortcuts; its causal effect has not been tested.",
        "An empty validation vector means none of that passage's retained word features appears in the fitted vocabulary. Its prediction is then based on the intercept alone. Empty-vector IDs are recorded below for inspection.",
    ])
    report = [
        "CS5785: TF-IDF + logistic regression, controlled comparison", "",
        "PROTOCOL", __doc__.split("WHAT THIS SCRIPT DOES\n", 1)[1].split("OUTPUTS", 1)[0],
        f"Split seed: {args.seed}; bootstrap seed: {args.seed + 1}; resamples: {args.bootstrap_repetitions}.",
        "Tokenization keeps single-letter words and digits; punctuation is discarded.",
        "min_df=2; raw term frequency; smoothed IDF; L2-normalized documents.",
        "Model: liblinear, L2 penalty, max_iter=2000, tol=0.0001, no class weighting.", "",
        "ALL RESULTS", metrics[["representation", "C", "n_features", "train_auc", "validation_auc", "train_validation_gap", "fit_seconds", "train_total_seconds", "validation_auc_ci_low", "validation_auc_ci_high"]].to_string(index=False, float_format=lambda value: f"{value:.6f}"), "",
        "BEST C WITHIN EACH REPRESENTATION", best_by_representation[["representation", "C", "validation_auc"]].to_string(index=False), "",
        f"Selected on validation AUROC: {best_experiment}; AUROC={best_auc:.6f}.",
        "The saved model remains trained on the 8,000 training examples only.", "",
        "RESULT INTERPRETATION", *findings, "",
        "PAIRED COMPARISONS (positive difference favors model A)", comparisons.to_string(index=False, float_format=lambda value: f"{value:.6f}"), "",
        "FEATURE ASSOCIATIONS FOR THE SELECTED MODEL", feature_weights.to_string(index=False), "",
        "EMPTY VALIDATION VECTORS", json.dumps(empty_vector_ids, indent=2), "",
        "OBSERVED HTML BREAK MARKERS BY CLASS", pd.DataFrame(markup_counts).to_string(index=False), "",
        "HOW TO READ THESE RESULTS",
        "A higher training AUROC alone does not establish a better model. Compare validation AUROC and the training-validation gap.",
        "Compare representations at the same C, then inspect the best tested C for each. C has the same setting but need not imply identical effective capacity across feature spaces.",
        "For combined features, unigrams and bigrams share one document-normalization step. Thus adding bigrams changes both the vocabulary and the relative feature scaling.",
        "Use paired intervals for differences, rather than judging significance from overlapping individual intervals. An interval including zero leaves the direction uncertain under this resampling analysis.",
        "Even an interval excluding zero is conditional on this split, these models, and this resampling method; it is not evidence of universal superiority.",
        "TF-IDF fitting is reused across C values. Each train_total_seconds includes the representation's fitting cost as if that model were trained independently; do not sum these values as actual wall time.", "",
        "LIMITATIONS", *metadata["limitations"], "",
        "CHECKS", json.dumps(metadata["checks"], indent=2), "",
        "ENVIRONMENT", json.dumps(metadata["versions"], indent=2), "",
        "AI ASSISTANCE: This script and analysis were prepared with Codex assistance. Review the implementation and results before using them in the course report.",
    ]
    (output_dir / "experiment_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + metrics[["representation", "C", "train_auc", "validation_auc", "fit_seconds"]].to_string(index=False), flush=True)
    print(f"\nSelected: {best_experiment}; validation AUROC={best_auc:.6f}. All checks passed.\nOutputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
