r"""Test punctuation while preserving the previous CS5785 experiment protocol.

Run in this folder:
    .\.venv\Scripts\python.exe punctuation_experiments.py

The original script and results/ are inputs and remain unchanged. This script
imports their shared settings/helpers and writes results_punctuation/.

PRIMARY EXPERIMENT: repeat all nine (representation, C) settings with the token
pattern r'(?u)\b\w+\b|[^\w\s]'. Words stay identical to the original baseline,
but each non-word, non-whitespace character is also retained as a token. This
includes Unicode punctuation (em dash, en dash, curly quotes), ASCII punctuation,
and non-word symbols. Hyphens, en dashes, and em dashes remain distinct. Spaces
and newlines are not tokens. Repeated marks are individual tokens: '!!' becomes
['!', '!'], so repeated punctuation can appear as a bigram.

Example: 'Hello, world -- really?' becomes
['hello', ',', 'world', '-', '-', 'really', '?'].
The terms 'unigrams' and 'bigrams' now mean single tokens and adjacent token
pairs, including punctuation. Punctuation changes adjacency: 'hello, world'
has 'hello ,' and ', world', rather than the word-only bigram 'hello world'.
Thus this tests a whole tokenization change, not simply appending new columns.

All other settings and the exact 8,000 / 2,000 split match the saved baseline.
Vocabulary and IDF are fitted on training text only; test/unlabeled files are
never read. Baseline scores are checked against their saved predictions.

EM-DASH CHECKS:
* Count literal U+2014 em dashes by class, separately in train and validation.
* Report raw em-dash count AUROC with the prespecified direction more=AI.
* Retrain the combined-token model at C=10 (the original baseline's best
  setting), removing every feature containing an em dash. All remaining terms
  and their IDF values are retained; document vectors are renormalized. The
  token stream stays unchanged, so removing a dash feature does NOT invent a
  word bigram across the dash. This isolates the value of em-dash-containing
  features more cleanly than deleting dashes from the raw text.
* Inspect the strongest punctuation-containing coefficients in the selected
  model. Weights are associations, not universal authorship rules.

Paired stratified bootstrap intervals use the same held-out examples for every
model. Intervals condition on this split and trained models, do not incorporate
retraining variation, and are not adjusted for multiple model comparisons.
The highest validation score is a development result, not a hidden-test score.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import warnings

import logistic_regression_experiments as baseline

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


PUNCTUATION_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
EM_DASH = "\u2014"
ABLATION_NAME = "unigrams_bigrams_C10_without_em_dash_features"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-dir", type=Path, default=baseline.PROJECT_DIR / "results")
    parser.add_argument("--output-dir", type=Path, default=baseline.PROJECT_DIR / "results_punctuation")
    parser.add_argument("--zip", type=Path, default=baseline.PROJECT_DIR.parent / "cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.baseline_dir.resolve() == args.output_dir.resolve():
        parser.error("Choose a different output directory to preserve the original baseline.")
    if args.bootstrap_repetitions < 100:
        parser.error("Use at least 100 bootstrap repetitions.")
    return args


def fit_classifier(vectorizer, x_train, x_validation, y_train, y_validation, c, seed, name, representation, prep_seconds):
    model = LogisticRegression(C=c, random_state=seed, **baseline.MODEL_SETTINGS)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    assert model.classes_.tolist() == [0, 1]
    p_train = model.predict_proba(x_train)[:, 1]
    p_validation = model.predict_proba(x_validation)[:, 1]
    assert np.isfinite(p_validation).all() and ((0 <= p_validation) & (p_validation <= 1)).all()
    train_auc = roc_auc_score(y_train, p_train)
    val_auc = roc_auc_score(y_validation, p_validation)
    row = {
        "experiment": name, "representation": representation, "C": c,
        "n_features": x_train.shape[1], "train_auc": train_auc,
        "validation_auc": val_auc, "train_validation_gap": train_auc - val_auc,
        "vectorizer_fit_seconds": prep_seconds, "fit_seconds": fit_seconds,
        "train_total_seconds": prep_seconds + fit_seconds,
        "solver_iterations": int(model.n_iter_.max()), "converged": True,
        "empty_validation_vectors": int((np.diff(x_validation.indptr) == 0).sum()),
    }
    print(f"{name}: train={train_auc:.6f}, validation={val_auc:.6f}, features={x_train.shape[1]:,}, fit={fit_seconds:.3f}s", flush=True)
    return row, p_validation, Pipeline([("tfidf", vectorizer), ("classifier", model)])


def punctuation_statistics(train, validation):
    marks = {"em_dash": "\u2014", "en_dash": "\u2013", "hyphen_minus": "-", "comma": ",", "period": ".", "colon": ":", "semicolon": ";", "exclamation_mark": "!", "question_mark": "?", "straight_double_quote": '"', "curly_right_apostrophe": "\u2019", "ellipsis_character": "\u2026"}
    rows, dash_aucs = [], {}
    for split_name, frame in [("train", train), ("validation", validation)]:
        word_counts = frame["text"].str.split().str.len()
        for name, mark in marks.items():
            counts = frame["text"].str.count(re.escape(mark))
            if name == "em_dash":
                dash_aucs[split_name] = float(roc_auc_score(frame["label"], counts))
            for label in (0, 1):
                mask = frame["label"].eq(label)
                selected = counts[mask]
                rows.append({"split": split_name, "mark": name, "unicode_codepoint": f"U+{ord(mark):04X}", "label": label, "class": "human" if label == 0 else "AI", "passages": int(mask.sum()), "passages_with_mark": int(selected.gt(0).sum()), "percent_with_mark": float(100 * selected.gt(0).mean()), "total_marks": int(selected.sum()), "mean_marks_per_100_words": float((100 * selected / word_counts[mask]).mean())})
    return pd.DataFrame(rows), dash_aucs


def save_comparison_plot(comparison, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for axis, representation in zip(axes, baseline.REPRESENTATIONS):
        part = comparison.loc[comparison["representation"].eq(representation)].sort_values("C")
        axis.plot(part["C"], part["baseline_validation_auc"], "o--", color="#777777", label="Words only (previous)")
        axis.plot(part["C"], part["punctuation_validation_auc"], "o-", color="#0072B2", label="Words + punctuation")
        axis.fill_between(part["C"], part["punctuation_ci_low"], part["punctuation_ci_high"], color="#0072B2", alpha=0.12)
        axis.set_xscale("log")
        axis.set_xticks(baseline.C_VALUES, ["0.1", "1", "10"])
        axis.set(title=baseline.DISPLAY_NAMES[representation], xlabel="C (larger = weaker regularization)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Validation AUROC")
    axes[0].set_ylim(min(comparison["baseline_validation_auc"].min(), comparison["punctuation_ci_low"].min()) - 0.015, min(1, comparison["punctuation_ci_high"].max() + 0.01))
    fig.suptitle("Does preserving punctuation improve text classification?", fontsize=15, y=0.99)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.025), frameon=False)
    fig.text(0.5, 0.003, "Identical 8,000 / 2,000 split and settings except tokenization. Shading: conditional 95% bootstrap intervals. Vertical scale is zoomed.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(output_dir / "punctuation_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_metadata = json.loads((args.baseline_dir / "metadata.json").read_text())
    seed = previous_metadata["seed"]
    assert baseline.sha256(args.zip) == previous_metadata["source_zip_sha256"]
    assert baseline.VECTORIZER_SETTINGS == previous_metadata["vectorizer_settings"]
    assert baseline.MODEL_SETTINGS == previous_metadata["model_settings"]
    assert list(baseline.C_VALUES) == previous_metadata["C_values"]
    assert {name: list(value) for name, value in baseline.REPRESENTATIONS.items()} == previous_metadata["representations"]
    train, validation, assignments = baseline.load_and_split(args.zip, seed)
    pd.testing.assert_frame_equal(assignments, pd.read_csv(args.baseline_dir / "split_assignments.csv"))
    old_predictions = pd.read_csv(args.baseline_dir / "validation_predictions.csv").set_index("id").loc[validation["id"]]
    assert np.array_equal(old_predictions["label"].to_numpy(), validation["label"].to_numpy())
    old_metrics = pd.read_csv(args.baseline_dir / "metrics.csv")
    for row in old_metrics.itertuples():
        assert np.isclose(roc_auc_score(validation["label"], old_predictions[row.experiment]), row.validation_auc, atol=1e-12, rtol=0)
    original_hashes = {p.name: baseline.sha256(p) for p in args.baseline_dir.iterdir() if p.is_file()}
    assignments.to_csv(args.output_dir / "split_assignments.csv", index=False)
    statistics, em_dash_aucs = punctuation_statistics(train, validation)
    statistics.to_csv(args.output_dir / "punctuation_statistics.csv", index=False)
    print("Same saved split confirmed. Nine punctuation models plus one em-dash ablation.", flush=True)

    settings = dict(baseline.VECTORIZER_SETTINGS, token_pattern=PUNCTUATION_PATTERN)
    tokenizer_check = TfidfVectorizer(**settings).build_tokenizer()
    assert tokenizer_check("a\u2014b\u2013c-d!!") == ["a", EM_DASH, "b", "\u2013", "c", "-", "d", "!", "!"]
    predictions = validation[["id", "label"]].reset_index(drop=True)
    y_train, y_validation = train["label"].to_numpy(), validation["label"].to_numpy()
    rows, best_pipeline, best_score, best_experiment = [], None, -np.inf, None
    ablation_row, ablation_pipeline = None, None
    with threadpool_limits(limits=1):
        for representation, ngram_range in baseline.REPRESENTATIONS.items():
            vectorizer = TfidfVectorizer(ngram_range=ngram_range, **settings)
            started = time.perf_counter()
            x_train = vectorizer.fit_transform(train["text"])
            prep_seconds = time.perf_counter() - started
            x_validation = vectorizer.transform(validation["text"])
            for c in baseline.C_VALUES:
                experiment = f"{representation}_C{c:g}"
                row, probabilities, pipeline = fit_classifier(vectorizer, x_train, x_validation, y_train, y_validation, c, seed, experiment, representation, prep_seconds)
                rows.append(row)
                predictions[experiment] = probabilities
                if row["validation_auc"] > best_score:
                    best_pipeline, best_score, best_experiment = pipeline, row["validation_auc"], experiment
                pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
                predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)

            if representation == "unigrams_bigrams":
                original_terms = vectorizer.get_feature_names_out()
                keep = np.array([EM_DASH not in term for term in original_terms])
                vocabulary = {term: index for index, term in enumerate(original_terms[keep])}
                assert keep.any() and (~keep).any(), "No em-dash terms available for the ablation."
                ablation_vectorizer = TfidfVectorizer(ngram_range=ngram_range, vocabulary=vocabulary, **settings)
                started = time.perf_counter()
                ablation_train = ablation_vectorizer.fit_transform(train["text"])
                ablation_prep = time.perf_counter() - started
                ablation_val = ablation_vectorizer.transform(validation["text"])
                assert np.allclose(ablation_vectorizer.idf_, vectorizer.idf_[keep], atol=1e-12, rtol=0)
                ablation_row, ablation_p, ablation_pipeline = fit_classifier(ablation_vectorizer, ablation_train, ablation_val, y_train, y_validation, 10.0, seed, ABLATION_NAME, "unigrams_bigrams_without_em_dash", ablation_prep)
                ablation_row["removed_em_dash_features"] = int((~keep).sum())
                predictions[ABLATION_NAME] = ablation_p

    metrics = pd.DataFrame(rows)
    experiments = metrics["experiment"].tolist()
    assert len(experiments) == 9
    # Column order: nine previous models, nine punctuation models, one ablation.
    matrix = np.column_stack([old_predictions[experiments].to_numpy(), predictions[experiments].to_numpy(), predictions[ABLATION_NAME].to_numpy()])
    print("Estimating paired uncertainty for punctuation gains and the em-dash ablation...", flush=True)
    draws = baseline.paired_bootstrap(y_validation, matrix, args.bootstrap_repetitions, seed + 1)
    metrics["validation_auc_ci_low"] = np.quantile(draws[:, 9:18], 0.025, axis=0)
    metrics["validation_auc_ci_high"] = np.quantile(draws[:, 9:18], 0.975, axis=0)
    comparison_rows = []
    for index, row in metrics.iterrows():
        old = old_metrics.loc[old_metrics["experiment"].eq(row["experiment"])].iloc[0]
        low, high = np.quantile(draws[:, index + 9] - draws[:, index], [0.025, 0.975])
        comparison_rows.append({"representation": row["representation"], "C": row["C"], "baseline_validation_auc": old["validation_auc"], "punctuation_validation_auc": row["validation_auc"], "difference": row["validation_auc"] - old["validation_auc"], "difference_ci_low": low, "difference_ci_high": high, "punctuation_ci_low": row["validation_auc_ci_low"], "punctuation_ci_high": row["validation_auc_ci_high"]})
    comparison = pd.DataFrame(comparison_rows)
    full_index = experiments.index("unigrams_bigrams_C10")
    ablation_interval = np.quantile(draws[:, full_index + 9] - draws[:, 18], [0.025, 0.975])
    ablation = {"full_model": "unigrams_bigrams_C10", "ablated_model": ABLATION_NAME, "full_validation_auc": float(metrics.iloc[full_index]["validation_auc"]), "ablated_validation_auc": ablation_row["validation_auc"], "full_minus_ablated_auc": float(metrics.iloc[full_index]["validation_auc"] - ablation_row["validation_auc"]), "difference_ci_low": float(ablation_interval[0]), "difference_ci_high": float(ablation_interval[1]), "removed_em_dash_features": ablation_row["removed_em_dash_features"]}
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "baseline_comparison.csv", index=False)
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame([ablation_row]).to_csv(args.output_dir / "em_dash_ablation_metrics.csv", index=False)
    pd.DataFrame([ablation]).to_csv(args.output_dir / "em_dash_ablation_comparison.csv", index=False)

    joblib.dump(best_pipeline, args.output_dir / "best_pipeline.joblib", compress=3)
    joblib.dump(ablation_pipeline, args.output_dir / "em_dash_ablation_pipeline.joblib", compress=3)
    for filename, experiment in [("best_pipeline.joblib", best_experiment), ("em_dash_ablation_pipeline.joblib", ABLATION_NAME)]:
        restored = joblib.load(args.output_dir / filename)
        assert np.allclose(restored.predict_proba(validation["text"])[:, 1], predictions[experiment], atol=1e-12, rtol=0)
    for row in metrics.itertuples():
        assert np.isclose(roc_auc_score(y_validation, predictions[row.experiment]), row.validation_auc, atol=1e-12, rtol=0)

    names = best_pipeline.named_steps["tfidf"].get_feature_names_out()
    weights = best_pipeline.named_steps["classifier"].coef_[0]
    contains_punctuation = np.array([re.search(r"[^\w\s]", name) is not None for name in names])
    features = pd.DataFrame({"feature": names, "coefficient": weights, "contains_punctuation_or_symbol": contains_punctuation})
    important = features.iloc[np.argsort(np.abs(weights))[::-1][:60]]
    punct_indices = np.flatnonzero(contains_punctuation)
    punctuation_features = features.iloc[punct_indices[np.argsort(np.abs(weights[punct_indices]))[::-1][:60]]]
    dash_features = features.loc[features["feature"].str.contains(EM_DASH, regex=False)].sort_values("coefficient", ascending=False)
    important.to_csv(args.output_dir / "best_model_features.csv", index=False, encoding="utf-8-sig")
    punctuation_features.to_csv(args.output_dir / "best_punctuation_features.csv", index=False, encoding="utf-8-sig")
    dash_features.to_csv(args.output_dir / "em_dash_features.csv", index=False, encoding="utf-8-sig")
    save_comparison_plot(comparison, args.output_dir)

    # A change in the best C/representation is allowed, but separately compare
    # the selected new model with the selected original model on paired rows.
    new_best_index = experiments.index(best_experiment)
    old_best_index = experiments.index(previous_metadata["best_experiment"])
    best_interval = np.quantile(draws[:, new_best_index + 9] - draws[:, old_best_index], [0.025, 0.975])
    for name, digest in original_hashes.items():
        assert baseline.sha256(args.baseline_dir / name) == digest, "An original baseline output was changed."
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_zip_sha256": baseline.sha256(args.zip), "script_sha256": baseline.sha256(Path(__file__)),
        "shared_baseline_script_sha256": baseline.sha256(Path(baseline.__file__)),
        "baseline_output_hashes": original_hashes, "seed": seed, "bootstrap_seed": seed + 1,
        "bootstrap_repetitions": args.bootstrap_repetitions, "versions": previous_metadata["versions"],
        "vectorizer_settings": settings, "model_settings": baseline.MODEL_SETTINGS,
        "representations": baseline.REPRESENTATIONS, "C_values": baseline.C_VALUES,
        "best_experiment": best_experiment, "best_validation_auc": best_score,
        "best_old_experiment": previous_metadata["best_experiment"],
        "best_new_minus_best_old": best_score - previous_metadata["best_validation_auc"],
        "best_difference_ci": best_interval.tolist(), "em_dash_count_auc": em_dash_aucs,
        "em_dash_ablation": ablation,
        "checks": {"same_saved_split": True, "baseline_auc_recomputed": True, "distinct_dash_tokens": True, "em_dash_ablation_keeps_other_idf_values": True, "all_fits_converged": True, "saved_models_reproduce_predictions": True, "previous_outputs_unchanged": True},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = [
        "CS5785: Does preserving punctuation help?", "", __doc__, "",
        "ALL NINE PUNCTUATION MODELS", metrics.to_string(index=False, float_format=lambda v: f"{v:.6f}"), "",
        "PAIRED COMPARISON AGAINST ORIGINAL WORD-ONLY MODELS", comparison.to_string(index=False, float_format=lambda v: f"{v:.6f}"), "",
        f"Best punctuation model: {best_experiment}, validation AUROC={best_score:.6f}.",
        f"Best punctuation model minus previous best: {metadata['best_new_minus_best_old']:+.6f}; paired 95% interval [{best_interval[0]:+.6f}, {best_interval[1]:+.6f}]. This comparison is conditional on model selection.", "",
        "EM DASH (LITERAL U+2014) FREQUENCY", statistics.loc[statistics['mark'].eq('em_dash')].to_string(index=False),
        f"Raw em-dash count AUROC, more=AI: train={em_dash_aucs['train']:.6f}, validation={em_dash_aucs['validation']:.6f}.",
        "These rates describe this dataset; they do not establish a general rule about AI writing.", "",
        "EM-DASH FEATURE ABLATION AT COMBINED FEATURES / C=10", json.dumps(ablation, indent=2),
        "If the full-minus-ablated interval includes zero, this analysis does not establish that em-dash features help. Other punctuation can still improve classification.", "",
        "STRONGEST PUNCTUATION/SYMBOL ASSOCIATIONS IN THE SELECTED MODEL", punctuation_features.head(25).to_string(index=False), "",
        "EM-DASH FEATURES IN THE SELECTED MODEL", dash_features.head(30).to_string(index=False), "",
        "INTERPRETATION LIMITS",
        "Tokenization changes both feature availability and which tokens are adjacent; normalization and regularization also interact with the new vocabulary.",
        "Negative coefficients favor human, positive coefficients favor AI; their effects depend on TF-IDF values and other correlated features.",
        "Formatting or source-specific punctuation may predict labels on a random split without generalizing to new domains, generators, or edited AI passages.",
        "This validation set is reused for development. Intervals do not adjust for model selection, multiple comparisons, retraining variability, or domain shift.",
        "The saved model is trained on 8,000 rows only; no test predictions or Kaggle submission were made.", "",
        "VERIFICATION", json.dumps(metadata["checks"], indent=2), "",
        "AI ASSISTANCE: Code and analysis prepared with Codex assistance; review before including in the course report.",
    ]
    (args.output_dir / "experiment_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + comparison[["representation", "C", "baseline_validation_auc", "punctuation_validation_auc", "difference", "difference_ci_low", "difference_ci_high"]].to_string(index=False), flush=True)
    print("\nEM DASH ABLATION " + json.dumps(ablation), flush=True)
    print(f"\nBest: {best_experiment}, AUROC={best_score:.6f}. All checks passed.\nOutputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
