import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from baseline import ROOT, SEED, load_data, load_split, make_model, prediction_frame

MODEL_NAMES = {
    "word_lr": "Word TF-IDF + Logistic Regression",
    "word_nb": "Word TF-IDF + Multinomial Naive Bayes",
    "char_svm": "Character TF-IDF + Linear SVM",
}
GRID = {
    "word_lr": ("C", [0.1, 1.0, 10.0]),
    "word_nb": ("alpha", [0.1, 0.5, 1.0]),
    "char_svm": ("C", [0.1, 1.0, 10.0]),
}


def make_q1_model(family, value):
    if family == "word_lr":
        return make_model().set_params(lr__C=float(value))
    if family == "word_nb":
        return Pipeline([
            ("tfidf", make_model().named_steps["tfidf"]),
            ("nb", MultinomialNB(alpha=float(value), fit_prior=True)),
        ])
    if family == "char_svm":
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), lowercase=True,
                min_df=2, max_df=1.0, max_features=50000,
                sublinear_tf=True, use_idf=True, smooth_idf=True,
                norm="l2", dtype=np.float64,
            )),
            ("svm", LinearSVC(
                C=float(value), dual="auto", max_iter=5000,
                tol=1e-4, random_state=SEED,
            )),
        ])
    raise ValueError("Unknown model family: " + str(family))


def get_scores(model, texts):
    if hasattr(model, "predict_proba"):
        ai_col = int(np.flatnonzero(model.classes_ == 1)[0])
        return model.predict_proba(texts)[:, ai_col]
    # AUROC can use SVM margins directly. These are not probabilities.
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected SVM class order.")
    return model.decision_function(texts)


def fit_timed(model, train):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        start = time.perf_counter()
        model.fit(train.text, train.label)
        elapsed = time.perf_counter() - start
    problems = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if problems:
        raise RuntimeError("Fit did not converge: " + problems[0])
    return elapsed


def compare_models(train, val):
    rows = []
    best = {}
    for family, (param, values) in GRID.items():
        for value in values:
            model = make_q1_model(family, value)
            seconds = fit_timed(model, train)
            train_score = get_scores(model, train.text)
            val_score = get_scores(model, val.text)
            if not np.isfinite(train_score).all() or not np.isfinite(val_score).all():
                raise ValueError("Nonfinite model scores.")
            row = {
                "family": family, "model": MODEL_NAMES[family],
                "parameter": param, "value": float(value),
                "training_auroc": float(roc_auc_score(train.label, train_score)),
                "validation_auroc": float(roc_auc_score(val.label, val_score)),
                "training_time_seconds": seconds,
                "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
                "score_type": "decision_function" if family == "char_svm" else "probability",
            }
            rows.append(row)
            # Ascending grids keep the smaller parameter value in an exact tie.
            if family not in best or row["validation_auroc"] > best[family]["row"]["validation_auroc"] + 1e-12:
                best[family] = {
                    "model": model, "row": row,
                    "train_score": train_score, "val_score": val_score,
                }
            print(f"{family}, {param}={value:g}: train={row['training_auroc']:.6f}, "
                  f"val={row['validation_auroc']:.6f}, fit={seconds:.3f}s", flush=True)
    summary = pd.DataFrame([best[f]["row"] for f in GRID])
    winner = max(GRID, key=lambda f: best[f]["row"]["validation_auroc"])
    return pd.DataFrame(rows), summary, best, winner


def make_probability_model(family, value):
    base = make_q1_model(family, value)
    if family != "char_svm":
        return base
    # Cross-validation wraps the full pipeline, so each calibration fold also
    # learns its own vocabulary and IDF from that fold's training rows only.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid", cv=cv, ensemble=False, n_jobs=1,
    )


def selected_probability_model(train, val, best, winner):
    chosen = best[winner]
    if winner != "char_svm":
        return chosen["model"], {
            "method": "native predict_proba", "extra_fit_time_seconds": 0.0,
        }
    model = make_probability_model(winner, chosen["row"]["value"])
    seconds = fit_timed(model, train)
    return model, {
        "method": "3-fold training-only sigmoid calibration of the full pipeline",
        "ensemble": False, "fold_seed": SEED,
        "extra_fit_time_seconds": seconds,
    }


def plain_params(model):
    out = {}
    for k, value in model.get_params(deep=True).items():
        if isinstance(value, np.generic):
            value = value.item()
        try:
            json.dumps(value)
        except TypeError:
            value = repr(value)
        out[k] = value
    return out


def run_q1(data_path=ROOT / "data" / "train.csv", project_root=ROOT):
    project_root = Path(project_root)
    split_dir = project_root / "splits"
    split_names = ["train_ids.csv", "validation_ids.csv", "split_manifest.json"]
    if not all((split_dir / f).exists() for f in split_names):
        raise FileNotFoundError("The original shared split files are required.")
    before = {f: hashlib.sha256((split_dir / f).read_bytes()).hexdigest() for f in split_names}
    df = load_data(data_path)
    train, val = load_split(df, split_dir)
    with threadpool_limits(limits=1):
        tuning, summary, best, winner = compare_models(train, val)
        model, calibration = selected_probability_model(train, val, best, winner)
    train_prob = model.predict_proba(train.text)[:, 1]
    val_prob = model.predict_proba(val.text)[:, 1]
    for p in [train_prob, val_prob]:
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError("Invalid selected-model probabilities.")
    after = {f: hashlib.sha256((split_dir / f).read_bytes()).hexdigest() for f in split_names}
    if before != after:
        raise RuntimeError("The shared split files changed.")

    output = project_root / "results" / "q1"
    output.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(output / "hyperparameter_results.csv", index=False)
    summary.to_csv(output / "model_comparison.csv", index=False)
    for family in GRID:
        chosen = best[family]
        # Only name a column probability when it actually contains probabilities.
        col = "decision_score" if family == "char_svm" else "predicted_ai_probability"
        out = val[["id", "label"]].rename(columns={"label": "true_label"}).copy()
        out[col] = chosen["val_score"]
        out.to_csv(output / (family + "_validation_scores.csv"), index=False)

    prediction_frame(train, train_prob).to_csv(output / "selected_training_predictions.csv", index=False)
    prediction_frame(val, val_prob).to_csv(output / "selected_validation_predictions.csv", index=False)
    selection = {
        "family": winner, "model": MODEL_NAMES[winner],
        "parameter": best[winner]["row"]["parameter"],
        "value": best[winner]["row"]["value"], "seed": SEED,
        "training_rows": 8000, "validation_rows": 2000,
        "training_auroc": float(roc_auc_score(train.label, train_prob)),
        "validation_auroc": float(roc_auc_score(val.label, val_prob)),
        "base_pipeline_fit_seconds": best[winner]["row"]["training_time_seconds"],
        "probability_output": calibration,
        "selection_rule": "Highest held-out validation AUROC among the 9 fixed candidates; ascending grid order breaks exact within-family ties, LR then NB then SVM breaks exact family ties.",
        "hyperparameter_search": GRID,
        "split_file_sha256": before,
        "status": "Q1 selected model; later team improvements may change the final approach.",
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    configs = {f: plain_params(best[f]["model"]) for f in GRID}
    configs["selected_probability_model"] = plain_params(model)
    (output / "model_configs.json").write_text(json.dumps(configs, indent=2) + "\n")
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
        "timing_scope": "Each candidate's full Pipeline.fit, excluding data loading and prediction. Single run per configuration on the same machine, one BLAS thread.",
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    model_dir = project_root / "models" / "q1"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "selected_model.joblib", compress=3)
    print("\nSelected:", MODEL_NAMES[winner], flush=True)
    print(summary.to_string(index=False), flush=True)
    return tuning, summary, selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    args = parser.parse_args()
    run_q1(args.train)


if __name__ == "__main__":
    main()
