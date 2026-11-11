"""Evaluate Q1's selected model for Q4; never tune on validation.

Usage: python q4_analysis.py --train /path/train.csv --out work/q4_results
Pass --val-ids /path/splits/val_ids.csv to preserve the original Q1 holdout.
Without it, the split follows common.py, so original CSV row order matters.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
Q1_DEFAULT = Path(__file__).parent / "human-or-ai-submissions/submissions/group-16"


def split_data(df, val_path):
    required = {"id", "text", "label"}
    if not required.issubset(df):
        raise ValueError(f"Required columns: {sorted(required)}")
    if len(df) != 10_000 or df[list(required)].isna().any().any():
        raise ValueError("Expected 10,000 complete labeled passages.")
    if not df.id.is_unique or set(df.label.unique()) != {0, 1}:
        raise ValueError("IDs must be unique; labels must be 0 (human) and 1 (AI).")
    if df.label.value_counts().to_dict() != {0: 5000, 1: 5000}:
        raise ValueError("Expected 5,000 human and 5,000 AI passages.")
    if not df.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("All passages must contain nonempty text.")
    if val_path:
        ids = pd.read_csv(val_path, dtype={"id": str})["id"]
        if len(ids) != 2000 or not ids.is_unique or not ids.isin(df.id).all():
            raise ValueError("Validation IDs must be 2,000 unique IDs present in train.csv.")
        source = "supplied validation IDs"
    else:
        _, val = train_test_split(df, test_size=2000, stratify=df.label, random_state=SEED)
        ids = val.id
        source = "reconstructed using Q1 common.py procedure; original row order required"
    mask = df.id.isin(set(ids))
    tr, va = df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)
    if len(tr) != 8000 or len(va) != 2000 or va.label.value_counts().to_dict() != {0: 1000, 1: 1000}:
        raise ValueError("Split does not match Q1's balanced 8,000/2,000 protocol.")
    assert set(tr.id).isdisjoint(va.id)
    return tr, va, source


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def length_results(pred):
    # Fixed before inspecting any validation outcomes. Whitespace-separated words.
    pred["length_bin"] = pd.cut(pred.word_count, bins=[-1, 149, 249, np.inf],
                                labels=["<150", "150-249", ">=250"])
    records = []
    for name, part in pred.groupby("length_bin", observed=False):
        records.append({"bin_words": str(name), "count": len(part),
                        "human_count": int((part.label == 0).sum()),
                        "ai_count": int((part.label == 1).sum()),
                        "validation_auroc": auc(part.label, part.p_ai),
                        "false_positives": int((part.error_type == "FP").sum()),
                        "false_negatives": int((part.error_type == "FN").sum())})
    return pd.DataFrame(records)


def feature_evidence(model, selected):
    """Exact additive log-odds contributions, not causal failure explanations."""
    vectorizer = model.named_steps["vec"]
    weights = model.named_steps["clf"].coef_[0]
    names = vectorizer.get_feature_names_out()
    features = vectorizer.transform(selected.text)
    records = []
    for i, row in enumerate(selected.itertuples(index=False)):
        x = features.getrow(i)
        contributions = x.data * weights[x.indices]
        order = np.argsort(contributions)
        item = {"id": row.id, "error_type": row.error_type, "p_ai": row.p_ai,
                "text": row.text, "toward_ai": [], "toward_human": []}
        for key, indexes in [("toward_ai", order[::-1]), ("toward_human", order)]:
            for j in indexes:
                value = float(contributions[j])
                if (key == "toward_ai" and value > 0) or (key == "toward_human" and value < 0):
                    item[key].append({"feature": str(names[x.indices[j]]), "log_odds_contribution": value})
                if len(item[key]) == 8:
                    break
        reconstructed = float(contributions.sum() + model.named_steps["clf"].intercept_[0])
        actual = float(model.decision_function([row.text])[0])
        if not np.isclose(reconstructed, actual, atol=1e-8):
            raise ValueError("Feature contributions failed reconstruction check.")
        records.append(item)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path)
    parser.add_argument("--q1-dir", type=Path, default=Q1_DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.train, dtype={"id": str})
    tr, va, split_source = split_data(df, args.val_ids)

    # Import the user's actual model factory to avoid silently changing preprocessing.
    sys.path.insert(0, str(args.q1_dir.resolve()))
    spec = importlib.util.spec_from_file_location("q1_models", args.q1_dir / "q1_model_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, model, _ = module.build_models()["A_word_tfidf_lr"]
    model.set_params(vec__ngram_range=(1, 2), vec__min_df=5, clf__C=30)
    # Q1 leaves liblinear's random_state unset; seed the global RNG explicitly here.
    np.random.seed(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(tr.text, tr.label)
    p_train = model.predict_proba(tr.text)[:, list(model.classes_).index(1)]
    p_val = model.predict_proba(va.text)[:, list(model.classes_).index(1)]
    pred = va.copy()
    pred["p_ai"] = p_val
    pred["prediction"] = (p_val >= 0.5).astype(int)
    pred["word_count"] = pred.text.str.split().str.len()
    pred["error_type"] = np.select(
        [(pred.label == 0) & (pred.prediction == 1), (pred.label == 1) & (pred.prediction == 0)],
        ["FP", "FN"], default="correct")
    bins = length_results(pred)
    chosen = pd.concat([
        pred.loc[pred.error_type == "FP"].sort_values(["p_ai", "id"], ascending=[False, True]).head(5),
        pred.loc[pred.error_type == "FN"].sort_values(["p_ai", "id"], ascending=[True, True]).head(5),
    ])
    train_auc, val_auc = auc(tr.label, p_train), auc(va.label, p_val)
    notes = []
    if not args.val_ids:
        notes.append("Original validation IDs unavailable; same CSV row order is required to reproduce Q1's holdout.")
    if round(val_auc, 3) != 0.966:
        notes.append("Validation AUROC does not round to Q1's reported 0.966; investigate before writing final report.")
    if len(chosen) < 10:
        notes.append("Fewer than five errors of at least one type; report honestly, do not manufacture examples.")
    normalized_train_texts = set(tr.text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip())
    normalized_val_texts = va.text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    overlap_count = int(normalized_val_texts.isin(normalized_train_texts).sum())
    if overlap_count:
        notes.append("Some validation passages match training text after lowercasing and whitespace normalization; assess leakage before claiming generalization.")
    summary = {
        "model": "A_word_tfidf_lr", "train_rows": len(tr), "validation_rows": len(va),
        "train_auroc": train_auc, "validation_auroc": val_auc, "auroc_gap": train_auc-val_auc,
        "threshold": 0.5, "confusion_matrix_rows_true_columns_predicted_0_1": confusion_matrix(va.label, pred.prediction, labels=[0, 1]).tolist(),
        "split_source": split_source, "seed": SEED,
        "train_csv_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "q1_model_code_sha256": hashlib.sha256((args.q1_dir / "q1_model_comparison.py").read_bytes()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256("\n".join(sorted(va.id)).encode()).hexdigest(),
        "validation_texts_matching_training_after_case_whitespace_normalization": overlap_count,
        "selected_error_rule": "Five most confidently wrong predictions per type, ties ordered by ID.",
        "word_count_rule": "Python str.split(): whitespace-separated tokens",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "model_settings": {k: str(v) for k, v in model.get_params().items() if k.startswith(("vec__", "clf__"))},
        "notes": notes,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    va[["id"]].to_csv(args.out / "val_ids.csv", index=False)
    pred.drop(columns="text").to_csv(args.out / "validation_predictions.csv", index=False)
    bins.to_csv(args.out / "length_bins.csv", index=False)
    chosen.drop(columns="text").to_csv(args.out / "selected_errors.csv", index=False)
    # Local analysis only: full passages are not copied into the report or committed.
    (args.out / "error_evidence.json").write_text(json.dumps(feature_evidence(model, chosen), indent=2))
    print(json.dumps(summary, indent=2))
    print(bins.to_string(index=False))


if __name__ == "__main__":
    main()
