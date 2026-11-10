"""Compare two additional Q1 models with Team 35's existing character baseline.

The 2,000 validation IDs match train.py when both scripts use the same CSV row order
and seed. This script never uses Kaggle test labels.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train import make_pipeline


SEED = 42
FEATURE_NAMES = [
    "log_chars", "log_words", "mean_word_length", "word_length_std",
    "sentence_count", "mean_sentence_words", "sentence_word_std",
    "type_token_ratio", "hapax_ratio", "long_word_ratio", "short_word_ratio",
    "uppercase_ratio", "digit_ratio", "punctuation_ratio", "comma_per_word",
    "period_per_word", "semicolon_per_word", "colon_per_word",
    "question_per_word", "exclamation_per_word", "quote_per_word",
    "parenthesis_per_word", "hyphen_per_word", "newline_per_word",
    "paragraph_count", "first_person_ratio", "second_person_ratio",
    "third_person_ratio", "contraction_ratio", "stopword_ratio",
]

WORD_RE = re.compile(r"\b[\w]+(?:['’][\w]+)?\b", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+")
STOPWORDS = frozenset(
    "a an and are as at be but by for from had has have he her hers him his i "
    "in is it its me my of on or our ours she that the their them they this "
    "to was we were what when where which who will with you your yours".split()
)
FIRST_PERSON = frozenset("i me my mine we us our ours".split())
SECOND_PERSON = frozenset("you your yours".split())
THIRD_PERSON = frozenset("he him his she her hers it its they them their theirs".split())


def style_features(texts: list[str]) -> np.ndarray:
    """Extract transparent, hand-designed stylometric features from raw passages."""
    rows = []
    for text in texts:
        words = WORD_RE.findall(text)
        lower = [word.lower() for word in words]
        n = len(words)
        denom = max(n, 1)
        lengths = np.fromiter((len(word) for word in words), dtype=float)
        sentences = [part for part in SENTENCE_RE.split(text) if part.strip()]
        sentence_lengths = np.array([len(WORD_RE.findall(part)) for part in sentences], dtype=float)
        counts = {}
        for word in lower:
            counts[word] = counts.get(word, 0) + 1
        char_denom = max(len(text), 1)
        punctuation = sum(char in ".,;:!?\"'()[]{}-—" for char in text)
        rows.append([
            np.log1p(len(text)), np.log1p(n),
            float(lengths.mean()) if n else 0.0,
            float(lengths.std()) if n else 0.0,
            len(sentences),
            float(sentence_lengths.mean()) if len(sentence_lengths) else 0.0,
            float(sentence_lengths.std()) if len(sentence_lengths) else 0.0,
            len(counts) / denom,
            sum(count == 1 for count in counts.values()) / denom,
            sum(length >= 7 for length in lengths) / denom,
            sum(length <= 3 for length in lengths) / denom,
            sum(char.isupper() for char in text) / char_denom,
            sum(char.isdigit() for char in text) / char_denom,
            punctuation / char_denom,
            text.count(",") / denom,
            text.count(".") / denom,
            text.count(";") / denom,
            text.count(":") / denom,
            text.count("?") / denom,
            text.count("!") / denom,
            sum(text.count(char) for char in "\"'“”‘’") / denom,
            sum(text.count(char) for char in "()[]{}") / denom,
            sum(text.count(char) for char in "-—") / denom,
            text.count("\n") / denom,
            len([part for part in re.split(r"\n\s*\n", text) if part.strip()]),
            sum(word in FIRST_PERSON for word in lower) / denom,
            sum(word in SECOND_PERSON for word in lower) / denom,
            sum(word in THIRD_PERSON for word in lower) / denom,
            sum("'" in word or "’" in word for word in words) / denom,
            sum(word in STOPWORDS for word in lower) / denom,
        ])
    return np.asarray(rows, dtype=np.float64)


class ScratchStyleLogistic:
    """L2 logistic regression optimized with full-batch Adam, implemented here."""

    def __init__(self, l2: float = 0.01, steps: int = 700, lr: float = 0.03):
        self.l2, self.steps, self.lr = l2, steps, lr
        self.scaler = StandardScaler()
        self.weights: np.ndarray | None = None
        self.bias = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> ScratchStyleLogistic:
        x = self.scaler.fit_transform(x)
        y = np.asarray(y, dtype=np.float64)
        self.weights = np.zeros(x.shape[1], dtype=np.float64)
        self.bias = 0.0
        mw = np.zeros_like(self.weights)
        vw = np.zeros_like(self.weights)
        mb = vb = 0.0
        for step in range(1, self.steps + 1):
            logits = np.clip(x @ self.weights + self.bias, -35, 35)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            error = probabilities - y
            gw = (x.T @ error) / len(y) + self.l2 * self.weights
            gb = float(error.mean())
            mw = 0.9 * mw + 0.1 * gw
            vw = 0.999 * vw + 0.001 * gw**2
            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * gb**2
            self.weights -= self.lr * (mw / (1 - 0.9**step)) / (
                np.sqrt(vw / (1 - 0.999**step)) + 1e-8
            )
            self.bias -= self.lr * (mb / (1 - 0.9**step)) / (
                np.sqrt(vb / (1 - 0.999**step)) + 1e-8
            )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit before predict_proba")
        logits = np.clip(self.scaler.transform(x) @ self.weights + self.bias, -35, 35)
        return 1.0 / (1.0 + np.exp(-logits))


def choose_column(frame: pd.DataFrame, requested: str | None, candidates: list[str], role: str) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"{role} column {requested!r} not found. Available: {list(frame.columns)}")
        return requested
    for name in candidates:
        matches = [column for column in frame.columns if column.lower() == name]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"Could not identify {role} column. Available: {list(frame.columns)}; pass --{role}-column")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    team_dir = Path(__file__).resolve().parent
    parser.add_argument("--train", type=Path, default=team_dir / "data" / "train.csv",
                        help="Supplied labeled train.csv (all 10,000 rows)")
    parser.add_argument("--output-dir", type=Path, default=team_dir / "data" / "q1-comparison")
    parser.add_argument("--text-column")
    parser.add_argument("--label-column")
    parser.add_argument("--id-column")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-rows", type=int, default=2000)
    args = parser.parse_args()

    frame = pd.read_csv(args.train)
    text_col = choose_column(frame, args.text_column, ["text", "passage", "essay", "content"], "text")
    label_col = choose_column(frame, args.label_column, ["label", "generated", "is_ai", "target"], "label")
    id_col = args.id_column
    if id_col and id_col not in frame.columns:
        raise ValueError(f"ID column {id_col!r} not found")
    if id_col is None:
        id_col = next((c for c in frame.columns if c.lower() in {"id", "text_id", "essay_id"}), None)
    frame = frame.copy()
    if frame[text_col].isna().any() or not frame[text_col].map(lambda x: isinstance(x, str)).all():
        raise ValueError("Text column contains missing or non-string values")
    if set(frame[label_col].dropna().unique()) != {0, 1}:
        raise ValueError("Labels must be numeric 0 (human) and 1 (AI)")
    if len(frame) <= args.validation_rows:
        raise ValueError("Need more rows than the fixed validation size; use the full train.csv")
    frame["_row_id"] = frame[id_col] if id_col else np.arange(len(frame))
    train_idx, val_idx = train_test_split(
        np.arange(len(frame)), test_size=args.validation_rows,
        stratify=frame[label_col], random_state=args.seed,
    )
    train = frame.iloc[train_idx].copy()
    val = frame.iloc[val_idx].copy()
    # Audit overlap without changing the existing baseline's 8,000/2,000 split.
    normalize = lambda s: " ".join(s.lower().split())
    val_texts = set(val[text_col].map(normalize))
    overlapping = train[text_col].map(normalize).isin(val_texts)
    normalized_overlap = int(overlapping.sum())
    exact_overlap = int(train[text_col].isin(set(val[text_col])).sum())
    if exact_overlap:
        raise ValueError(f"Found {exact_overlap} exact duplicate training passages in validation; resolve leakage before comparing")
    if normalized_overlap:
        print(f"Warning: {normalized_overlap} normalized text copies cross the split; inspect before reporting")
    if train[label_col].nunique() != 2 or val[label_col].nunique() != 2:
        raise ValueError("Both training and validation partitions need both classes")
    train_text = train[text_col].tolist()
    val_text = val[text_col].tolist()
    y_train = train[label_col].to_numpy(dtype=int)
    y_val = val[label_col].to_numpy(dtype=int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    records = []
    predictions = pd.DataFrame({"id": val["_row_id"].to_numpy(), "label": y_val})

    print("Extracting writing-style features...")
    start = time.perf_counter()
    x_train = style_features(train_text)
    x_val = style_features(val_text)
    style_feature_seconds = time.perf_counter() - start
    style_search_start = time.perf_counter()
    style_candidates = [0.001, 0.01, 0.1]
    style_scores = {}
    for l2 in style_candidates:
        fold_scores = []
        for fit_idx, check_idx in cv.split(x_train, y_train):
            model = ScratchStyleLogistic(l2=l2).fit(x_train[fit_idx], y_train[fit_idx])
            fold_scores.append(roc_auc_score(y_train[check_idx], model.predict_proba(x_train[check_idx])))
        style_scores[str(l2)] = float(np.mean(fold_scores))
    chosen_l2 = max(style_candidates, key=lambda x: style_scores[str(x)])
    style_search_seconds = time.perf_counter() - style_search_start
    fit_start = time.perf_counter()
    own = ScratchStyleLogistic(l2=chosen_l2).fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_start
    own_train = own.predict_proba(x_train)
    own_val = own.predict_proba(x_val)
    records.append({
        "model": "A: own stylometric logistic", "representation": "30 hand-designed style features",
        "train_auc": roc_auc_score(y_train, own_train), "validation_auc": roc_auc_score(y_val, own_val),
        "fit_seconds": fit_seconds, "search_seconds": style_search_seconds,
        "feature_seconds": style_feature_seconds, "chosen_hyperparameters": {"l2": chosen_l2},
        "cv_auc_by_candidate": style_scores,
    })
    predictions["model_a_p_ai"] = own_val

    print("Training word TF-IDF logistic regression...")
    word_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                                   min_df=2, max_df=0.95, sublinear_tf=True, max_features=100_000)),
        ("clf", LogisticRegression(max_iter=1500, solver="liblinear", random_state=args.seed)),
    ])
    search_start = time.perf_counter()
    word_search = GridSearchCV(word_pipeline, {"clf__C": [0.1, 1.0, 10.0]},
                               scoring="roc_auc", cv=cv, n_jobs=1, refit=False)
    word_search.fit(train_text, y_train)
    word_search_seconds = time.perf_counter() - search_start
    word_pipeline.set_params(clf__C=word_search.best_params_["clf__C"])
    fit_start = time.perf_counter()
    word_pipeline.fit(train_text, y_train)
    word_fit_seconds = time.perf_counter() - fit_start
    word_train = word_pipeline.predict_proba(train_text)[:, 1]
    word_val = word_pipeline.predict_proba(val_text)[:, 1]
    records.append({
        "model": "B: word TF-IDF logistic", "representation": "word 1-2 gram TF-IDF",
        "train_auc": roc_auc_score(y_train, word_train), "validation_auc": roc_auc_score(y_val, word_val),
        "fit_seconds": word_fit_seconds, "search_seconds": word_search_seconds,
        "feature_seconds": None, "chosen_hyperparameters": word_search.best_params_,
        "cv_auc_by_candidate": {str(row["params"]["clf__C"]): row["mean_test_score"]
                                for row in pd.DataFrame(word_search.cv_results_).to_dict("records")},
    })
    predictions["model_b_p_ai"] = word_val

    print("Reproducing Team 35's selected character baseline...")
    char_pipeline = make_pipeline(10.0, args.seed)
    fit_start = time.perf_counter()
    char_pipeline.fit(train_text, y_train)
    char_fit_seconds = time.perf_counter() - fit_start
    char_train = char_pipeline.predict_proba(train_text)[:, 1]
    char_val = char_pipeline.predict_proba(val_text)[:, 1]
    records.append({
        "model": "C: existing character baseline", "representation": "character 3-5 gram TF-IDF",
        "train_auc": roc_auc_score(y_train, char_train), "validation_auc": roc_auc_score(y_val, char_val),
        "fit_seconds": char_fit_seconds, "search_seconds": 0.0,
        "feature_seconds": None, "chosen_hyperparameters": {"C": 10.0},
        "cv_auc_by_candidate": None,
    })
    predictions["model_c_p_ai"] = char_val

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": args.seed, "training_csv": str(args.train.resolve()),
        "text_column": text_col, "label_column": label_col, "id_column": id_col,
        "training_rows": len(train), "validation_rows": len(val),
        "exact_text_overlap": exact_overlap,
        "normalized_text_overlap": normalized_overlap,
        "validation_class_counts": val[label_col].value_counts().sort_index().to_dict(),
        "style_feature_names": FEATURE_NAMES, "models": records,
        "note": "A and B hyperparameters were selected by three-fold CV on training rows only. C reproduces the existing baseline's validation-selected C=10. All use identical train/validation IDs.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    table = pd.DataFrame(records)[["model", "representation", "train_auc", "validation_auc", "fit_seconds"]]
    table.to_csv(args.output_dir / "model_comparison.csv", index=False)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved Q1 results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
