"""Character 2-5 grams plus a blank-line feature.

The character SVM is trained on its own. A logistic regression then combines
its margin with an indicator for a blank line. Those combination weights are
fit on 3-fold out-of-fold scores from the training split, not on validation.

The current newline character model is scored on the same split as a reference.
Whitespace and typo views match the Q5 stress test: whitespace collapses every
run of whitespace to one space, and typos swap one adjacent pair in 3% of words.

    python q5_char_blank.py --train data/train.csv
    python q5_char_blank.py --train data/train.csv --test data/test.csv --out submission.csv
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from baseline import ROOT, SEED, load_data, load_split

NGRAM_RANGE = (2, 5)


def has_blank(texts):
    return np.fromiter(
        ("\n\n" in text.replace("\r\n", "\n") for text in texts),
        dtype=np.float64,
        count=len(texts),
    )


def make_char(ngram_range):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            lowercase=True,
            min_df=2,
            max_df=1.0,
            max_features=50000,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm="l2",
            dtype=np.float64,
        )),
        ("svm", LinearSVC(
            C=1.0, dual="auto", max_iter=5000, tol=1e-4, random_state=SEED,
        )),
    ])


def mark_newlines(text):
    return text.lower().replace("\r\n", "\n").replace("\n", "¶")


def make_newline_char():
    model = make_char((3, 5))
    model.set_params(tfidf__preprocessor=mark_newlines)
    return model


def margin(model, texts):
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected SVM class order.")
    return model.decision_function(texts)


def perturb_whitespace(texts):
    return texts.map(lambda text: re.sub(r"\s+", " ", text).strip())


def perturb_typos(texts, rate=0.03, seed=SEED):
    rng = random.Random(seed)

    def swap(word):
        if len(word) < 4:
            return word
        i = rng.randrange(1, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]

    def one(text):
        parts = re.split(r"(\s+)", text)
        return "".join(swap(part) if not part.isspace() and rng.random() < rate else part for part in parts)

    return texts.map(one)


class CharBlankStack:
    """Character SVM margin plus a blank-line indicator."""

    def __init__(self, ngram_range=NGRAM_RANGE):
        self.ngram_range = ngram_range

    def fit(self, texts, labels):
        texts = pd.Series(texts).reset_index(drop=True)
        labels = pd.Series(labels).reset_index(drop=True).astype(int)
        oof = np.zeros(len(texts), dtype=np.float64)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        for train_idx, hold_idx in cv.split(np.zeros(len(texts)), labels):
            fold = make_char(self.ngram_range)
            fold.fit(texts.iloc[train_idx], labels.iloc[train_idx])
            oof[hold_idx] = margin(fold, texts.iloc[hold_idx])
        self.lr_ = LogisticRegression(
            C=1.0, solver="liblinear", max_iter=500, random_state=SEED,
        )
        self.lr_.fit(np.column_stack([oof, has_blank(texts)]), labels)
        self.char_ = make_char(self.ngram_range)
        self.char_.fit(texts, labels)
        return self

    def score(self, texts):
        texts = pd.Series(texts).reset_index(drop=True)
        features = np.column_stack([margin(self.char_, texts), has_blank(texts)])
        return self.lr_.predict_proba(features)[:, 1]


def evaluate(name, score_fn, val, views):
    row = {"variant": name}
    for view, texts in views.items():
        row[f"val_auroc_{view}"] = float(roc_auc_score(val.label, score_fn(texts)))
    return row


def run(train_path, output_dir=ROOT):
    train, val = load_split(load_data(train_path))
    views = {
        "clean": val.text,
        "whitespace": perturb_whitespace(val.text),
        "typos": perturb_typos(val.text),
    }
    stack = CharBlankStack().fit(train.text, train.label)
    newline = make_newline_char().fit(train.text, train.label)
    rows = [
        evaluate("char_2_5_plus_blank", stack.score, val, views),
        evaluate("newline_char", lambda texts: margin(newline, texts), val, views),
    ]
    metrics = {
        "model": "Character TF-IDF 2-5 + blank-line logistic stack",
        "reference": "Character TF-IDF 3-5 with newlines kept as paragraph marks",
        "seed": SEED,
        "ngram_range": list(NGRAM_RANGE),
        "max_features": 50000,
        "svm_C": 1.0,
        "stack": "Logistic regression on the SVM margin and a blank-line indicator",
        "stack_fit": "3-fold out-of-fold margins on the 8,000-row training split",
        "logistic_coefficients_margin_blank": [float(v) for v in stack.lr_.coef_.ravel()],
        "logistic_intercept": float(stack.lr_.intercept_[0]),
        "training_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "rows": rows,
    }
    output_dir = Path(output_dir)
    results = output_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "q5_char_blank.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(results / "q5_char_blank.csv", index=False)
    return metrics


def write_submission(train_path, test_path, out_path):
    """Fit on every labeled row, then write P(AI) for each test id."""
    train = load_data(train_path)
    test = pd.read_csv(test_path, dtype={"id": str})
    if list(test.columns) != ["id", "text"] and not {"id", "text"}.issubset(test.columns):
        raise ValueError("Expected id and text columns in test.csv.")
    test = test[["id", "text"]].copy()
    if len(test) != 3000 or not test.id.is_unique or test.text.isna().any() or test.text.str.strip().eq("").any():
        raise ValueError("Expected 3,000 unique test rows with non-empty text.")
    stack = CharBlankStack().fit(train.text, train.label)
    probability = stack.score(test.text)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Predicted probabilities must lie in [0, 1].")
    submission = pd.DataFrame({"id": test.id, "label": probability})
    sample_path = Path(test_path).resolve().parent / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path, dtype={"id": str})
        if sample.id.tolist() != submission.id.tolist():
            raise ValueError("Submission IDs do not match sample_submission.csv.")
    out_path = Path(out_path)
    submission.to_csv(out_path, index=False)
    return submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "submission.csv")
    args = parser.parse_args()
    if args.test is not None:
        submission = write_submission(args.train, args.test, args.out)
        print(f"wrote {args.out} ({len(submission)} rows)")
        print(submission.label.describe().to_string())
        return
    metrics = run(args.train)
    print(pd.DataFrame(metrics["rows"]).to_string(index=False))


if __name__ == "__main__":
    main()
