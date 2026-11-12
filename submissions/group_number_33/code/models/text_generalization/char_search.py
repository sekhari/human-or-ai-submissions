"""Training-only character TF-IDF experiments and reusable model builders.

This module owns no split policy. Its experiment entry point delegates split
construction and scoring to the shared ``common`` module.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import clean_text


def minimal_text(text: str) -> str:
    """Preserve text content while normalizing casing and whitespace."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def shared_clean_text(text: str) -> str:
    """Apply existing cleaning without changing the shared implementation."""
    return clean_text(str(text)).lower()


def build_pipeline(config: dict) -> Pipeline:
    """Return an unfitted, cloneable pipeline for one serializable config."""
    text_view = config.get("text_view", "minimal")
    if text_view not in {"minimal", "clean"}:
        raise ValueError(f"Unknown text_view: {text_view}")
    vectorizer = TfidfVectorizer(
        analyzer=config.get("analyzer", "char_wb"),
        ngram_range=tuple(config.get("ngram_range", (3, 5))),
        max_features=int(config.get("max_features", 100_000)),
        min_df=int(config.get("min_df", 3)),
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
        lowercase=False,
        preprocessor=minimal_text if text_view == "minimal" else shared_clean_text,
    )
    classifier_name = config.get("classifier", "logistic")
    c_value = float(config.get("C", 2.0))
    if classifier_name == "logistic":
        classifier = LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=1000,
            tol=1e-4,
            random_state=12,
        )
    elif classifier_name == "linear_svc":
        classifier = LinearSVC(
            C=c_value,
            dual=True,
            max_iter=5000,
            tol=1e-4,
            random_state=12,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def candidate_configs() -> list[dict]:
    """Bounded initial grid with paired cleaning and classifier comparisons."""
    settings = [
        ("char_wb_min_35_lr2", "char_wb", "minimal", (3, 5), "logistic", 2),
        ("char_wb_min_35_lr8", "char_wb", "minimal", (3, 5), "logistic", 8),
        ("char_wb_min_35_svc03", "char_wb", "minimal", (3, 5), "linear_svc", 0.3),
        ("char_wb_min_35_svc1", "char_wb", "minimal", (3, 5), "linear_svc", 1),
        ("char_min_25_svc1", "char", "minimal", (2, 5), "linear_svc", 1),
        ("char_min_36_lr8", "char", "minimal", (3, 6), "logistic", 8),
        ("char_wb_clean_35_lr8", "char_wb", "clean", (3, 5), "logistic", 8),
        ("char_wb_clean_35_svc1", "char_wb", "clean", (3, 5), "linear_svc", 1),
    ]
    return [
        {
            "name": name,
            "family": "char_tfidf",
            "analyzer": analyzer,
            "text_view": text_view,
            "ngram_range": list(ngram_range),
            "classifier": classifier,
            "C": c_value,
            "max_features": 100_000,
            "min_df": 3,
        }
        for name, analyzer, text_view, ngram_range, classifier, c_value in settings
    ]


def main() -> None:
    from models.text_generalization.common import run_candidates

    run_candidates("char", candidate_configs(), build_pipeline)


if __name__ == "__main__":
    main()
