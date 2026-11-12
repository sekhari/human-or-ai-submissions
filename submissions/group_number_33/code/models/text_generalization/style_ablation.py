"""Two preregistered category ablations of the raw style HGB31 model.

These experiments remove explicit feature categories, without using feature
importance or validation results to select columns. Remaining sentence and
entropy statistics can still depend indirectly on formatting or document
length, so this is not an invariance guarantee.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from models.text_generalization.style_search import FEATURE_NAMES, StyleFeatures


LAYOUT_NAMES = frozenset({
    "paragraph_count",
    "line_count",
    "whitespace_char_fraction",
    "newline_per_char",
    "tab_per_char",
    "carriage_return_per_char",
    "multiple_space_run_per_char",
    "blank_line_fraction",
    "indented_line_fraction",
    "trailing_whitespace_line_fraction",
    "markdown_heading_line_fraction",
    "markdown_list_line_fraction",
    "markdown_quote_line_fraction",
    "code_fence_count",
    "markdown_link_per_token",
})
LAYOUT_PREFIXES = ("paragraph_tokens_", "line_chars_", "whitespace_run_")
HGB31_PARAMS = {
    "max_iter": 250,
    "max_leaf_nodes": 31,
    "learning_rate": 0.06,
    "min_samples_leaf": 30,
    "l2_regularization": 5.0,
}


class NamedStyleSubset(TransformerMixin, BaseEstimator):
    """Select preregistered columns and reject source schema changes."""

    def __init__(self, kept_features, source_features):
        self.kept_features = kept_features
        self.source_features = source_features

    def fit(self, X, y=None):
        if tuple(self.source_features) != FEATURE_NAMES:
            raise ValueError("Style feature schema changed after ablation registration")
        if len(set(self.kept_features)) != len(self.kept_features):
            raise ValueError("Duplicate retained feature names")
        if not self.kept_features:
            raise ValueError("An ablation must retain at least one feature")
        source_positions = {name: index for index, name in enumerate(self.source_features)}
        try:
            self.indices_ = np.asarray([source_positions[name] for name in self.kept_features])
        except KeyError as exc:
            raise ValueError(f"Unknown retained style feature: {exc.args[0]}") from exc
        self.n_features_in_ = len(self.source_features)
        self._validate_matrix(X)
        return self

    def _validate_matrix(self, X):
        if np.ndim(X) != 2 or np.shape(X)[1] != self.n_features_in_:
            raise ValueError("Input columns do not match the registered style schema")

    def transform(self, X):
        check_is_fitted(self, "indices_")
        self._validate_matrix(X)
        return np.asarray(X)[:, self.indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_features, dtype=object)


def candidate_configs() -> list[dict]:
    source_features = list(FEATURE_NAMES)
    layout = {
        name for name in source_features
        if name in LAYOUT_NAMES or name.startswith(LAYOUT_PREFIXES)
    }
    unknown_layout = LAYOUT_NAMES.difference(source_features)
    if unknown_layout:
        raise ValueError(f"Missing preregistered layout fields: {sorted(unknown_layout)}")
    absolute_counts = {name for name in source_features if name.endswith("_count")}
    definitions = [
        ("style_hgb31_no_layout", "no_layout", layout),
        ("style_hgb31_normalized_only", "normalized_only", layout | absolute_counts),
    ]
    return [
        {
            "name": name,
            "ablation": ablation,
            "params": dict(HGB31_PARAMS),
            "source_features": source_features,
            "kept_features": [feature for feature in source_features if feature not in dropped],
            "dropped_features": [feature for feature in source_features if feature in dropped],
        }
        for name, ablation, dropped in definitions
    ]


def build_pipeline(config: dict) -> Pipeline:
    """Build the same HGB31 estimator with an explicitly named feature subset."""
    source_features = config["source_features"]
    if tuple(source_features) != FEATURE_NAMES:
        raise ValueError("Style feature schema does not match the registered configuration")
    kept, dropped = config["kept_features"], config["dropped_features"]
    if set(kept) & set(dropped) or set(kept) | set(dropped) != set(source_features):
        raise ValueError("Kept and dropped columns must partition the style feature schema")
    return Pipeline([
        ("style_features", StyleFeatures()),
        ("feature_subset", NamedStyleSubset(kept, source_features)),
        ("classifier", HistGradientBoostingClassifier(
            **config["params"], early_stopping=False, random_state=12,
        )),
    ])


def main() -> None:
    from models.text_generalization.common import OUT, run_candidates

    configs = candidate_configs()
    destination = OUT / "style_ablation"
    destination.mkdir(parents=True, exist_ok=True)
    source_file = Path(__file__).with_name("style_search.py")
    preregistration = {
        "selection_policy": "Fixed feature categories; no feature-importance-driven selection",
        "source_style_module_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "source_feature_count": len(FEATURE_NAMES),
        "limitations": [
            "Sentence units split on newlines, so retained sentence statistics can encode formatting.",
            "Entropy and normalized rates can still correlate with document length or formatting.",
            "The names denote feature-category ablations, not invariance to arbitrary reformatting.",
        ],
        "configs": configs,
    }
    (destination / "preregistered_feature_lists.json").write_text(
        json.dumps(preregistration, indent=2)
    )
    print(json.dumps({
        config["name"]: {
            "kept": len(config["kept_features"]),
            "dropped": len(config["dropped_features"]),
        }
        for config in configs
    }), flush=True)
    run_candidates("style_ablation", configs, build_pipeline)


if __name__ == "__main__":
    main()
