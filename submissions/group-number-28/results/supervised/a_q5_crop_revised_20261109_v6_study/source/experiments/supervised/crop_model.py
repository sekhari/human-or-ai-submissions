"""Fold-contained, weight-preserving crop training for the fixed WC120 recipe.

The estimator receives original documents only. Vocabulary and IDF are learned
from those originals before their crops are generated for classifier training.
Consequently a calibrator can split original rows before this estimator creates
any variants. Decision margins are not probabilities; use the prescribed
training-only calibration wrapper for probability artifacts.
"""
from copy import deepcopy
import hashlib
import math
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from .run_q5 import feature_transformer, load_plan


def text_sha256(text):
    """Hash a document for fitting-boundary receipts without storing passages."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _original_documents(documents):
    if isinstance(documents, (str, bytes)):
        raise TypeError("Expected a sequence of original documents, not one string")
    documents = list(documents)
    if not documents or any(not isinstance(text, str) for text in documents):
        raise ValueError("Expected a nonempty sequence of text strings")
    return documents


def _validate_crop(crop_weight, min_words, crop_words):
    if (isinstance(crop_weight, bool) or not isinstance(crop_weight, Real)
            or not math.isfinite(crop_weight) or not 0 <= crop_weight <= 1):
        raise ValueError("crop_weight must be finite and between zero and one")
    for name, value in (("min_words", min_words), ("crop_words", crop_words)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_words < crop_words:
        raise ValueError("min_words must be at least crop_words")


def weighted_training_rows(documents, y, crop_weight=0.1, min_words=150,
                           crop_words=100, original_sample_weight=None):
    """Return texts, labels, weights and parent-original positions.

    Originals retain their input order; eligible prefix variants follow in that
    order. Parent positions identify the original row for every training row.
    At crop_weight=0 no variants are appended. If original weights are supplied,
    each parent's total is conserved at that weight rather than implicitly one.
    No vectorizer or classifier is fitted here.
    """
    documents = _original_documents(documents)
    _validate_crop(crop_weight, min_words, crop_words)
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("Expected one label per original document")
    check_consistent_length(documents, labels)
    if original_sample_weight is None:
        weights = np.ones(len(documents), dtype=np.float64)
    else:
        weights = np.asarray(original_sample_weight, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError("Expected one sample weight per original document")
        check_consistent_length(documents, weights)
        if not np.isfinite(weights).all() or np.any(weights < 0) or not weights.sum() > 0:
            raise ValueError("Original sample weights must be finite, nonnegative and nonzero in total")
        weights = weights.copy()
    original_weights = weights.copy()
    parents = np.arange(len(documents), dtype=np.int64)
    if crop_weight == 0:
        return documents, labels.copy(), weights, parents
    eligible = np.asarray([i for i, text in enumerate(documents)
                           if len(text.split()) >= min_words], dtype=np.int64)
    crops = [" ".join(documents[i].split()[:crop_words]) for i in eligible]
    weights[eligible] *= 1 - crop_weight
    expanded_weights = np.concatenate([weights, original_weights[eligible] * crop_weight])
    expanded_parents = np.concatenate([parents, eligible])
    np.testing.assert_allclose(
        np.bincount(expanded_parents, weights=expanded_weights, minlength=len(documents)),
        original_weights, atol=1e-14, rtol=1e-14,
    )
    return (documents + crops, np.concatenate([labels, labels[eligible]]),
            expanded_weights, expanded_parents)


class CropAugmentedClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable text classifier whose augmentation is internal to ``fit``.

    ``reference_plan`` uses the existing Q5 parameter dictionary; passing None
    loads that reference at fit time. Both parameters and this dictionary remain
    untouched so sklearn cloning and reproducibility preserve the chosen recipe.
    """

    def __init__(self, crop_weight=0.0, reference_plan=None, min_words=150, crop_words=100):
        self.crop_weight = crop_weight
        self.reference_plan = reference_plan
        self.min_words = min_words
        self.crop_words = crop_words

    def fit(self, X, y, sample_weight=None):
        originals = _original_documents(X)
        labels = np.asarray(y)
        check_classification_targets(labels)
        _validate_crop(self.crop_weight, self.min_words, self.crop_words)
        check_consistent_length(originals, labels)
        if len(np.unique(labels)) != 2:
            raise ValueError("Both classes are required for classifier fitting")
        self.effective_reference_plan_ = deepcopy(load_plan() if self.reference_plan is None
                                                else self.reference_plan)
        # Only originals enter feature fitting. Crops are created below.
        self.transformer_ = feature_transformer("wc120", self.effective_reference_plan_)
        original_x = self.transformer_.fit_transform(originals)
        expanded, expanded_y, weights, parents = weighted_training_rows(
            originals, labels, self.crop_weight, self.min_words, self.crop_words, sample_weight,
        )
        if len(expanded) == len(originals):
            matrix = original_x
        else:
            from scipy.sparse import vstack
            crop_x = self.transformer_.transform(expanded[len(originals):])
            matrix = vstack([original_x, crop_x], format="csr")
        self.classifier_ = LinearSVC(
            random_state=self.effective_reference_plan_["seeds"]["model"],
            **self.effective_reference_plan_["classifier"],
        )
        self.classifier_.fit(matrix, expanded_y, sample_weight=weights)
        self.classes_ = self.classifier_.classes_
        self.fitting_text_hashes_ = tuple(text_sha256(text) for text in originals)
        self.crop_text_hashes_ = tuple(text_sha256(text) for text in expanded[len(originals):])
        self.training_row_parent_indices_ = parents
        self.training_sample_weight_ = weights
        total_weights = np.bincount(parents, weights=weights, minlength=len(originals))
        self.training_weight_receipt_ = {
            "n_originals": len(originals), "n_training_rows": len(expanded),
            "n_crops": len(expanded) - len(originals), "crop_weight": float(self.crop_weight),
            "sample_weight_sum": float(weights.sum()),
            "minimum_parent_weight": float(total_weights.min()),
            "maximum_parent_weight": float(total_weights.max()),
            "n_features": int(matrix.shape[1]),
        }
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["transformer_", "classifier_", "classes_"])
        return self.classifier_.decision_function(self.transformer_.transform(_original_documents(X)))

    def predict(self, X):
        check_is_fitted(self, ["transformer_", "classifier_", "classes_"])
        return self.classifier_.predict(self.transformer_.transform(_original_documents(X)))

    def _more_tags(self):
        return {"X_types": ["string"], "requires_y": True}
