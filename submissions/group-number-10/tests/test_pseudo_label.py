"""Pseudo-label selection and extra training data loading (no model download needed)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers")
from experiments.distilbert_train import load_extra_train  # noqa: E402
from experiments.pseudo_label import select_confident  # noqa: E402


def test_select_confident_keeps_only_both_tails():
    probs = np.array([0.0, 0.02, 0.03, 0.5, 0.97, 0.98, 1.0])
    keep, labels = select_confident(probs, threshold=0.02)
    assert keep.tolist() == [True, True, False, False, False, True, True]
    assert labels[keep].tolist() == [0, 0, 1, 1]


def test_load_extra_train(tmp_path):
    path = tmp_path / "large8k.csv"
    pd.DataFrame({"id": [7, 8], "text": ["a poem", "an essay"], "label": [1, 0]}).to_csv(path, index=False)
    texts, labels, tag = load_extra_train(path)
    assert texts == ["a poem", "an essay"]
    assert labels.tolist() == [1, 0] and labels.dtype.kind == "i"
    assert tag == "_pl-large8k"


def test_load_extra_train_none_is_empty():
    texts, labels, tag = load_extra_train(None)
    assert texts == [] and len(labels) == 0 and tag == ""
