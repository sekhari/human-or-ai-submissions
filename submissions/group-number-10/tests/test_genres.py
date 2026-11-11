"""Genre detection and per-genre rank normalization."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

pytest.importorskip("transformers")
from experiments.genre_normalize import genre_rank_normalize  # noqa: E402
from src.genres import genre  # noqa: E402


def test_genre_rules():
    assert genre("Ingredients:\n- 2 cups of flour\n- 1 teaspoon salt") == "recipe"
    assert genre("In this paper we propose a new method for parsing.") == "abstract"
    assert genre("The rose is red\nThe sky is wide\nI walk alone\nBy the river side") == "poetry"
    assert genre("- one item here\n- two items here\n- three items here\n- four items here") == "other"
    assert genre("A single paragraph of ordinary prose about a town.") == "other"


def test_normalize_keeps_order_within_genre_and_equalizes_scales():
    scores = np.array([0.9, 0.95, 0.99, 0.1, 0.2, 0.3])
    genres = np.array(["poetry"] * 3 + ["other"] * 3)
    norm = genre_rank_normalize(scores, genres)
    assert np.array_equal(np.argsort(norm[:3]), np.argsort(scores[:3]))
    assert np.allclose(norm[:3], norm[3:])


def test_single_genre_leaves_auroc_unchanged():
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 200), rng.random(200)
    assert roc_auc_score(y, genre_rank_normalize(s, np.array(["other"] * 200))) == pytest.approx(roc_auc_score(y, s))


def test_normalize_fixes_a_genre_wide_shift():
    # Two balanced genres with the same within-genre separation; genre B is shifted toward AI.
    y = np.array([0, 1] * 50 + [0, 1] * 50)
    base = np.where(y == 1, 0.6, 0.4) + np.linspace(0, 0.01, 200)
    genres = np.array(["a"] * 100 + ["b"] * 100)
    shifted = base + np.where(genres == "b", 0.3, 0.0)
    assert roc_auc_score(y, genre_rank_normalize(shifted, genres)) > roc_auc_score(y, shifted)
