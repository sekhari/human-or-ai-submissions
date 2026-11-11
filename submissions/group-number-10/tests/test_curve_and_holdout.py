"""Learning-curve subsets and genre-holdout splits (no model download needed)."""

import numpy as np
import pytest

pytest.importorskip("transformers")
from experiments.distilbert_train import subsample_train  # noqa: E402
from experiments.genre_holdout import holdout_split  # noqa: E402
from src.data import SplitData  # noqa: E402
from src.genres import is_recipe  # noqa: E402


def _split(n=200):
    labels = np.array([0, 1] * (n // 2))
    return SplitData(np.arange(n), [f"t{i}" for i in range(n)], labels,
                     np.arange(n, n + 10), ["v"] * 10, np.array([0, 1] * 5), 42)


def test_subsample_train_balanced_nested_and_val_unchanged():
    s = _split()
    small, big = subsample_train(s, 40), subsample_train(s, 80)
    assert len(small.train_ids) == 40 and small.train_labels.sum() == 20
    assert set(small.train_ids) <= set(big.train_ids)
    assert np.array_equal(small.val_ids, s.val_ids)
    assert np.array_equal(subsample_train(s, 40).train_ids, small.train_ids)


def test_is_recipe():
    assert is_recipe("Ingredients:\n- 2 cups of flour\n- 1 teaspoon salt\nPreheat the oven.")
    assert not is_recipe("The committee met on Tuesday to bake in new rules.")


@pytest.mark.parametrize("condition", ["holdout", "control"])
def test_holdout_split_disjoint(condition):
    is_genre = np.zeros(300, dtype=bool)
    is_genre[:90] = True
    idx = holdout_split(is_genre, condition, train_size=100, n_eval=30)
    sets = [set(idx[k]) for k in ("eval_genre", "eval_other", "train")]
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
    assert is_genre[idx["eval_genre"]].all() and not is_genre[idx["eval_other"]].any()
    assert len(idx["train"]) == 100
    assert (is_genre[idx["train"]].sum() == 0) == (condition == "holdout")


def test_holdout_eval_sets_identical_across_conditions():
    is_genre = np.arange(300) % 3 == 0
    h, c = holdout_split(is_genre, "holdout", 100, n_eval=30), holdout_split(is_genre, "control", 100, n_eval=30)
    assert np.array_equal(h["eval_genre"], c["eval_genre"]) and np.array_equal(h["eval_other"], c["eval_other"])
