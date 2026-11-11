"""Checks against the real Kaggle files. Skipped when data/ is not populated."""

import pytest

from src.data import DATA_DIR, SPLIT_PATH, load_labeled, load_split, normalize_for_dedup, read_csv

TRAIN = DATA_DIR / "train.csv"
pytestmark = pytest.mark.skipif(not TRAIN.exists(), reason="data/train.csv not downloaded")


def test_train_shape_and_balance():
    data = load_labeled(TRAIN)
    assert len(data.labels) == 10_000
    assert (data.labels == 0).sum() == 5000 and (data.labels == 1).sum() == 5000


@pytest.mark.skipif(not SPLIT_PATH.exists(), reason="split not created yet (python -m src.data split)")
def test_shared_split():
    s = load_split()  # verifies hash, disjointness, alignment, balance
    assert len(s.train_labels) == 8000 and len(s.val_labels) == 2000
    assert not set(map(str, s.train_ids)) & set(map(str, s.val_ids))
    assert s.val_labels.sum() == 1000


@pytest.mark.skipif(not (DATA_DIR / "train_starter_2000.csv").exists(), reason="starter file missing")
def test_starter_is_subset_of_train():
    data = load_labeled(TRAIN)
    starter = load_labeled(DATA_DIR / "train_starter_2000.csv")
    train_texts = {normalize_for_dedup(t) for t in data.texts}
    assert len(starter.texts) == 2000
    assert all(normalize_for_dedup(t) in train_texts for t in starter.texts)


@pytest.mark.parametrize("name", ["unlabeled.csv", "test.csv"])
def test_unlabeled_files(name):
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"{name} missing")
    assert len(read_csv(path)) == 3000
