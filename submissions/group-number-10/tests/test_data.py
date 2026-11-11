"""Split and data-loading tests on synthetic data (no Kaggle files needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data import (
    Columns,
    detect_columns,
    find_near_duplicates,
    load_labeled,
    load_split,
    make_split,
    save_split,
)


@pytest.fixture
def synthetic_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 10_000
    labels = np.array([0] * 5000 + [1] * 5000)
    rng.shuffle(labels)
    ids = rng.permutation(n) + 100  # IDs not in file order
    df = pd.DataFrame({"id": ids, "text": [f"Passage number {i}, NA." for i in ids], "label": labels})
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


def test_load_preserves_text_and_sorts_by_id(synthetic_csv):
    data = load_labeled(synthetic_csv)
    assert len(data.texts) == 10_000
    assert list(data.ids) == sorted(data.ids)
    assert data.texts[0] == f"Passage number {data.ids[0]}, NA."  # "NA" not turned into NaN
    assert data.labels.dtype == np.int64


def test_split_sizes_disjoint_balanced(synthetic_csv):
    data = load_labeled(synthetic_csv)
    split = make_split(data, seed=42, n_val=2000)
    tr, va = set(split["train_index"]), set(split["val_index"])
    assert len(tr) == 8000 and len(va) == 2000
    assert not tr & va
    assert tr | va == set(range(10_000))
    assert data.labels[list(va)].sum() == 1000
    assert data.labels[list(tr)].sum() == 4000


def test_split_is_reproducible(synthetic_csv):
    data = load_labeled(synthetic_csv)
    assert make_split(data, seed=42) == make_split(data, seed=42)
    assert make_split(data, seed=42)["val_ids"] != make_split(data, seed=7)["val_ids"]


def test_split_roundtrip_aligns_ids_texts_labels(synthetic_csv, tmp_path):
    data = load_labeled(synthetic_csv)
    path = save_split(make_split(data), tmp_path / "split.json")
    s = load_split(data, path)
    assert len(s.train_texts) == 8000 and len(s.val_texts) == 2000
    original = pd.read_csv(synthetic_csv).set_index("id")
    for ids, texts, labels in ((s.train_ids, s.train_texts, s.train_labels), (s.val_ids, s.val_texts, s.val_labels)):
        assert list(original.loc[ids, "text"]) == texts
        assert list(original.loc[ids, "label"]) == list(labels)


def test_split_independent_of_file_row_order(synthetic_csv, tmp_path):
    data = load_labeled(synthetic_csv)
    shuffled = tmp_path / "shuffled.csv"
    pd.read_csv(synthetic_csv).sample(frac=1, random_state=1).to_csv(shuffled, index=False)
    assert make_split(load_labeled(shuffled))["val_ids"] == make_split(data)["val_ids"]


def test_save_split_refuses_overwrite(synthetic_csv, tmp_path):
    split = make_split(load_labeled(synthetic_csv))
    path = save_split(split, tmp_path / "split.json")
    with pytest.raises(FileExistsError):
        save_split(split, path)


def test_load_split_rejects_different_train_file(synthetic_csv, tmp_path):
    data = load_labeled(synthetic_csv)
    path = save_split(make_split(data), tmp_path / "split.json")
    tampered = json.loads(path.read_text())
    tampered["source_sha256"] = "0" * 64
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="differs"):
        load_split(data, path)


def test_detect_columns():
    assert detect_columns(pd.DataFrame(columns=["ID", "Text", "label"])) == Columns("Text", "label", "ID")
    assert detect_columns(pd.DataFrame(columns=["text"]), require_label=False) == Columns("text", None, None)
    with pytest.raises(ValueError, match="Ambiguous"):
        detect_columns(pd.DataFrame(columns=["text", "essay", "label"]))
    with pytest.raises(ValueError, match="No label"):
        detect_columns(pd.DataFrame(columns=["text"]))


def test_find_near_duplicates():
    texts = [
        "The quick brown fox jumps over the lazy dog near the river bank today.",
        "The quick brown fox jumps over the lazy dog near the river bank today!",
        "Completely unrelated sentence about machine learning and evaluation metrics.",
    ]
    pairs = find_near_duplicates(texts, threshold=0.9)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
