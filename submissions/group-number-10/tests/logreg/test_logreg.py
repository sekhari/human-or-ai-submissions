"""Logistic regression track: cleaning, style features, LM cache, both models on synthetic text."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from logreg import features
from logreg.features import clean_text, lm_matrix, style_features, text_key
from logreg.model import BlockLogReg, StackedLogReg
from src.classifiers.base import TextClassifier


def test_clean_text_strips_attacks_and_keeps_newlines():
    text = 'wоrd​ don’t — end\nnext'          # Cyrillic о, zero-width space, curly quote, em dash
    assert clean_text(text) == "word don't - end NEWLINE next"


def test_style_features_on_fixed_text():
    f = style_features('One two three. Four five\nsix')
    assert len(f) == 20
    assert f['ends_mid_sentence'] == 1.0
    assert f['sent_len_mean'] == 3.0
    assert f['newline_rate'] == pytest.approx(1 / 28)
    assert f['homoglyph_count'] == 0


def test_lm_matrix_reads_cache_by_text(tmp_path, monkeypatch):
    cache = tmp_path / 'lm.csv'
    rows = [[text_key(t)] + [float(i)] * len(features.LM_COLUMNS) for i, t in enumerate(['a b c', 'd e f'])]
    pd.DataFrame(rows, columns=['key'] + features.LM_COLUMNS).to_csv(cache, index=False)
    monkeypatch.setattr(features, 'LM_CACHE', cache)
    X = lm_matrix(['d e f', 'a b c', 'd e f'])   # order and repeats follow the input
    assert X[:, 0].tolist() == [1.0, 0.0, 1.0]


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    human = ['honestly', 'yeah', 'kinda', 'lol', 'gonna', 'my', 'dog', 'yesterday']
    ai = ['furthermore', 'additionally', 'crucial', 'delve', 'moreover', 'landscape', 'notably', 'overall']
    texts, labels = [], []
    for i in range(n):
        label = i % 2
        words = rng.choice(ai if label else human, size=12)
        texts.append(' '.join(words) + ('.' if label else ''))
        labels.append(label)
    return texts, np.array(labels)


@pytest.mark.parametrize('make', [
    lambda: BlockLogReg(['words', 'style'], C=10.0),
    lambda: StackedLogReg({'words': 10.0, 'chars': 10.0, 'style': 1.0}),
])
def test_models_follow_team_interface(make):
    train_texts, train_labels = synthetic(60, seed=0)
    val_texts, val_labels = synthetic(40, seed=1)
    model = make()
    assert isinstance(model, TextClassifier)

    p = model.fit(train_texts, train_labels).predict_proba(val_texts)
    assert p.shape == (40,) and np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(val_labels, p) > 0.9
    json.dumps(model.get_config())
