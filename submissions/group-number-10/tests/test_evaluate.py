"""Evaluation utilities on small synthetic predictions."""

import numpy as np
import pytest

from src.classifiers import TextClassifier
from src.data import SplitData
from src.evaluate import (
    Timer,
    compute_metrics,
    evaluate_classifier,
    load_all_results,
    load_predictions,
    load_result,
    validate_probabilities,
)


def test_auc_known_values():
    y = [0, 0, 1, 1]
    assert compute_metrics(y, [0.1, 0.2, 0.8, 0.9])["auc"] == 1.0
    assert compute_metrics(y, [0.9, 0.8, 0.2, 0.1])["auc"] == 0.0
    # pairs (pos, neg): (0.35>0.1) (0.35<0.4) (0.8>0.1) (0.8>0.4) -> 3/4
    assert compute_metrics(y, [0.1, 0.4, 0.35, 0.8])["auc"] == 0.75


def test_threshold_metrics_and_confusion_matrix():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == 0.5


@pytest.mark.parametrize("bad", [np.zeros((3, 2)), [0.1, np.nan, 0.3], [0.1, 1.2, 0.3], [-0.1, 0.5, 0.5]])
def test_validate_probabilities_rejects_bad_output(bad):
    with pytest.raises(ValueError):
        validate_probabilities(bad)


def test_validate_probabilities_length():
    with pytest.raises(ValueError, match="Expected 4"):
        validate_probabilities([0.1, 0.2, 0.3], n_expected=4)


def test_timer():
    with Timer() as t:
        sum(range(1000))
    assert t.seconds > 0


class KeywordModel:
    """Toy model: P(AI) is high if the text contains 'delve'."""

    def fit(self, texts, labels):
        self.fitted_on = len(texts)
        return self

    def predict_proba(self, texts):
        return np.array([0.9 if "delve" in t else 0.2 for t in texts])

    def get_config(self):
        return {"keyword": "delve", "seed": 0}


@pytest.fixture
def tiny_split():
    return SplitData(
        train_ids=np.array([10, 11, 12, 13]),
        train_texts=["we delve", "hello", "delve deeper", "plain text"],
        train_labels=np.array([1, 0, 1, 0]),
        val_ids=np.array([20, 21, 22]),
        val_texts=["plain", "let us delve", "ordinary"],
        val_labels=np.array([0, 1, 1]),
        seed=0,
    )


def test_model_satisfies_protocol():
    assert isinstance(KeywordModel(), TextClassifier)


def test_evaluate_classifier_end_to_end(tiny_split, tmp_path):
    model = KeywordModel()
    res = evaluate_classifier(model, tiny_split, "keyword", seed=0,
                              metrics_dir=tmp_path / "m", predictions_dir=tmp_path / "p")
    assert model.fitted_on == 4  # trained on training partition only
    assert res.train_auc == 1.0 and res.val_auc == 0.75
    assert res.hyperparameters == {"keyword": "delve", "seed": 0}

    loaded = load_result(tmp_path / "m" / f"{res.run_name}.json")
    assert loaded == res

    preds = load_predictions(tmp_path / "p" / f"{res.run_name}_val.csv")
    assert list(preds.columns) == ["id", "label", "prob_ai"]
    assert list(preds["id"]) == [20, 21, 22]
    assert list(preds["label"]) == [0, 1, 1]
    assert list(preds["prob_ai"]) == [0.2, 0.9, 0.2]

    table = load_all_results(tmp_path / "m")
    assert len(table) == 1 and table.loc[0, "val_auc"] == 0.75


def test_load_all_results_empty(tmp_path):
    assert load_all_results(tmp_path).empty
