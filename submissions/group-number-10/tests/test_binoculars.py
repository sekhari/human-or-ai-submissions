"""Binoculars feature math on tiny hand-made logits (no model download needed)."""

import math

import pytest

torch = pytest.importorskip("torch")
from experiments.binoculars import binoculars_features  # noqa: E402


def test_uniform_models_give_log_vocab_and_score_one():
    logits = torch.zeros(1, 5, 4)
    f = binoculars_features(logits, logits, torch.tensor([[0, 1, 2, 3, 0]]), torch.ones(1, 5))
    for k in ("log_ppl_obs", "log_ppl_perf", "log_xppl"):
        assert f[k].item() == pytest.approx(math.log(4))
    assert f["score"].item() == pytest.approx(1.0)
    assert f["n_tokens"].item() == 4


def test_padding_is_ignored():
    torch.manual_seed(0)
    obs, perf = torch.randn(1, 6, 10), torch.randn(1, 6, 10)
    ids = torch.randint(0, 10, (1, 6))
    short = binoculars_features(obs[:, :4], perf[:, :4], ids[:, :4], torch.ones(1, 4))
    padded = binoculars_features(obs, perf, ids, torch.tensor([[1, 1, 1, 1, 0, 0]]))
    for k in short:
        assert padded[k].item() == pytest.approx(short[k].item(), rel=1e-5)


def test_predictable_text_has_lower_perplexity():
    ids = torch.tensor([[1, 2, 3, 1, 2]])
    sharp = torch.full((1, 5, 4), -5.0)
    sharp[0, torch.arange(4), ids[0, 1:]] = 5.0  # each position predicts the actual next token
    flat = torch.zeros(1, 5, 4)
    mask = torch.ones(1, 5)
    assert (binoculars_features(sharp, sharp, ids, mask)["log_ppl_perf"]
            < binoculars_features(flat, flat, ids, mask)["log_ppl_perf"])
