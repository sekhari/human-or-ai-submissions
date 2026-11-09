"""Tests for extract_features.py on the first 20 train passages.

Run:  python -m pytest -v test_features.py
Env:  FEATURE_TEST_SIZE=0.5B|1.5B (default 0.5B), FEATURE_TEST_DEVICE=cpu|mps|cuda (default auto)
"""
import math
import os

import pandas as pd
import pytest
import torch

import extract_features as ef
from normalize import normalize

N_PASSAGES = 20
BATCH = 4  # passages per padded batch in the invariance test
ATOL = 1e-3


@pytest.fixture(scope="module")
def models():
    ef.set_seeds()
    device = ef.pick_device(os.environ.get("FEATURE_TEST_DEVICE"))
    return ef.load_models(os.environ.get("FEATURE_TEST_SIZE", "0.5B"), device,
                          os.environ.get("FEATURE_TEST_DTYPE", "float32"))


@pytest.fixture(scope="module")
def texts():
    df = pd.read_csv(os.path.join(ef.HERE, "data", "train.csv")).head(N_PASSAGES)
    return [normalize(t) for t in df["text"].astype(str)]


@pytest.fixture(scope="module")
def batched(models, texts):
    """Features from padded batches (input order kept so batches mix lengths)."""
    tok, base, instr = models
    rows, padded = [], 0
    for s in range(0, len(texts), BATCH):
        enc, bl, il = ef.forward_logits(texts[s:s + BATCH], tok, base, instr)
        padded += int((enc.attention_mask == 0).any(dim=1).sum())
        rows += ef.features_from_logits(enc, bl, il)
    return rows, padded


# ---------- 1. padding invariance ----------
def test_padding_invariance(models, texts, batched):
    tok, base, instr = models
    rows, padded = batched
    assert padded > 0, "no batch actually contained padding; test would be vacuous"
    for i, t in enumerate(texts):
        enc, bl, il = ef.forward_logits([t], tok, base, instr)
        assert int(enc.attention_mask.sum()) == enc.input_ids.shape[1]  # alone: no padding
        alone = ef.features_from_logits(enc, bl, il)[0]
        for c in ef.FEATURE_COLS:
            assert alone[c] == pytest.approx(rows[i][c], abs=ATOL), f"passage {i} feature {c}"


# ---------- 2. finiteness ----------
def test_all_features_finite(batched):
    rows, _ = batched
    assert len(rows) == N_PASSAGES
    for i, r in enumerate(rows):
        assert set(r) == set(ef.FEATURE_COLS)
        for c, v in r.items():
            assert math.isfinite(v), f"passage {i} feature {c} = {v}"
    assert all(2 <= r["n_tokens"] <= ef.MAX_TOKENS for r in rows)


# ---------- 3. Binoculars matches the reference implementation ----------
# Batched re-statement of ahans30/Binoculars binoculars/metrics.py (perplexity, entropy)
# and detector.py compute_score: ppl from performer logits, x_ppl = CE(softmax(observer),
# performer logits), score = ppl / x_ppl.
_ce = torch.nn.CrossEntropyLoss(reduction="none")


def _ref_perplexity(enc, logits):
    shifted_logits = logits[..., :-1, :].contiguous().float()
    shifted_labels = enc.input_ids[..., 1:].contiguous()
    mask = enc.attention_mask[..., 1:].contiguous()
    return (_ce(shifted_logits.transpose(1, 2), shifted_labels) * mask).sum(1) / mask.sum(1)


def _ref_entropy(p_logits, q_logits, enc, pad_token_id):
    vocab, total = p_logits.shape[-1], q_logits.shape[-2]
    p_proba = torch.softmax(p_logits.float(), dim=-1).view(-1, vocab)
    ce = _ce(input=q_logits.float().view(-1, vocab), target=p_proba).view(-1, total)
    mask = (enc.input_ids != pad_token_id).type(torch.uint8)
    return (ce * mask).sum(1) / mask.sum(1)


def test_binoculars_matches_reference(models, texts):
    tok, base, instr = models
    enc, bl, il = ef.forward_logits(texts[:BATCH], tok, base, instr)  # padded batch
    ours = ef.features_from_logits(enc, bl, il)
    ppl = _ref_perplexity(enc, il)                          # performer = instruct
    x_ppl = _ref_entropy(bl, il, enc, tok.pad_token_id)     # observer = base
    score = (ppl / x_ppl).cpu()
    for i, r in enumerate(ours):
        assert r["bino_ppl"] == pytest.approx(ppl[i].item(), abs=1e-4)
        assert r["bino_xppl"] == pytest.approx(x_ppl[i].item(), abs=1e-4)
        assert r["binoculars"] == pytest.approx(score[i].item(), abs=1e-4)
        # perplexity comes from the performer: equals -mean instruct token log-prob
        assert r["bino_ppl"] == pytest.approx(-r["instr_lp_mean"], abs=1e-4)
    # roles matter: swapping observer/performer gives a different score
    swapped = (_ref_perplexity(enc, bl) / _ref_entropy(il, bl, enc, tok.pad_token_id)).cpu()
    assert not torch.allclose(swapped, score, atol=1e-3)


# ---------- 4. shift by one, first token excluded ----------
def test_shift_synthetic():
    ids = torch.tensor([3, 1, 4, 0, 5])
    V = 6
    logits = torch.full((len(ids), V), -20.0)
    for t in range(len(ids) - 1):
        logits[t, ids[t + 1]] = 20.0     # position t confidently predicts token t+1
    logits[-1, 2] = 20.0                 # last position predicts beyond the sequence
    stats = ef.token_level(logits, ids)
    assert stats["lp"].shape == (len(ids) - 1,)               # first token not scored
    assert torch.allclose(stats["lp"], torch.zeros(len(ids) - 1), atol=1e-6)
    assert torch.all(stats["logrank"] == 0)                   # every target is rank 1

    # first token's identity and last position's logits never affect token stats
    ids2 = ids.clone(); ids2[0] = 2
    logits2 = logits.clone(); logits2[-1] = torch.randn(V)
    s2 = ef.token_level(logits2, ids2)
    for k in stats:
        assert torch.equal(stats[k], s2[k])

    # an unshifted scorer would get these badly wrong
    unshifted = torch.log_softmax(logits, -1).gather(1, ids[:, None]).squeeze(1)
    assert unshifted[1:].mean() < -10


def test_shift_real_model(models, texts):
    tok, base, instr = models
    enc, bl, il = ef.forward_logits([texts[0]], tok, base, instr)
    ids, logits = enc.input_ids[0], bl[0].float()
    manual = [torch.log_softmax(logits[t - 1], -1)[ids[t]].item() for t in range(1, len(ids))]
    stats = ef.token_level(bl[0], ids)
    assert len(manual) == stats["lp"].shape[0] == len(ids) - 1
    assert torch.allclose(stats["lp"].cpu(), torch.tensor(manual), atol=1e-4)
    row = ef.features_from_logits(enc, bl, il)[0]
    assert row["base_lp_mean"] == pytest.approx(sum(manual) / len(manual), abs=1e-4)
