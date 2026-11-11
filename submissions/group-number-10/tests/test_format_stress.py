"""Layout perturbations change line breaks only, never the words."""

import numpy as np
import pytest

pytest.importorskip("transformers")
from experiments.format_stress import hard_wrap, no_newlines, short_lines, unwrap, variants  # noqa: E402

TEXT = "The first line of a wrapped\nparagraph goes on here.\n\nA second paragraph\nwith two lines."


def test_words_are_preserved():
    for v in variants([TEXT]).values():
        assert v[0].split() == TEXT.split()


def test_no_newlines_and_unwrap():
    assert "\n" not in no_newlines(TEXT)
    assert unwrap(TEXT) == "The first line of a wrapped paragraph goes on here.\n\nA second paragraph with two lines."


def test_hard_wrap_width_and_paragraphs():
    long = " ".join(["word"] * 60) + "\n\nshort"
    out = hard_wrap(long, width=70)
    assert max(len(line) for line in out.split("\n")) <= 70
    assert out.endswith("\n\nshort")


def test_short_lines_lengths():
    out = short_lines(" ".join(map(str, range(100))), np.random.default_rng(0))
    sizes = [len(line.split()) for line in out.split("\n")]
    assert all(4 <= s <= 9 for s in sizes[:-1]) and sum(sizes) == 100


def test_augment_layouts_keeps_words_and_rate():
    from src.layout import augment_layouts
    texts = [f"line one {i}\nline two of passage {i} with more words here" for i in range(400)]
    out = augment_layouts(texts, 0.5, np.random.default_rng(0))
    assert all(o.split() == t.split() for o, t in zip(out, texts))
    assert 0.3 < np.mean([o != t for o, t in zip(out, texts)]) < 0.55  # some draws leave the text unchanged
    assert augment_layouts(texts, 0.0, np.random.default_rng(0)) == texts
