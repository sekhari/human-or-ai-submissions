import numpy as np
import pytest

pytest.importorskip("transformers")
from experiments.tta import apply_layout, tta_average  # noqa: E402


def test_apply_layout_keeps_words_and_none_is_identity():
    texts = ["a line that\nwraps here\n\nnew paragraph", "no breaks at all"]
    assert apply_layout(texts, "none") == texts
    for layout in ("unwrap", "hard_wrap", "no_newlines"):
        out = apply_layout(texts, layout)
        assert [t.split() for t in out] == [t.split() for t in texts]
    assert apply_layout(texts, "unwrap")[0] == "a line that wraps here\n\nnew paragraph"
    with pytest.raises(ValueError):
        apply_layout(texts, "short_lines")


def test_tta_average_weights_layouts_equally():
    parts = {("m1", "none"): np.array([0.2, 0.8]), ("m2", "none"): np.array([0.4, 0.6]),
             ("m1", "unwrap"): np.array([0.0, 1.0]), ("m2", "unwrap"): np.array([0.2, 0.8])}
    # none: [0.3, 0.7]; unwrap: [0.1, 0.9]; mean: [0.2, 0.8]
    np.testing.assert_allclose(tta_average(parts), [0.2, 0.8])
    assert np.allclose(tta_average({k: v for k, v in parts.items() if k[1] == "none"}), [0.3, 0.7])
