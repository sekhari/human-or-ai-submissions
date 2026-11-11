"""Run names and checkpoint folders for each model family (no model download needed)."""

import pytest

pytest.importorskip("transformers")
from experiments.distilbert_train import model_family, run_name_for  # noqa: E402
from src.classifiers.distilbert import DistilBertConfig  # noqa: E402


@pytest.mark.parametrize("model_name, family", [
    ("distilbert-base-uncased", "distilbert"),
    ("checkpoints/mlm/dapt-split-nl", "distilbert"),
    ("microsoft/deberta-v3-base", "deberta-v3-base"),
    ("models/deberta-v3-large", "deberta-v3-large"),
    ("FacebookAI/roberta-large", "roberta-large"),
    ("models/roberta-large/", "roberta-large"),
    ("models/electra-large-discriminator", "electra-large-discriminator"),
])
def test_model_family(model_name, family):
    assert model_family(DistilBertConfig(model_name=model_name)) == family


def test_run_name_tags():
    cfg = DistilBertConfig(model_name="FacebookAI/roberta-large", learning_rate=1e-5, epochs=3,
                           newline_marker="[NL]", warmup_ratio=0.1)
    assert run_name_for(cfg) == "roberta-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed42"
    assert run_name_for(DistilBertConfig()) == "distilbert_lr2e-05_ep3_bs16_len512_seed42"
    xl = DistilBertConfig(model_name="models/deberta-v2-xlarge", learning_rate=5e-6, newline_marker="[NL]",
                          warmup_ratio=0.1, bf16=True, gradient_checkpointing=True)
    assert run_name_for(xl) == "deberta-v2-xlarge_nl_wu0.1_bf16_lr5e-06_ep3_bs16_len512_seed42"
    aug = DistilBertConfig(model_name="models/deberta-v3-large", learning_rate=1e-5, newline_marker="[NL]",
                           warmup_ratio=0.1, layout_aug=0.5)
    assert run_name_for(aug) == "deberta-v3-large_nl_wu0.1_aug0.5_lr1e-05_ep3_bs16_len512_seed42"
