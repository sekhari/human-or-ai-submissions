"""Small offline checks for the custom training loop; no pretrained weights needed."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

from src.deberta import LengthBatches, PadBatch, optimizer_update, precision_for, predict, train_model


class TinyTokenizer:
    def __call__(self, texts, truncation=True, max_length=32, **kwargs):
        ids = [[1] + [3 + len(w) % 50 for w in s.split()][:max_length - 2] + [2] for s in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}

    def pad(self, examples, padding=True, pad_to_multiple_of=8, return_tensors="pt"):
        size = ((max(len(x["input_ids"]) for x in examples) + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        batch = {key: torch.tensor([x[key] + [0] * (size - len(x[key])) for x in examples])
                 for key in ("input_ids", "attention_mask")}
        if "labels" in examples[0]:
            batch["labels"] = torch.tensor([x["labels"] for x in examples])
        return batch

    def save_pretrained(self, path):
        pass


def tiny_model():
    return DebertaV2ForSequenceClassification(DebertaV2Config(
        vocab_size=64, hidden_size=24, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=32, max_position_embeddings=64, num_labels=2,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        pooler_hidden_size=24, problem_type="single_label_classification",
    ))


class DebertaTrainingTests(unittest.TestCase):
    def test_t4_precision_selection_and_bf16_guard(self):
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            self.assertEqual(precision_for(torch.device("cuda")), "fp16")
            with self.assertRaisesRegex(ValueError, "T4"):
                precision_for(torch.device("cuda"), "bf16")
        self.assertEqual(precision_for(torch.device("cpu")), "fp32")

    def test_amp_unscales_before_clipping_and_skips_overflow_updates(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cpu", init_scale=128)
        old = model.weight.detach().clone()
        scaler.scale(model(torch.ones(2, 2)).square().mean()).backward()
        _, applied = optimizer_update(model, optimizer, scheduler, scaler)
        self.assertTrue(applied)
        self.assertFalse(torch.equal(old, model.weight))
        epoch = scheduler.last_epoch
        old = model.weight.detach().clone()
        scaler.scale(model(torch.ones(2, 2)).sum()).backward()
        model.weight.grad.fill_(float("inf"))
        _, applied = optimizer_update(model, optimizer, scheduler, scaler)
        self.assertFalse(applied)
        self.assertTrue(torch.equal(old, model.weight))
        self.assertEqual(scheduler.last_epoch, epoch)

    def test_bounded_padding_preserves_tokens_and_labels(self):
        tokenizer = TinyTokenizer()
        examples = [{"input_ids": [1, 5, 2], "attention_mask": [1, 1, 1], "labels": 1},
                    {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": 0}]
        batch = PadBatch(tokenizer, 128)(examples)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 128))
        self.assertEqual(batch["attention_mask"].sum(dim=1).tolist(), [3, 2])
        self.assertEqual(batch["labels"].tolist(), [1, 0])

    def test_batches_cover_every_row_once_and_are_reproducible(self):
        sampler = LengthBatches([2, 15, 6, 9, 3, 21, 7], 3, 42)
        first = list(sampler)
        self.assertEqual(sorted(i for b in first for i in b), list(range(7)))
        self.assertEqual(first, list(sampler))
        self.assertEqual(len(first), len(sampler))

    def test_training_updates_encoder_and_writes_reloadable_checkpoint(self):
        torch.set_num_threads(2)
        frame = pd.DataFrame({"text": ["a short text", "these are somewhat longer words"] * 9,
                              "label": [0, 1] * 9, "id": list(range(18))})
        model = tiny_model()
        before = model.deberta.encoder.layer[0].attention.self.query_proj.weight.detach().clone()
        config = {"seed": 42, "batch_size": 3, "gradient_accumulation": 2,
                  "epochs": 2, "max_length": 32, "eval_batch_size": 4,
                  "encoder_lr": 0.001, "head_lr": 0.002, "weight_decay": 0.01,
                  "warmup_ratio": 0.1, "precision": "fp32"}
        with tempfile.TemporaryDirectory() as tmp, patch("src.deberta.load_model", return_value=model), patch("src.deberta.load_tokenizer", return_value=TinyTokenizer()):
            updates = []
            epoch, history = train_model(frame.iloc[:13], frame.iloc[13:], config,
                                         torch.device("cpu"), Path(tmp), lambda _: None, updates.append)
            self.assertIn(epoch, (1, 2))
            self.assertEqual(len(history), 2)
            self.assertEqual(updates[-1]["optimizer_updates"], 6)
            self.assertFalse(torch.equal(before, model.deberta.encoder.layer[0].attention.self.query_proj.weight))
            restored = DebertaV2ForSequenceClassification.from_pretrained(Path(tmp) / "best")
            p = predict(restored, frame.iloc[13:], TinyTokenizer(), torch.device("cpu"),
                        max_length=32, batch_size=3, precision="fp32")
            self.assertEqual(p.shape, (5,))
            self.assertTrue(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all())
            # Evaluation is ordered and gives identical rows identical predictions.
            self.assertAlmostEqual(p[0], p[2], places=7)
            self.assertAlmostEqual(p[1], p[3], places=7)

    def test_epoch_resume_matches_uninterrupted_optimizer_updates(self):
        frame = pd.DataFrame({"id": range(13), "text": ["some example text"] * 13,
                              "label": [0, 1] * 6 + [0]})
        config = {"seed": 42, "batch_size": 3, "gradient_accumulation": 2,
                  "epochs": 2, "max_length": 32, "eval_batch_size": 4,
                  "encoder_lr": 0.001, "head_lr": 0.002, "weight_decay": 0.01,
                  "warmup_ratio": 0.1, "precision": "fp32"}
        def factory(seed, checkpoint=None):
            torch.manual_seed(seed)
            return tiny_model() if checkpoint is None else DebertaV2ForSequenceClassification.from_pretrained(checkpoint)
        with tempfile.TemporaryDirectory() as tmp, patch("src.deberta.load_model", side_effect=factory), patch("src.deberta.load_tokenizer", return_value=TinyTokenizer()):
            root = Path(tmp)
            train_model(frame, None, config, torch.device("cpu"), root / "continuous", lambda _: None, lambda _: None)
            resumed_config = {**config, "resume": True, "one_epoch_per_process": True}
            _, first = train_model(frame, None, resumed_config, torch.device("cpu"), root / "resumed", lambda _: None, lambda _: None)
            self.assertEqual(len(first), 1)
            _, second = train_model(frame, None, resumed_config, torch.device("cpu"), root / "resumed", lambda _: None, lambda _: None)
            self.assertEqual(len(second), 2)
            a = DebertaV2ForSequenceClassification.from_pretrained(root / "continuous/final")
            b = DebertaV2ForSequenceClassification.from_pretrained(root / "resumed/final")
            for name, value in a.state_dict().items():
                self.assertTrue(torch.equal(value, b.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
