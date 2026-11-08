"""Offline integration checks for pseudo supervision and masked-only MLM."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification

from src.roberta_experiment import MaskPad, SoftDataset, SoftPad, agreement_pseudo
from src.roberta_experiment import masked_loss, predict, soft_loss, train_epoch


class TinyTokenizer:
    mask_token_id = 63

    def __len__(self):
        return 64

    def __call__(self, texts, truncation=True, max_length=32, **kwargs):
        ids = [[0] + [3+len(w)%50 for w in s.split()][:max_length-2] + [2] for s in texts]
        return {"input_ids": ids, "attention_mask": [[1]*len(row) for row in ids]}

    def pad(self, examples, padding=True, pad_to_multiple_of=8, return_tensors="pt"):
        n = ((max(len(row["input_ids"]) for row in examples)+pad_to_multiple_of-1)//pad_to_multiple_of)*pad_to_multiple_of
        return {key: torch.tensor([row[key]+[1 if key=="input_ids" else 0]*(n-len(row[key])) for row in examples]) for key in ["input_ids", "attention_mask"]}

    def get_special_tokens_mask(self, ids, already_has_special_tokens=True):
        return [int(i in {0, 1, 2, 63}) for i in ids]

    def save_pretrained(self, path):
        pass


def tiny_config():
    return RobertaConfig(vocab_size=64, hidden_size=24, num_hidden_layers=2, num_attention_heads=4,
                         intermediate_size=32, max_position_embeddings=66, num_labels=2,
                         hidden_dropout_prob=0, attention_probs_dropout_prob=0, classifier_dropout=0,
                         pad_token_id=1, bos_token_id=0, eos_token_id=2)


class RobertaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_soft_binary_loss_matches_cross_entropy_and_handles_uncertainty(self):
        logits = torch.tensor([[1., -2.], [3., 5.], [-1., -.5]], requires_grad=True)
        y = torch.tensor([0, 1, 1])
        torch.testing.assert_close(soft_loss(logits, y.float()), torch.nn.functional.cross_entropy(logits, y))
        targets = torch.tensor([.1, .8, .4])
        expected = -(torch.stack([1-targets, targets], 1)*logits.log_softmax(-1)).sum(-1).mean()
        torch.testing.assert_close(soft_loss(logits, targets), expected)
        soft_loss(logits, targets).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_pseudo_selection_requires_both_confidences_and_agreement(self):
        b = np.array([.999, .001, .999, .8, .999])
        n = np.array([.95, .05, .01, .99, .6])
        keep, targets = agreement_pseudo(b, n)
        np.testing.assert_array_equal(keep, [True, True, False, False, False])
        np.testing.assert_allclose(targets, (b+n)/2)
        with self.assertRaises(ValueError): agreement_pseudo(b, n[:2])
        with self.assertRaises(ValueError): agreement_pseudo(b*2, n)

    def test_masking_protects_special_tokens_and_padding(self):
        tok = TinyTokenizer(); row = tok(["one two three four five six"])
        examples = [{k: v[0] for k, v in row.items()}]
        torch.manual_seed(42); batch = MaskPad(tok, 8)(examples)
        selected = batch["labels"].ne(-100)
        self.assertTrue(selected.any())
        self.assertFalse(selected[0, 0]); self.assertFalse(selected[0, 7])
        torch.manual_seed(42); repeat = MaskPad(tok, 8)(examples)
        torch.testing.assert_close(batch["input_ids"], repeat["input_ids"])
        padded = MaskPad(tok, 16)(examples)
        self.assertTrue(padded["labels"][padded["attention_mask"].eq(0)].eq(-100).all())

    def test_masked_position_loss_matches_full_vocabulary_forward(self):
        model = RobertaForMaskedLM(tiny_config()).eval()
        tok = TinyTokenizer(); data = tok(["a text with enough words for masking", "another longer text with tokens"])
        batch = MaskPad(tok, 8)([{k: v[i] for k, v in data.items()} for i in range(2)])
        expected = model(**batch).loss
        observed = masked_loss(model, batch)
        torch.testing.assert_close(observed, expected)
        observed.backward()
        self.assertGreater(model.roberta.encoder.layer[-1].attention.self.query.weight.grad.abs().sum().item(), 0)

    def test_pseudo_batches_preserve_supervised_optimizer_budget_and_resume(self):
        frame = pd.DataFrame({"id": range(13), "text": ["some example words here"]*13,
                              "soft_target": [0., 1.]*6+[0.]})
        pseudo = pd.DataFrame({"id": range(100, 106), "text": ["another example of text"]*6,
                               "soft_target": [.99, .01]*3})
        config = {"seed": 42, "batch_size": 3, "gradient_accumulation": 2, "epochs": 2,
                  "max_length": 32, "encoder_lr": .001, "head_lr": .002,
                  "weight_decay": .01, "pseudo_weight": .25}
        def factory(task="classification", checkpoint=None, adapted=None, seed=42):
            torch.manual_seed(seed)
            return RobertaForSequenceClassification(tiny_config()) if checkpoint is None else RobertaForSequenceClassification.from_pretrained(checkpoint)
        with tempfile.TemporaryDirectory() as tmp, patch("src.roberta_experiment.tokenizer", return_value=TinyTokenizer()), patch("src.roberta_experiment.load_model", side_effect=factory):
            path = Path(tmp)
            first = train_epoch(frame, config, path, 1, torch.device("cpu"), lambda row: None, pseudo=pseudo)
            second = train_epoch(frame, config, path, 2, torch.device("cpu"), lambda row: None, pseudo=pseudo)
            self.assertEqual(first["updates"], 3); self.assertEqual(second["updates"], 6)
            self.assertEqual(first["pseudo_batches_used"], 3)
            restored = factory(checkpoint=path / "epoch_2")
            p = predict(restored, frame, TinyTokenizer(), torch.device("cpu"), batch_size=3)
            self.assertEqual(len(p), 13); self.assertTrue(np.isfinite(p).all())
            self.assertLess(float(np.max(p)-np.min(p)), 1e-7)
            changed = config.copy(); changed["encoder_lr"] = .1
            with self.assertRaisesRegex(AssertionError, "Resume inputs changed"):
                train_epoch(frame, changed, path, 3, torch.device("cpu"), lambda row: None, pseudo=pseudo)


if __name__ == "__main__":
    unittest.main()
