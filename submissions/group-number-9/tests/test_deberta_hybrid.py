"""Offline checks for representation ordering, normalization, and neural fusion."""
import unittest

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from src.deberta import predict
from src.deberta_hybrid import FusionHead, extract, prepare_inputs, train_head
from test_deberta import TinyTokenizer, tiny_model


class HybridTests(unittest.TestCase):
    def test_sorted_extraction_matches_classifier_and_restores_row_order(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = tiny_model().eval()
        frame = pd.DataFrame({"text": ["one two three four five", "a", "one two", "a", "longer text with six separate words"]})
        tokenizer = TinyTokenizer()
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        vectors, logits = extract(model, frame, tokenizer, torch.device("cpu"), batch_size=2, max_length=32)
        probabilities = predict(model, frame, tokenizer, torch.device("cpu"), max_length=32, batch_size=2, precision="fp32")
        np.testing.assert_allclose(expit(logits), probabilities, atol=1e-7)
        np.testing.assert_allclose(vectors[1], vectors[3], atol=1e-7)
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(before[name], p))
            self.assertIsNone(p.grad)

    def test_normalizers_fit_training_rows_only_and_keep_feature_branches(self):
        vectors = np.array([[1., 2.], [3., 6.], [5., 10.]], dtype=np.float32)
        style = np.array([[0., 1.], [1., 2.], [3., 5.]], dtype=np.float32)
        xf, xe, dim, ts, ss = prepare_inputs(vectors, style, vectors + 30, style + 50, "hybrid")
        np.testing.assert_allclose(ts.mean_, vectors.mean(axis=0))
        np.testing.assert_allclose(ss.mean_, np.log1p(style).mean(axis=0), atol=1e-7)
        np.testing.assert_allclose(xf.mean(axis=0), 0, atol=1e-6)
        self.assertFalse(np.allclose(xe.mean(axis=0), 0))
        self.assertEqual((xf.shape[1], dim), (4, 2))

    def test_fusion_backpropagates_through_both_branches_and_trains(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        inputs = rng.normal(size=(128, 6)).astype(np.float32)
        labels = (inputs[:, 0] + inputs[:, 4] > 0).astype(int)
        model = FusionHead(4, 2, width=16, dropout=0)
        torch.nn.functional.cross_entropy(model(torch.tensor(inputs)), torch.tensor(labels)).backward()
        self.assertGreater(float(model.text[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.style[0].weight.grad.abs().sum()), 0)
        config = {"text_dim": 4, "width": 16, "dropout": 0, "lr": 0.01, "weight_decay": 0.001}
        fitted, probabilities, history = train_head(inputs, labels, inputs, config, 42, 20, [20])
        self.assertLess(history[-1]["training_loss"], history[0]["training_loss"])
        self.assertGreater(((probabilities[20] >= 0.5) == labels).mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
