import unittest

import numpy as np
import torch

from src.gpt2_robust_features import from_logits
from src.robust_boost import predict_candidate, topic_weights, training_views


class FeatureTests(unittest.TestCase):
    def test_chunked_matches_independent_dense_moments_with_padding(self):
        torch.manual_seed(14)
        logits = torch.randn(3, 12, 37)
        ids = torch.randint(0, 37, (3, 12))
        mask = torch.ones(3, 12, dtype=torch.long)
        mask[1, 8:] = 0
        mask[2, 4:] = 0
        actual = from_logits(logits, ids, mask, token_chunk=3)
        logp = logits[:, :-1].log_softmax(-1).numpy()
        probability = np.exp(logp)
        observed = np.take_along_axis(logp, ids[:, 1:, None].numpy(), axis=2).squeeze(-1)
        expected = (probability * logp).sum(-1)
        variance = np.maximum((probability * logp ** 2).sum(-1) - expected ** 2, 1e-6)
        for i, count in enumerate([11, 7, 3]):
            n = -observed[i, :count]
            rank = (logp[i, :count] > observed[i, :count, None]).sum(-1)
            gap = observed[i, :count] - expected[i, :count]
            z = gap / np.sqrt(variance[i, :count])
            reference = {
                "gpt2_nll_mean": n.mean(), "gpt2_nll_std": n.std(),
                "gpt2_nll_q10": np.quantile(n, .1), "gpt2_nll_q50": np.quantile(n, .5),
                "gpt2_nll_q90": np.quantile(n, .9),
                "gpt2_nll_first_half": n[:count // 2].mean(),
                "gpt2_nll_second_half": n[count // 2:].mean(),
                "gpt2_top1": (rank == 0).mean(), "gpt2_top10": (rank < 10).mean(),
                "gpt2_top100": (rank < 100).mean(), "gpt2_rare": (rank >= 1000).mean(),
                "gpt2_diff_std": np.diff(n).std(),
                "curv_total": gap.sum() / np.sqrt(variance[i, :count].sum()),
                "curv_z_mean": z.mean(), "curv_z_std": z.std(),
                "curv_z_q10": np.quantile(z, .1), "curv_z_q90": np.quantile(z, .9),
                "curv_entropy_mean": -expected[i, :count].mean(),
                "curv_entropy_std": expected[i, :count].std(),
            }
            for key, value in reference.items():
                np.testing.assert_allclose(actual.loc[i, key], value, rtol=3e-5, atol=2e-6)

    def test_empty_predicted_sequence_rejected(self):
        with self.assertRaises(ValueError):
            from_logits(torch.zeros(1, 3, 5), torch.zeros(1, 3, dtype=torch.long),
                        torch.tensor([[1, 0, 0]]))


class TrainingTests(unittest.TestCase):
    def test_topic_weight_preserves_class_mass_and_raises_rare_topics(self):
        y = np.array([0] * 5 + [1] * 5)
        topics = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
        w = topic_weights(y, topics)
        self.assertGreater(w[4], w[0])
        for label in (0, 1):
            self.assertAlmostEqual(w[y == label].sum(), (y == label).sum())
        np.testing.assert_array_equal(topic_weights(y, topics, power=0), np.ones(10))

    def test_paired_views_preserve_document_labels_and_total_mass(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        flat, y, w = x + 100, np.array([0, 1, 0]), np.array([.5, 2, .5])
        augmented, labels, weights = training_views(x, flat, y, w, .25)
        np.testing.assert_array_equal(augmented[:3], x)
        np.testing.assert_array_equal(augmented[3:], flat)
        np.testing.assert_array_equal(labels[:3], labels[3:])
        np.testing.assert_allclose(weights[:3] + weights[3:], w)

    def test_bootstrap_samples_ids_before_expanding_and_disables_early_stop(self):
        from unittest.mock import patch
        x = np.arange(24).reshape(12, 2)
        flat, y = x + 1000, np.arange(12) % 2
        fits = []
        class FakeBooster:
            def __init__(self, **kw):
                self.kw = kw
            def fit(self, data, labels, sample_weight):
                fits.append((self.kw, data.copy(), labels.copy(), sample_weight.copy()))
            def predict_proba(self, data):
                return np.tile([.4, .6], (len(data), 1))
        spec = dict(params={}, bags=2, topic_power=0., flat_share=.25)
        with patch("src.robust_boost.HistGradientBoostingClassifier", FakeBooster):
            p, q = predict_candidate(spec, x, flat, y, np.zeros(12), x[:3], flat[:3])
        self.assertEqual(len(p), 3)
        self.assertEqual(len(q), 3)
        for kwargs, data, labels, weights in fits:
            self.assertFalse(kwargs["early_stopping"])
            np.testing.assert_array_equal(data[12:], data[:12] + 1000)
            np.testing.assert_array_equal(labels[:12], labels[12:])
            np.testing.assert_allclose(weights[:12] + weights[12:], 1)


if __name__ == "__main__":
    unittest.main()
