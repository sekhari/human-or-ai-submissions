"""Synthetic, data-free checks for the raw-text style feature contract."""
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.text_generalization.style_search import (
    StyleFeatures, build_pipeline, candidate_configs, extract_style_features,
)


class StyleFeatureChecks(unittest.TestCase):
    def test_literal_counts_and_rate_denominators(self):
        features = extract_style_features('Hi Hi!\n\nOK 12.')
        self.assertEqual(features['char_count'], 14)
        self.assertEqual(features['token_count'], 4)
        self.assertEqual(features['unique_token_count'], 3)
        self.assertEqual(features['sentence_count'], 2)
        self.assertEqual(features['paragraph_count'], 2)
        self.assertEqual(features['word_type_token_ratio'], 0.75)
        self.assertEqual(features['adjacent_word_repeat_rate'], 1 / 3)
        self.assertEqual(features['digit_char_fraction'], 2 / 14)
        self.assertEqual(features['exclamation_per_char'], 1 / 14)

    def test_empty_and_unicode_are_finite(self):
        rows = ['', None, '   \n\t', 'Été 中文🙂 — café 3.14', 'a' * 5000]
        x = StyleFeatures().fit_transform(rows)
        self.assertEqual(x.shape[0], 5)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(extract_style_features('')['token_count'], 0)
        self.assertEqual(extract_style_features(None)['char_count'], 0)

    def test_features_have_no_batch_fitted_statistics(self):
        text = 'A short sentence. Another one?'
        first = StyleFeatures().fit(['training only']).transform([text])
        second = StyleFeatures().fit(['totally different fit data']).transform([text, 'other'])[0]
        np.testing.assert_array_equal(first[0], second)
        self.assertEqual(len(StyleFeatures().get_feature_names_out()), first.shape[1])

    def test_markdown_and_repeat_signals(self):
        features = extract_style_features('# Heading\n\n- alpha alpha alpha\n- beta\n```\nx=1\n```')
        self.assertEqual(features['markdown_heading_line_fraction'], 1 / 7)
        self.assertEqual(features['markdown_list_line_fraction'], 2 / 7)
        self.assertEqual(features['code_fence_count'], 2)
        self.assertGreater(features['duplicate_bigram_rate'], 0)

    def test_configured_pipelines_are_cloneable_and_fit_toy_data(self):
        texts = ['short simple.'] * 64 + ['# HEADING\n\n- LONG FORM 123!!\n```x=3```'] * 64
        y = np.array([0] * 64 + [1] * 64)
        for config in candidate_configs():
            pipeline = clone(build_pipeline(config))
            pipeline.fit(texts, y)
            p = pipeline.predict_proba(texts)
            self.assertEqual(p.shape, (128, 2))
            self.assertTrue(np.isfinite(p).all())
            self.assertGreater(p[-1, 1], p[0, 1])


if __name__ == '__main__':
    unittest.main()
