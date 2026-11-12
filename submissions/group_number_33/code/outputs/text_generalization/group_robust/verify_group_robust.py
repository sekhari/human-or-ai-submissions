"""Synthetic checks for train-only layout by label reweighting."""
from pathlib import Path
import sys
import unittest
import numpy as np
from sklearn.base import clone, is_classifier

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from models.text_generalization.group_robust import (
    GroupBalancedStyleClassifier, balance_layout_weights, build_pipeline, candidate_configs,
)


class GroupRobustChecks(unittest.TestCase):
    def test_hand_computed_equal_cell_weighting(self):
        texts=['plain']*4+['line\nnext']*4
        labels=np.array([0,0,0,1,0,0,1,1])
        weights,audit=balance_layout_weights(texts,labels,power=1.,cap=10.)
        np.testing.assert_allclose(weights,[2/3,2/3,2/3,2,1,1,1,1])
        self.assertEqual(audit['target_cell_count'],2.)
        self.assertEqual(audit['present_layout_group_count'],2)

    def test_cap_precedes_mean_normalization(self):
        texts=['plain']*101+['line\nnext']*103
        labels=np.array([0]*100+[1]+[0]*100+[1]*3)
        weights,audit=balance_layout_weights(texts,labels,power=1.,cap=10.)
        self.assertEqual(audit['pre_normalization_weight_stats']['max'],10.)
        self.assertGreater(weights.max(),10.)
        self.assertAlmostEqual(weights.max(),10/(142/204))
        self.assertAlmostEqual(weights.mean(),1.)
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreater(weights.min(),0.)

    def test_validation_labels_are_outside_weight_api(self):
        train_texts=['plain']*4+['line\nnext']*4
        train_labels=np.array([0,0,0,1,0,0,1,1])
        validation_labels=np.array([0,1,0,1])
        before,audit_before=balance_layout_weights(train_texts,train_labels,.5,10.)
        validation_labels=validation_labels[::-1]
        after,audit_after=balance_layout_weights(train_texts,train_labels,.5,10.)
        np.testing.assert_array_equal(before,after)
        self.assertEqual(audit_before,audit_after)
        changed,_=balance_layout_weights(train_texts,np.array([0,0,1,1,0,0,1,1]),.5,10.)
        self.assertFalse(np.array_equal(before,changed))

    def test_missing_cells_remain_finite(self):
        weights,audit=balance_layout_weights(['plain']*3+['a\n\nb']*3,np.array([0,0,0,1,1,1]),1.,10.)
        np.testing.assert_allclose(weights,np.ones(6))
        self.assertEqual(sum(cell['count']==0 for cell in audit['cells']),2)

    def test_cloneable_raw_text_classifier_contract(self):
        texts=['a plain short sentence.']*64+['# TITLE\n\n- one two 123!']*64
        labels=np.array([0]*64+[1]*64)
        for config in candidate_configs():
            estimator=clone(build_pipeline(config))
            self.assertTrue(is_classifier(estimator))
            estimator.fit(texts,labels)
            np.testing.assert_array_equal(estimator.classes_,[0,1])
            self.assertEqual(estimator.predict_proba(texts).shape,(128,2))
            self.assertEqual(estimator.decision_function(texts).shape,(128,))
            self.assertEqual(estimator.n_features_in_,145)
            self.assertGreater(estimator.decision_function(texts)[-1],estimator.decision_function(texts)[0])
            self.assertAlmostEqual(estimator.sample_weight_.mean(),1.)


if __name__=='__main__':
    unittest.main()
