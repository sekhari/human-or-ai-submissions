"""Small contract tests; no competition examples or historical artifacts needed."""
import hashlib
import io
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier

from .model import CHAMPION_PARAMS, DaveExtraTreesClassifier
from .train import indices_sha256, validate_submission, validate_sources
from ..text_generalization.style_search import StyleFeatures


class ChampionTests(unittest.TestCase):
    def setUp(self):
        self.texts=['plain words, simply written.']*16+['# TITLE\n\n- First item!\n- SECOND 123.']*16
        self.y=np.array([0]*16+[1]*16)
        self.params={**CHAMPION_PARAMS,'n_estimators':11}

    def test_exact_recipe_and_raw_numeric_parity(self):
        self.assertEqual(CHAMPION_PARAMS,dict(n_estimators=800,max_features=.8,min_samples_leaf=2,
            max_depth=None,criterion='gini',random_state=12,n_jobs=2,bootstrap=True,
            max_samples=.8,class_weight='balanced_subsample'))
        model=DaveExtraTreesClassifier(self.params).fit(self.texts,self.y)
        direct=ExtraTreesClassifier(**self.params).fit(StyleFeatures().transform(self.texts),self.y)
        expected=np.clip(direct.predict_proba(StyleFeatures().transform(self.texts))[:,1],1e-7,1-1e-7)
        np.testing.assert_allclose(model.predict_proba(self.texts)[:,1],expected,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(model.classes_,[0,1])
        self.assertEqual(model.n_training_rows_,32)
        self.assertEqual(model.classifier_._n_samples,32)
        self.assertEqual(model.classifier_.n_features_in_,145)

    def test_clone_serialization_and_batch_invariance(self):
        model=DaveExtraTreesClassifier(self.params).fit(self.texts,self.y)
        self.assertFalse(hasattr(clone(model),'classifier_'))
        p=model.predict_proba(self.texts)
        np.testing.assert_allclose(p,np.concatenate([model.predict_proba(self.texts[:7]),
            model.predict_proba(self.texts[7:])]),rtol=0,atol=1e-12)
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0)
        np.testing.assert_allclose(joblib.load(stream).predict_proba(self.texts),p,rtol=0,atol=1e-12)
        self.assertGreater(p[self.y==1,1].mean(),p[self.y==0,1].mean())

    def test_reject_bad_training_input(self):
        for y in (np.zeros(32),np.array([0]*16+[2]*16),self.y[:-1]):
            with self.assertRaises(ValueError):DaveExtraTreesClassifier(self.params).fit(self.texts,y)
        with self.assertRaises(ValueError):DaveExtraTreesClassifier(self.params).fit(['ok',None],[0,1])

    def test_submission_checks_ids_order_and_finite_probabilities(self):
        test=pd.DataFrame({'id':np.arange(3000),'text':['x']*3000})
        sample=pd.DataFrame({'id':np.arange(3000),'label':np.zeros(3000)})
        prediction=pd.DataFrame({'id':np.arange(3000),'label':np.full(3000,.5)})
        validate_submission(prediction,test,sample)
        for broken in (prediction.iloc[::-1],prediction.iloc[:-1],prediction.assign(text='x'),
                       prediction.assign(label=np.nan),prediction.assign(label=1.1),
                       prediction.assign(id=0)):
            with self.assertRaises(ValueError):validate_submission(broken,test,sample)

    def test_input_contract_and_split_digest(self):
        train=pd.DataFrame({'id':np.arange(10000),'text':['x']*10000,'label':[0]*5000+[1]*5000})
        test=pd.DataFrame({'id':np.arange(10000,13000),'text':['x']*3000})
        sample=pd.DataFrame({'id':test.id,'label':np.zeros(3000)})
        validate_sources(train,test,sample)
        for broken in (train.assign(label='human'),train.assign(id=0),train.assign(text=None)):
            with self.assertRaises(ValueError):validate_sources(broken,test,sample)
        self.assertEqual(indices_sha256(np.array([3,1,2])),hashlib.sha256(b'[3,1,2]').hexdigest())


if __name__=='__main__':unittest.main()
