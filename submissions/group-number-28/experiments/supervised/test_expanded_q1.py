"""Guard feature leakage and the predeclared refinement rules in expanded Q1."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from . import expanded_q1 as q


class ExpandedQ1Checks(unittest.TestCase):
    def setUp(self):
        self.plan, self.original = q.load_plans()

    def test_representation_and_calibration_boundaries(self):
        examples = {"lr":{"C":1},"nb":{"alpha":0.1},"svm":{"C":1},
                    "tree":{"max_depth":5},"forest":{"n_estimators":10},
                    "boost":{"max_iter":10},"mlp":{"hidden_layer_sizes":[64]},"lr_svd":{"C":1}}
        for family,params in examples.items():
            model = q.make_model(family,params,self.plan,self.original)
            if family=="svm":
                self.assertIsInstance(model,CalibratedClassifierCV)
                model=model.estimator
            self.assertIn("tfidf",model.named_steps)
            rep=q.representation(family,self.plan)
            self.assertEqual("svd" in model.named_steps,rep!="sparse")
            self.assertEqual("scale" in model.named_steps,rep=="scaled_dense")

    def test_refinement_uses_best_training_setting_without_mutation(self):
        cases={"lr":{"C":100},"nb":{"alpha":0.1},"svm":{"C":1},
               "tree":{"max_depth":None,"min_samples_leaf":10},
               "forest":{"n_estimators":100,"max_depth":20,"min_samples_leaf":1,"max_features":"sqrt"},
               "boost":{"max_iter":150,"learning_rate":0.1,"max_leaf_nodes":7,"l2_regularization":1},
               "mlp":{"hidden_layer_sizes":[64],"alpha":0.01,"learning_rate_init":0.001}}
        for family,best in cases.items():
            saved=deepcopy(best)
            proposals=q.refinement(family,best)
            self.assertEqual(best,saved)
            self.assertTrue(proposals)
        self.assertEqual([p["n_estimators"] for p in q.refinement("forest",cases["forest"])],[200,400])
        refined=q.refinement("lr",cases["lr"])
        self.assertAlmostEqual(refined[0]["C"],100/3)
        self.assertEqual(refined[1]["C"],300)

    def test_cached_features_refit_per_inner_fold(self):
        rows=[{"id":str(i),"text":f"record{i} " + ("human natural narrative" if i%2 else "machine formal composition"),
               "label":str(i%2)} for i in range(36)]
        seen=[]
        original_vectorizer=q.vectorizer
        original_svd=q.TruncatedSVD
        def tracking(plan):
            vec=original_vectorizer(plan)
            method=vec.fit_transform
            def fit(documents,y=None):
                seen.append(set(documents))
                return method(documents,y)
            vec.fit_transform=fit
            return vec
        def tiny_svd(**params):
            return original_svd(**{**params,"n_components":2})
        with tempfile.TemporaryDirectory() as tmp,patch.object(q,"vectorizer",tracking),patch.object(q,"TruncatedSVD",tiny_svd),patch("builtins.print"):
            cache=q.prepare_folds(rows,self.plan,self.original,Path(tmp))
        cv=StratifiedKFold(3,shuffle=True,random_state=142)
        self.assertEqual(len(cache),3)
        for docs,(fit,held) in zip(seen,cv.split(rows,[int(r["label"]) for r in rows]),strict=True):
            self.assertEqual(docs,{rows[i]["text"] for i in fit})
            self.assertTrue(docs.isdisjoint({rows[i]["text"] for i in held}))


if __name__=="__main__":
    unittest.main()
