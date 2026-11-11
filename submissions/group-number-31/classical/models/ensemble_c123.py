"""Extra row: weighted rank/CDF blend of C1 (tfidf_lr), C2 (stylo_gbm) and C3 (transformer_scratch).

Everything happens inside fit()'s training rows, so the fixed validation set stays untouched:
  1. each member (with the params its own train.py run chose, results/<m>/metrics.json) gets 5-fold
     out-of-fold predictions on the training rows;
  2. weights = greedy forward selection with replacement (Caruana et al., 2004) maximising the AUROC
     of the weighted mean of the members' OOF percentile ranks;
  3. each member is refit on all training rows. At prediction time a member's score is mapped through
     the CDF of its own OOF scores, so the blend is on one scale and does not depend on the batch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .base import BaseModel

HERE = Path(__file__).resolve().parents[1]


def _rank01(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


class EnsembleC123(BaseModel):
    name = "ensemble_c123"

    def __init__(self, members=("tfidf_lr", "stylo_gbm", "transformer_scratch"), folds: int = 5,
                 greedy_iters: int = 20, device: str = "auto", seed: int = 0):
        self.members, self.folds, self.greedy_iters = list(members), folds, greedy_iters
        self.device, self.seed = device, seed

    def _make(self, m, seed):
        from models import get_model_class
        params = json.loads((HERE / "results" / m / "metrics.json").read_text(encoding="utf-8"))["params"]
        cls = get_model_class(m)
        import inspect
        kw = dict(params, seed=seed)
        if "device" in inspect.signature(cls.__init__).parameters:
            kw["device"] = self.device
        return cls(**kw)

    def fit(self, texts, y, unlabeled_texts=None):
        texts, y = list(texts), np.asarray(y)
        skf = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
        oof = {m: np.zeros(len(y)) for m in self.members}
        for f, (tr, va) in enumerate(skf.split(texts, y)):
            for m in self.members:
                mdl = self._make(m, self.seed + f).fit([texts[i] for i in tr], y[tr], unlabeled_texts=unlabeled_texts)
                oof[m][va] = mdl.predict_proba([texts[i] for i in va])
                if getattr(mdl, "device_used", "cpu") != "cpu":  # report the GPU even if that member gets weight 0
                    self.device_used = mdl.device_used
            print(f"  ensemble: inner fold {f + 1}/{self.folds} done", flush=True)
        R = {m: _rank01(oof[m]) for m in self.members}
        counts = dict.fromkeys(self.members, 0)
        cur = np.zeros(len(y))
        for it in range(self.greedy_iters):
            n = it + 1
            best = max(self.members, key=lambda m: roc_auc_score(y, (cur * (n - 1) + R[m]) / n))
            counts[best] += 1
            cur = (cur * (n - 1) + R[best]) / n
        self.weights = {m: c / self.greedy_iters for m, c in counts.items()}
        self.oof_auroc = {m: float(roc_auc_score(y, oof[m])) for m in self.members}
        self.oof_auroc["blend"] = float(roc_auc_score(y, cur))
        print(f"  ensemble weights {self.weights}, inner OOF AUROC {self.oof_auroc}", flush=True)
        self.ref = {m: np.sort(oof[m]) for m in self.members}
        self.models = {m: self._make(m, self.seed).fit(texts, y, unlabeled_texts=unlabeled_texts)
                       for m in self.members if self.weights[m] > 0}
        return self

    def predict_proba(self, texts):
        texts = list(texts)
        out = np.zeros(len(texts))
        for m, mdl in self.models.items():
            out += self.weights[m] * np.searchsorted(self.ref[m], mdl.predict_proba(texts), side="right") / len(self.ref[m])
        return out

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for m, mdl in self.models.items():
            mdl.save(path / m)
        (path / "ensemble.json").write_text(json.dumps({"members": self.members, "weights": self.weights,
                                                        "oof_auroc": self.oof_auroc}), encoding="utf-8")
        np.savez(path / "ref.npz", **self.ref)

    @classmethod
    def load(cls, path):
        from models import get_model_class
        path = Path(path)
        meta = json.loads((path / "ensemble.json").read_text(encoding="utf-8"))
        e = cls(members=meta["members"])
        e.weights, e.oof_auroc = meta["weights"], meta["oof_auroc"]
        e.ref = dict(np.load(path / "ref.npz"))
        e.models = {m: get_model_class(m).load(path / m) for m in e.members if e.weights[m] > 0}
        return e
