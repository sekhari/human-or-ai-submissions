"""Logistic regression models. Both follow the team interface (src/base.py):
fit(texts, labels) -> self, predict_proba(texts) -> 1-D P(AI), get_config() -> dict.

BlockLogReg:   glue the chosen feature blocks side by side, one logreg on top.
StackedLogReg: one logreg per block, each with its own C, then a meta logreg on their outputs.
               Level 1 train rows get out-of-fold predictions (from a model that never saw them),
               so the meta model learns how much to trust each block on unseen text.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from logreg import SEED
from logreg.features import BLOCKS


def make_lr(C, seed=SEED):
    # liblinear = fast solver for sparse binary problems
    return LogisticRegression(C=C, solver='liblinear', max_iter=2000, random_state=seed)


def folds(seed=SEED):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def to_logit(p):
    # probabilities pile up near 0 and 1; logits spread them back out for a linear model
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class BlockLogReg:
    def __init__(self, blocks, C, vocab_texts=(), seed=SEED):
        self.blocks, self.C, self.vocab_texts, self.seed = list(blocks), C, list(vocab_texts), seed

    def fit(self, texts, labels):
        self.features = {name: BLOCKS[name]() for name in self.blocks}
        X = sp.hstack([f.fit_transform(texts, self.vocab_texts) for f in self.features.values()]).tocsr()
        self.lr = make_lr(self.C, self.seed).fit(X, np.asarray(labels))
        return self

    def predict_proba(self, texts):
        X = sp.hstack([f.transform(texts) for f in self.features.values()]).tocsr()
        return self.lr.predict_proba(X)[:, 1]

    def get_config(self):
        return {'blocks': self.blocks, 'C': self.C, 'seed': self.seed,
                'vocab_extra_texts': len(self.vocab_texts)}


class StackedLogReg:
    def __init__(self, block_C, vocab_texts=(), seed=SEED):
        self.block_C, self.vocab_texts, self.seed = dict(block_C), list(vocab_texts), seed

    def fit(self, texts, labels):
        y = np.asarray(labels)
        self.features, self.base, oof = {}, {}, []
        for name, C in self.block_C.items():
            self.features[name] = BLOCKS[name]()
            X = self.features[name].fit_transform(texts, self.vocab_texts)
            # honest predictions for the train rows: 5-fold rotation
            p_oof = cross_val_predict(make_lr(C, self.seed), X, y, cv=folds(self.seed),
                                      method='predict_proba')[:, 1]
            oof.append(to_logit(p_oof))
            # predictions for new rows: base model refit on all train rows
            self.base[name] = make_lr(C, self.seed).fit(X, y)

        M = np.column_stack(oof)
        # z-score so the meta weights are comparable across blocks
        self.mean, self.std = M.mean(axis=0), M.std(axis=0) + 1e-9
        self.meta = LogisticRegression(C=1.0).fit((M - self.mean) / self.std, y)
        return self

    def predict_proba(self, texts):
        M = np.column_stack([
            to_logit(self.base[name].predict_proba(self.features[name].transform(texts))[:, 1])
            for name in self.block_C
        ])
        return self.meta.predict_proba((M - self.mean) / self.std)[:, 1]

    def meta_weights(self):
        return dict(zip(self.block_C, self.meta.coef_[0].tolist()))

    def get_config(self):
        return {'block_C': self.block_C, 'meta_C': 1.0, 'seed': self.seed,
                'vocab_extra_texts': len(self.vocab_texts)}
