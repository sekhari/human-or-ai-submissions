"""three views of a model. the fixed split is what the report asks for, the other two are harder
and closer to what the test set does: leave one inferred domain out, and attacked validation text."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc

from .attacks import ATTACKS, apply


def attack_table(score_fn, texts, y, names=None, seed=0):
    """score_fn: list of str to scores."""
    rows = [{"attack": "none", "auroc": auc(y, score_fn(list(texts)))}]
    for n in names or sorted(ATTACKS):
        rows.append({"attack": n, "auroc": auc(y, score_fn(apply(n, texts, seed)))})
    return pd.DataFrame(rows)


def lodo_splits(domains, y, min_each=20):
    """yield (domain, fit_idx, held_idx) over a labeled array of domain ids."""
    for d in np.unique(domains):
        held = np.flatnonzero(domains == d)
        if min(np.sum(y[held] == 0), np.sum(y[held] == 1)) >= min_each:
            yield d, np.flatnonzero(domains != d), held


def lodo_table(fit_score, texts, y, domains):
    """fit_score(fit_idx, held_idx) -> scores for held_idx."""
    rows = []
    for d, fit, held in lodo_splits(domains, y):
        rows.append({"domain": int(d), "n": len(held), "auroc": auc(y[held], fit_score(fit, held))})
    t = pd.DataFrame(rows)
    return t, t.auroc.mean(), t.auroc.min()
