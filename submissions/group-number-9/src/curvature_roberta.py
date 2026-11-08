"""ID-safe probability ensembles and training-only curvature/RoBERTa selection."""
import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from src.blend_selection import logit_mixture

WEIGHTS = [0., .01, .025, .05, .1, .2, .35, .5]


def aligned_probability(wanted_ids, source_ids, probabilities):
    wanted, source, p = np.asarray(wanted_ids), np.asarray(source_ids), np.asarray(probabilities, dtype=float)
    if len(source) != len(p) or len(np.unique(source)) != len(source):
        raise ValueError("Duplicate or unaligned source IDs")
    if len(np.unique(wanted)) != len(wanted):
        raise ValueError("Duplicate requested IDs")
    if p.ndim != 1 or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid probabilities")
    lookup = dict(zip(source.tolist(), p.tolist()))
    if not set(wanted.tolist()).issubset(lookup):
        raise ValueError("Missing requested IDs")
    return np.array([lookup[i] for i in wanted.tolist()])


def average_probabilities(arrays):
    if not arrays or len({np.asarray(a).shape for a in arrays}) != 1:
        raise ValueError("Missing or differently shaped ensemble members")
    p = np.asarray(arrays, dtype=float)
    if p.ndim != 2 or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid ensemble probabilities")
    return p.mean(axis=0)


def blend(base, neural, weight):
    return logit_mixture(base, logit(np.clip(neural, 1e-12, 1 - 1e-12)), weight)


def select_configuration(y_inner, base_inner, inner_epochs, y_domain, base_domain, domain_epochs):
    """Freeze common epoch/weight before fixed validation; domain has 148 rows."""
    rows = []
    domain_baseline = roc_auc_score(y_domain, base_domain)
    for epoch in sorted(inner_epochs):
        if epoch not in domain_epochs:
            raise ValueError("Missing matched domain epoch")
        for weight in WEIGHTS:
            pi = blend(base_inner, inner_epochs[epoch], weight)
            pd = blend(base_domain, domain_epochs[epoch], weight)
            i, d = roc_auc_score(y_inner, pi), roc_auc_score(y_domain, pd)
            rows.append({"epoch": int(epoch), "weight": weight, "inner_auroc": i,
                         "domain_auroc": d, "selection_score": .75 * i + .25 * d,
                         "eligible": bool(d >= domain_baseline - .0005)})
    chosen = max((r for r in rows if r["eligible"]),
                 key=lambda r: (r["selection_score"], r["inner_auroc"], -r["weight"], -r["epoch"]))
    base_score = .75 * roc_auc_score(y_inner, base_inner) + .25 * domain_baseline
    return {"chosen": chosen, "training_only_gain": chosen["selection_score"] - base_score,
            "reference_inner_auroc": roc_auc_score(y_inner, base_inner),
            "reference_domain_auroc": domain_baseline, "fixed_validation_used": False,
            "candidates": rows}
