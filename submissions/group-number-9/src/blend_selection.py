"""Training-only blend selection with an explicitly conservative heuristic."""
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

WEIGHTS = [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 1]


def logit_mixture(boosting, neural_logits, weight):
    b = np.asarray(boosting, dtype=np.float64)
    n = np.asarray(neural_logits, dtype=np.float64)
    if b.shape != n.shape or not np.isfinite(b).all() or not np.isfinite(n).all():
        raise ValueError("Unaligned or nonfinite component scores.")
    if not 0 <= weight <= 1 or not ((0 <= b) & (b <= 1)).all():
        raise ValueError("Invalid blend weight or probabilities.")
    if weight == 0:
        return b.copy()
    return expit((1 - weight) * logit(np.clip(b, 1e-12, 1 - 1e-12)) + weight * n)


def select_weights(labels, boosting, neural_logits, resamples=400, seed=20261105):
    """Prefer less neural weight when a paired bootstrap SE covers its loss.

    This is a training-only regularization heuristic, not a confidence bound on
    final performance. All weights and the rule must be fixed before evaluation.
    """
    y = np.asarray(labels, dtype=int)
    if set(np.unique(y)) != {0, 1} or len(y) != len(boosting):
        raise ValueError("Both binary classes and aligned rows are required.")
    predictions = [logit_mixture(boosting, neural_logits, w) for w in WEIGHTS]
    scores = np.array([roc_auc_score(y, p) for p in predictions])
    best = int(np.argmax(scores))
    rng = np.random.default_rng(seed)
    human, ai = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    boot = np.empty((resamples, len(WEIGHTS)))
    for i in range(resamples):
        sample = np.r_[rng.choice(human, len(human), replace=True), rng.choice(ai, len(ai), replace=True)]
        boot[i] = [roc_auc_score(y[sample], p[sample]) for p in predictions]
    standard_errors = (boot[:, best, None] - boot).std(axis=0, ddof=1)
    eligible = scores[best] - scores <= standard_errors + 1e-15
    conservative = int(np.flatnonzero(eligible)[0])
    return {"best_weight": WEIGHTS[best], "conservative_weight": WEIGHTS[conservative],
            "resamples": resamples, "seed": seed,
            "candidates": [{"weight": w, "inner_auroc": float(score),
                            "loss_from_best": float(scores[best] - score),
                            "paired_difference_bootstrap_se": float(se),
                            "eligible_for_conservative_rule": bool(ok)}
                           for w, score, se, ok in zip(WEIGHTS, scores, standard_errors, eligible)]}
