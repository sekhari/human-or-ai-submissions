"""Blend an existing mean-rank score with one additional predictor's ranks."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.curvature_roberta import aligned_probability


def blend_rank_reference(test_ids, reference_ids, reference_scores,
                         neural_ids, neural_probabilities, neural_weight=.1):
    """Preserve the reference's rank scale and the exact requested ID order.

    The reference already averages two percentile ranks. Do not re-rank it or
    interpret it as a calibrated probability before adding the neural rank.
    """
    wanted = np.asarray(test_ids)
    if not len(wanted) or len(np.unique(wanted)) != len(wanted):
        raise ValueError("Test IDs must be nonempty and unique")
    if not 0 <= neural_weight <= 1:
        raise ValueError("Neural weight must lie in [0,1]")
    if set(wanted.tolist()) != set(np.asarray(reference_ids).tolist()):
        raise ValueError("Reference must contain exactly the test IDs")
    if set(wanted.tolist()) != set(np.asarray(neural_ids).tolist()):
        raise ValueError("Neural predictions must contain exactly the test IDs")
    base = aligned_probability(wanted, reference_ids, reference_scores)
    neural = aligned_probability(wanted, neural_ids, neural_probabilities)
    neural_rank = rankdata(neural, method="average") / len(wanted)
    score = (1 - neural_weight) * base + neural_weight * neural_rank
    if not np.isfinite(score).all() or not ((score >= 0) & (score <= 1)).all():
        raise ValueError("Invalid blended scores")
    return pd.DataFrame({"id": wanted, "label": score})
