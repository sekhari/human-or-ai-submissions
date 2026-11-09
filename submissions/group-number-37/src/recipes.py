"""Accepted inference recipes; base-model fitting is shared between them."""
import numpy as np

BASELINE_WEIGHTS = np.array([.375, .225, .15, .25])
CURRENT_WEIGHTS = np.array([.35, .234, .156, .26])


def recipe_weights(bundle, recipe="current"):
    if recipe == "current":
        weights = CURRENT_WEIGHTS
    elif recipe == "baseline":
        weights = BASELINE_WEIGHTS
    elif recipe == "saved":
        weights = bundle["ensemble"].weights
    else:
        raise ValueError(f"Unknown recipe: {recipe}")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Expected four nonnegative, finite weights summing to one.")
    return weights.copy()
