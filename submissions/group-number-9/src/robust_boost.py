"""Train-only topic weights and paired formatting augmentation for boosting."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


BASE_PARAMS = dict(learning_rate=.1, max_iter=200, max_leaf_nodes=15,
                   l2_regularization=0.)
REG_PARAMS = dict(learning_rate=.05, max_iter=400, max_leaf_nodes=15,
                  l2_regularization=1.)
CANDIDATES = {
    "curv_bag": dict(params=BASE_PARAMS, bags=10, topic_power=0., flat_share=0.),
    "curv": dict(params=BASE_PARAMS, bags=1, topic_power=0., flat_share=0.),
    "regularized_bag": dict(params=REG_PARAMS, bags=10, topic_power=0., flat_share=0.),
    "topic_bag": dict(params=BASE_PARAMS, bags=10, topic_power=.5, flat_share=0.),
    "augment25_bag": dict(params=BASE_PARAMS, bags=10, topic_power=0., flat_share=.25),
    "augment50_bag": dict(params=BASE_PARAMS, bags=10, topic_power=0., flat_share=.5),
    "topic_augment25_bag": dict(params=BASE_PARAMS, bags=10, topic_power=.5, flat_share=.25),
    "regularized_augment25_bag": dict(params=REG_PARAMS, bags=10, topic_power=0., flat_share=.25),
}


def topic_weights(y, topics, power=.5):
    """Partly equalize topic sizes within each class, keeping class total mass fixed.

    Only fitting rows enter counts. Clipping bounds each topic's initial influence;
    normalization preserves each class's original aggregate sample weight.
    """
    y, topics = np.asarray(y), np.asarray(topics)
    weights = np.ones(len(y), dtype=float)
    for cls in np.unique(y):
        mask = y == cls
        names, counts = np.unique(topics[mask], return_counts=True)
        lookup = dict(zip(names, counts))
        w = (mask.sum() / len(names) / np.array([lookup[t] for t in topics[mask]])) ** power
        w = np.clip(w, .25, 4.)
        weights[mask] = w / w.mean()
    return weights


def training_views(x, flat, y, weight, share):
    """Each document keeps its total mass across original/flattened copies."""
    if share == 0:
        return x, y, weight
    return (np.concatenate([x, flat]), np.tile(y, 2),
            np.concatenate([(1 - share) * weight, share * weight]))


def predict_candidate(spec, x, flat, y, topics, evaluation, evaluation_flat):
    weight = topic_weights(y, topics, spec["topic_power"]) if spec["topic_power"] else None
    rng = np.random.default_rng(42)
    predictions = []
    for _ in range(spec["bags"]):
        # Bootstrap original IDs before expanding copies, never individual views.
        idx = rng.integers(0, len(x), len(x)) if spec["bags"] > 1 else np.arange(len(x))
        w = weight[idx] if weight is not None else None
        if spec["flat_share"] and w is None:
            w = np.ones(len(idx))
        xf, yf, wf = training_views(x[idx], flat[idx], y[idx], w, spec["flat_share"])
        model = HistGradientBoostingClassifier(random_state=42, early_stopping=False,
                                                **spec["params"])
        model.fit(xf, yf, sample_weight=wf)
        predictions.append(model.predict_proba(np.concatenate([evaluation, evaluation_flat]))[:, 1])
    p = np.mean(predictions, axis=0)
    return p[:len(evaluation)], p[len(evaluation):]
