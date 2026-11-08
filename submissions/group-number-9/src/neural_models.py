"""CPU neural-network experiments; every learned transform lives in a pipeline."""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.features import build_features

SEED = 42


def text_column(frame):
    return frame.text.tolist()


def log_style(frame):
    # Counts have long tails; log1p keeps zero meaningful and reduces their scale.
    return np.log1p(build_features(frame).to_numpy(dtype=np.float64))


def representation(name, components=128):
    text = Pipeline([
        ("text", FunctionTransformer(text_column)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                   max_features=60_000, sublinear_tf=True,
                                   dtype=np.float32)),
        ("svd", TruncatedSVD(n_components=components, n_iter=5, random_state=SEED)),
        ("scale", StandardScaler()),
    ])
    style = Pipeline([
        ("style", FunctionTransformer(log_style)),
        ("scale", StandardScaler()),
    ])
    if name == "text":
        return text
    if name == "style":
        return style
    if name == "combined":
        return FeatureUnion([("text", text), ("style", style)])
    raise ValueError(f"Unknown representation: {name}")


def classifier(kind, params, seed=SEED):
    if kind == "lr":
        return LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed, **params)
    if kind == "mlp":
        params = dict(params)
        ensemble_seeds = params.pop("ensemble_seeds", None)
        if ensemble_seeds:
            return VotingClassifier(
                estimators=[(f"seed_{s}", classifier("mlp", params, s)) for s in ensemble_seeds],
                voting="soft", n_jobs=1,
            )
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        return MLPClassifier(
            solver="adam", batch_size=128, learning_rate_init=0.001,
            max_iter=200, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, tol=0.0001, random_state=seed, **params,
        )
    raise ValueError(f"Unknown classifier: {kind}")


def candidates(profile="initial"):
    # A bounded, predeclared search. Both classifiers get identical input features.
    result = [("lr", {"C": c}) for c in (0.01, 0.1, 1.0, 10.0, 100.0)]
    result += [
        ("mlp", {"hidden_layer_sizes": layers, "alpha": alpha})
        for layers in (((64,), (128, 64)) if profile == "initial" else ((64,), (128, 64), (256, 128)))
        for alpha in ((0.01, 1.0) if profile == "initial" else (0.1, 1.0, 10.0))
    ]
    if profile == "refined":
        result += [("mlp", {"hidden_layer_sizes": layers, "alpha": alpha,
                             "ensemble_seeds": [42, 43, 44]})
                   for layers in ((64,), (128, 64)) for alpha in (1.0, 10.0)]
    return result


def fitted_pipeline(name, kind, params, components=128, seed=SEED):
    return Pipeline([
        ("features", representation(name, components)),
        ("classifier", classifier(kind, params, seed)),
    ])
