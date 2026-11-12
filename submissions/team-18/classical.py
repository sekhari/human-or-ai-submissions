"""Laptop-friendly classifiers with train-only text features."""

import re
import string

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def style_features(texts):
    features = []
    for text in texts:
        words = re.findall(r"\b\w+\b", text)
        lengths = [len(word) for word in words] or [0]
        n_chars = max(1, len(text))
        features.append([
            np.log1p(len(text)),
            np.log1p(len(words)),
            np.mean(lengths),
            np.std(lengths),
            len({word.casefold() for word in words}) / max(1, len(words)),
            sum(char.isupper() for char in text) / n_chars,
            sum(char.isdigit() for char in text) / n_chars,
            sum(char in string.punctuation for char in text) / n_chars,
            text.count("\n") / n_chars,
            len(re.findall(r"[.!?]+", text)) / max(1, len(words)),
        ])
    return np.asarray(features)


def ngrams():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60_000,
                                 sublinear_tf=True, dtype=np.float32)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                 max_features=100_000, sublinear_tf=True, dtype=np.float32)),
    ])


def make_model(name, seed, jobs):
    if name == "linear":
        return Pipeline([("ngrams", ngrams()), ("classifier", LogisticRegression(max_iter=500))])

    classifier = {
        "tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=seed),
        "forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=3,
                                         n_jobs=jobs, random_state=seed),
        "boost": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07,
                                                early_stopping=False, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(256, 64), alpha=0.01,
                             max_iter=45, early_stopping=False, random_state=seed),
    }[name]
    features = FeatureUnion([
        ("latent", Pipeline([("ngrams", ngrams()),
                             ("svd", TruncatedSVD(n_components=128, random_state=seed))])),
        ("style", FunctionTransformer(style_features)),
    ])
    steps = [("features", features)]
    if name == "mlp":
        steps.append(("scale", StandardScaler()))
    return Pipeline([*steps, ("classifier", classifier)])
