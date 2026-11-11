"""Topic clusters for leave-cluster-out CV.

Texts are clustered by content words (TF-IDF -> SVD -> k-means) on the train and
test text together; no labels are used. Holding out one cluster at a time tests
generalization to a genre the model was not trained on, which matters here
because test is ~36% poetry while train is ~1.5%.

Assignments are built once from SEED and cached to clusters.csv (id -> cluster),
like folds.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from folds import SEED

N_CLUSTERS = 8
DEFAULT_CLUSTERS_PATH = Path(__file__).resolve().parent / "clusters.csv"


def _fit_clusters(texts: pd.Series) -> tuple[np.ndarray, list[str]]:
    vectorizer = TfidfVectorizer(stop_words="english", min_df=5, max_features=50_000,
                                 sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=100, random_state=SEED)
    Z = svd.fit_transform(X)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)  # some texts have no content words
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit(Z)
    terms = np.asarray(vectorizer.get_feature_names_out())
    centers = svd.inverse_transform(km.cluster_centers_)
    names = [", ".join(terms[np.argsort(-c)[:4]]) for c in centers]
    return km.labels_, names


def make_clusters(train: pd.DataFrame, test: pd.DataFrame,
                  path: Path = DEFAULT_CLUSTERS_PATH) -> tuple[np.ndarray, dict[int, str]]:
    """Return a cluster id for each row of `train`, and a {cluster: top terms} map."""
    if not path.exists():
        texts = pd.concat([train["text"], test["text"]], ignore_index=True)
        labels, names = _fit_clusters(texts)
        ids = pd.concat([train["id"], test["id"]], ignore_index=True)
        pd.DataFrame({"id": ids, "cluster": labels,
                      "split": ["train"] * len(train) + ["test"] * len(test),
                      "top_terms": [names[c] for c in labels]}).to_csv(path, index=False)
    clusters = pd.read_csv(path)
    train_clusters = clusters[clusters["split"] == "train"].set_index("id")
    if set(train_clusters.index) != set(train["id"]):
        raise ValueError(f"{path} does not match the ids in train; delete it to rebuild")
    names = clusters.drop_duplicates("cluster").set_index("cluster")["top_terms"].sort_index()
    return train["id"].map(train_clusters["cluster"]).to_numpy(), names.to_dict()


def split_shares(path: Path = DEFAULT_CLUSTERS_PATH) -> pd.DataFrame:
    """Share of each cluster within train and within test (columns 'train', 'test')."""
    clusters = pd.read_csv(path)
    return pd.crosstab(clusters["cluster"], clusters["split"], normalize="columns")
