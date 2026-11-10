"""Topic clusters from notebook 01/02: KMeans on TF-IDF+SVD over train + unlabeled + test (no labels used)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .data import load_test, load_train, load_unlabeled

CLUSTER_SEED = 77
N_CLUSTERS = 10


def assign_clusters():
    """Return dict {'train': Series(id->cluster), 'test': ...}, cluster top terms, and the fiction cluster id."""
    splits = {"train": load_train(), "unlabeled": load_unlabeled(), "test": load_test()}
    vec = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.5, stop_words="english", max_features=30_000)
    pipe = make_pipeline(TruncatedSVD(100, random_state=CLUSTER_SEED), Normalizer())
    all_texts = pd.concat([df["text"] for df in splits.values()], ignore_index=True)
    tfidf_all = vec.fit_transform(all_texts)
    emb = pipe.fit_transform(tfidf_all)
    km = KMeans(N_CLUSTERS, n_init=10, random_state=CLUSTER_SEED).fit(emb)
    labels = km.predict(emb)

    bounds = np.cumsum([len(df) for df in splits.values()])
    out = {}
    for (name, df), start, stop in zip(splits.items(), np.r_[0, bounds[:-1]], bounds):
        out[name] = pd.Series(labels[start:stop], index=df["id"].values, name="cluster")

    terms = vec.get_feature_names_out()
    top_terms = {c: ", ".join(terms[np.asarray(tfidf_all[labels == c].mean(axis=0)).ravel().argsort()[::-1][:4]])
                 for c in range(N_CLUSTERS)}
    train_share = out["train"].value_counts(normalize=True).sort_index()
    test_share = out["test"].value_counts(normalize=True).sort_index()
    fiction = int((test_share / train_share).idxmax())
    return out, top_terms, fiction
