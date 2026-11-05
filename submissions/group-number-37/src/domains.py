"""rough content domains inferred without labels, fit over train, unlabeled and test text."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def infer_domains(texts, k=10, seed=3737):
    v = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.4, sublinear_tf=True, max_features=40000)
    x = v.fit_transform(texts)
    z = normalize(TruncatedSVD(60, random_state=seed).fit_transform(x))
    labels = KMeans(k, n_init=5, random_state=seed).fit_predict(z)
    names = np.array(v.get_feature_names_out())
    tops = {c: names[np.argsort(np.asarray(x[labels == c].mean(0)).ravel())[-8:]].tolist() for c in range(k)}
    return labels, tops
