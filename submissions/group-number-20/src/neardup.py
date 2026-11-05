"""Find near-duplicate passages so that they cannot straddle the train/validation split."""
import numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

def groups(texts, threshold=0.7, ngram=(3, 3)):
    X = TfidfVectorizer(lowercase=True, ngram_range=ngram, min_df=1, sublinear_tf=True).fit_transform(texts)
    S = (X @ X.T).tocoo()
    m = (S.row < S.col) & (S.data >= threshold)
    r, c, d = S.row[m], S.col[m], S.data[m]
    n = len(texts)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab, pd.DataFrame({"i": r, "j": c, "cos": d})
