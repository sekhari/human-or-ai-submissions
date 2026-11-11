"""Shared TF-IDF training and batched prediction."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def fit_model(texts, labels):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=26000,
                          sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
    char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3,
                          max_features=35000, sublinear_tf=True, dtype=np.float32)
    features = hstack([word.fit_transform(texts), char.fit_transform(texts)], format='csr')
    classifier = LogisticRegression(penalty=None, solver='lbfgs', max_iter=3000)
    classifier.fit(features, labels)
    return {'word': word, 'char': char, 'classifier': classifier}


def predict_probabilities(bundle, texts):
    probabilities = np.empty(len(texts))
    for start in range(0, len(texts), 100):
        batch = texts.iloc[start:start + 100]
        features = hstack([bundle['word'].transform(batch), bundle['char'].transform(batch)], format='csr')
        probabilities[start:start + 100] = bundle['classifier'].predict_proba(features)[:, 1]
    return probabilities
