"""Model definitions. Add new models here as functions returning sklearn estimators."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from features import StyleFeatures


def word_tfidf():
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                           max_features=200_000, sublinear_tf=True)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=300_000, sublinear_tf=True)


def tfidf_logreg(C=4.0):
    """Baseline: word + char TF-IDF -> logistic regression."""
    feats = FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])
    return Pipeline([("features", feats),
                     ("clf", LogisticRegression(C=C, max_iter=2000))])


def word_logreg(C=4.0):
    """Course reference model: word TF-IDF only."""
    return Pipeline([("features", word_tfidf()),
                     ("clf", LogisticRegression(C=C, max_iter=2000))])


def char_logreg(C=4.0):
    """Character n-grams only (should be more robust to typos/edits)."""
    return Pipeline([("features", char_tfidf()),
                     ("clf", LogisticRegression(C=C, max_iter=2000))])


def naive_bayes(alpha=0.1):
    """Multinomial NB on word 1-2 gram counts."""
    return Pipeline([("features", CountVectorizer(ngram_range=(1, 2), min_df=2,
                                                  max_features=200_000)),
                     ("clf", MultinomialNB(alpha=alpha))])


def linear_svm(C=0.5):
    """Linear SVM on word + char TF-IDF (calibrated so it outputs probabilities)."""
    feats = FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])
    return Pipeline([("features", feats),
                     ("clf", CalibratedClassifierCV(LinearSVC(C=C), cv=3))])


def style_gbm():
    """Gradient-boosted trees on ~26 stylometric features (no n-grams at all)."""
    return Pipeline([("features", StyleFeatures()),
                     ("clf", HistGradientBoostingClassifier(
                         max_iter=300, learning_rate=0.05, random_state=42))])


MODELS = {
    "tfidf_logreg": tfidf_logreg,
    "word_logreg": word_logreg,
    "char_logreg": char_logreg,
    "naive_bayes": naive_bayes,
    "linear_svm": linear_svm,
    "style_gbm": style_gbm,
}
