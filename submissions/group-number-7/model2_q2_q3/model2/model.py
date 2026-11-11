"""Keep vocabulary fitting inside every training/calibration fold."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_pipeline(config):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=config["analyzer"],
            ngram_range=tuple(config["ngram_range"]),
            min_df=config["min_df"],
            max_features=config["max_features"],
            lowercase=config["lowercase"],
            sublinear_tf=config["sublinear_tf"],
        )),
        ("svm", LinearSVC(C=config["C"], dual="auto",
                          max_iter=config["max_iter"], random_state=config["seed"])),
    ])


def build_calibrated_model(config):
    folds = StratifiedKFold(n_splits=config["calibration_folds"],
                           shuffle=True, random_state=config["seed"])
    # Out-of-fold margins fit the sigmoid; the final pipeline fits all supplied
    # training rows. The outer validation set must never be supplied to fit().
    return CalibratedClassifierCV(build_pipeline(config), method="sigmoid",
                                  cv=folds, ensemble=False, n_jobs=1)
