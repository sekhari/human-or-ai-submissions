from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import _data_file, dave_145_stylization, dave_145_stylization_full_data, preprocess_data, process_full_data, unlabel_data
from scorer import create_kaggle_submission, evaluate_model




def extra_trees(X_train, y_train):
    print("\nExtra Trees Classifier")

    # model_et = ExtraTreesClassifier(
    #     n_estimators=100,
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     random_state=12,
    #     n_jobs=-1
    # )
    model_et = ExtraTreesClassifier(
        n_estimators=800,
        max_features=0.8,
        min_samples_leaf=2,
        max_depth=None,
        random_state=12,
        n_jobs=2,
        bootstrap=True,
        max_samples=0.8,
        class_weight='balanced_subsample'
    )
    
    model_et.fit(X_train, y_train)

    return model_et

start = perf_counter()
X_train, y_train, X_validation, y_validation, X_test = preprocess_data(featurization=dave_145_stylization)
print(f"Feature extraction: {perf_counter() - start:.2f} seconds")


start = perf_counter()
model = extra_trees(X_train, y_train)
print(f"Model training: {perf_counter() - start:.2f} seconds")

evaluate_model(model, X_validation, y_validation)



# X_train, y_train, X_test = process_full_data(featurization=dave_145_stylization_full_data)

# model_full = extra_trees(X_train, y_train)

# create_kaggle_submission(model_full, X_test, "submissions/submission_will_extra_trees_dave_stylization_minlm.csv")