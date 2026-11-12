"""Shared data split, features, and ExtraTrees model for sections 4 and 5."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ..model import CHAMPION_PARAMS
from ...text_generalization.style_search import StyleFeatures


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "q4_q5"


def load_data():
    """Load the labeled data and reproduce the shared 8,000/2,000 split."""
    data = pd.read_csv(DATA_DIR / "train.csv", keep_default_na=False)

    train_indices, validation_indices = train_test_split(
        np.arange(len(data)),
        test_size=0.2,
        random_state=12,
        stratify=data["label"],
    )

    return data, train_indices, validation_indices


def extract_features(texts):
    """Convert each passage into the model's 145 style features."""
    return StyleFeatures().transform(texts)


def build_model():
    """Create the selected ExtraTrees model."""
    return ExtraTreesClassifier(**CHAMPION_PARAMS)


def auroc(model, features, labels):
    """Calculate AUROC from the model's AI probabilities."""
    probabilities = model.predict_proba(features)[:, 1]
    return roc_auc_score(labels, probabilities)
