import pandas as pd
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split

# load datasets from kaggle zip file
def load_data(zip_path):
    zip_path = Path(zip_path).expanduser()

    folder = "cornell-tech-cs-5785-combined-2026-ai-vs-human-text/"

    with zipfile.ZipFile(zip_path) as z:
        train = pd.read_csv(z.open(folder + "train.csv"))
        unlabeled = pd.read_csv(z.open(folder + "unlabeled.csv"))
        test = pd.read_csv(z.open(folder + "test.csv"))

    return train, unlabeled, test


# split the training data into training and validation sets
# for model comparison
def split_data(train):
    train, val = train_test_split(train, test_size=0.2, stratify=train["label"],random_state=42)
    return train, val

