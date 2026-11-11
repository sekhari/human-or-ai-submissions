"""Train the detector or evaluate a stratified holdout."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from model import fit_model, predict_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--model-out', type=Path, default=Path('model.joblib'))
    args = parser.parse_args()
    data = pd.read_csv(args.train)
    texts = data.text.fillna('').astype(str)
    labels = data.label.to_numpy()
    if args.validation:
        train, validation = train_test_split(range(len(data)), test_size=2000, stratify=labels, random_state=args.seed)
        bundle = fit_model(texts.iloc[train], labels[train])
        print('train AUROC', roc_auc_score(labels[train], predict_probabilities(bundle, texts.iloc[train])))
        print('validation AUROC', roc_auc_score(labels[validation], predict_probabilities(bundle, texts.iloc[validation])))
    else:
        bundle = fit_model(texts, labels)
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.model_out)
        print('saved', args.model_out)


if __name__ == '__main__':
    main()
