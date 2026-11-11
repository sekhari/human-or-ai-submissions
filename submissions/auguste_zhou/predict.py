"""Write Kaggle probabilities using a saved detector."""
import argparse
from pathlib import Path
from model import predict_probabilities
import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('model.joblib'))
    parser.add_argument('--sample', type=Path, default=Path('data/sample_submission.csv'))
    parser.add_argument('--out', type=Path, default=Path('submission.csv'))
    args = parser.parse_args()
    test = pd.read_csv(args.test)
    bundle = joblib.load(args.model)
    probabilities = predict_probabilities(bundle, test.text.fillna('').astype(str))
    output = pd.DataFrame({'id': test.id, 'label': probabilities})
    sample = pd.read_csv(args.sample)
    assert output.id.equals(sample.id)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False, float_format='%.12g')
    print('saved', args.out)


if __name__ == '__main__':
    main()
