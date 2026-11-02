# Human or AI? — Team code

**Team members:** TODO (names + NetIDs)
**Kaggle team name:** TODO

## Setup

Python version: 3.13

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `train.csv`, `unlabeled.csv`, and `test.csv` in `./data/` (not committed).

## Train and evaluate

```bash
python src/train.py --train data/train.csv
```

This evaluates on the fixed validation set in `splits/val_ids.txt`, writes `results/baseline.json`, then refits on all 10,000 labeled rows and saves `model.joblib`.

## Write submission.csv

```bash
python src/predict.py --test data/test.csv --out submission.csv
```

## Validation protocol and seeds

- Global seed: `20261023`
- Validation: 2,000 rows from `train.csv`, 1,000 per class, with near-duplicate groups (word-trigram TF-IDF cosine >= 0.7) kept on one side.
- `python src/make_split.py` recreates `splits/val_ids.txt`.
- `python src/shift_check.py` measures train/unlabeled/test shift from text only.

## AI assistance

TODO: match Q6.3 in the report. Claude drafted the initial split, baseline, and shift-check code; describe what the team verified or changed.
