**Team members:** Sydney Patel (sjp294), Teddy Taylor (tft32), Zhenghao Zhou (zz2222)
**Kaggle team name:** `group-number-34`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Kaggle data files (`train.csv`, `unlabeled.csv`, `test.csv`) in `./data/` (not committed).

## Train / validate

```bash
python train.py --train data/train.csv --seed 20261023
```

## Predict (writes Kaggle submission)

```bash
python predict.py --test data/test.csv --out submission.csv
```

## TF-IDF seeds and notes

- Global seed: `20261023`
- Validation: stratified 2,000-row holdout from `train.csv`, fixed across the TF-IDF comparisons

## GPT-2

The GPT-2 implementation, training commands, prediction command, and measured results are in [gpt2/README.md](gpt2/README.md).

GPT-2 uses seed `42` and the validation IDs in `gpt2/code/val_ids.csv`. The TF-IDF workflow uses its existing split in `splits/val_ids.json`.
