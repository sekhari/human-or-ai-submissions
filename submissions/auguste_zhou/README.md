# Team "Auguste Zhou" Kaggle Competition Submission

**Team members:** Auguste Zhou (zz285), Nishant Ray (nr487), Chenghao Guan (cg777)  
**Kaggle team name:** `auguste-zhou`

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

## Seeds and notes

- Global seed: `20261023`
- Validation: stratified 2,000-row holdout from `train.csv`, fixed across comparisons

## AI assistance

Briefly describe tools used and what you verified manually (required for Q6.3).
