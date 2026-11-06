# group number 37

human or ai, cornell tech cs 5785. predicts the probability that a passage was written by ai.

**team:** cris nicoglu (netid: TODO), derrick cardenas (netid: TODO), alexander siladie (netid: TODO)
**kaggle team name:** Group Number 37

## setup

python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

download the competition files into `data/` at the repo root (`train.csv`, `unlabeled.csv`, `test.csv`, `train_starter_2000.csv`,
`sample_submission.csv`). they are not committed. gpt2 and distilroberta-base weights download from hugging face on first use.

## train and validate

```bash
python submissions/group-number-37/experiments/04_surprisal_cache.py      # optional cache, about 40 minutes on an m4
python submissions/group-number-37/train.py --train data/train.csv --validate --model-out model.joblib
```

`--validate` fits on the fixed 8000 rows, prints train and validation auroc, then refits on all 10000 rows for the saved model.

## write submission.csv

```bash
python submissions/group-number-37/predict.py --test data/test.csv --model model.joblib --out submission.csv
```

## seeds

global seed 3737, which also fixes the 8000/2000 stratified split. the first notebook's recipe in `src/baseline.py` keeps its seed 5785.

## layout

- `src/` data, features, models, validation helpers, final pipeline
- `experiments/` one numbered script per question, each writes to `reports/`
- `reports/FINDINGS.md` what we tried and what we learned
- `legacy/` the first notebook and its predictions
- `results/` caches and model files, not committed

## ai assistance

TODO: team to write before submission. required by the course, see report q6.3.
