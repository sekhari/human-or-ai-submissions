# <kaggle-team-name>

**Team members:** Your Name (netid1), Teammate Name (netid2)
**Kaggle team name:** `<kaggle-team-name>`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Kaggle data files (`train.csv`, `unlabeled.csv`, `test.csv`) in `./data/` (not committed).

## Train / validate

Q1 model comparison (5 models, 5-fold CV grid search, scored on the fixed validation set):

```bash
python q1_model_comparison.py --text_view raw
python q1_model_comparison.py --text_view norm
```

`raw` uses the text as given; `norm` collapses all whitespace first. Results go to `results/`. Add `--quick` for a 2-minute smoke test.

## Predict (writes Kaggle submission)

```bash
python predict.py --test data/test.csv --out submission.csv   # to be added
```

## Seeds and notes

- Global seed: `42` (set in `common.py`)
- Validation: stratified 2,000-row holdout from `train.csv`, fixed across comparisons; ids saved in `splits/val_ids.csv`
- Hyperparameters chosen by 5-fold stratified CV on the 8,000 training rows only
- AI-tool disclosure: <matches report Q6.3>

## AI Assistance
Used Claude to help check work