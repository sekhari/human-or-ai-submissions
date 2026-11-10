# Keyan Miao (km2299) — Team 35 contribution

Kaggle team name: `team 35`.

## Word TF-IDF baselines

`baseline_models.py` compares three word unigram/bigram TF-IDF baselines on the shared stratified 8,000 / 2,000 validation split (seed `42`):

- Logistic regression (`C=1.0`)
- Linear SVM (`C=1.0`)
- Multinomial naive Bayes (`alpha=1.0`)

TF-IDF settings: `ngram_range=(1, 2)`, `min_df=2`, `max_features=100000`, `sublinear_tf=True`.

Before splitting, the script checks for missing `id`/`text`/`label` values, duplicate IDs or texts, and non-binary labels.

## Setup and run

```bash
python -m pip install -r requirements.txt
python baseline_models.py --train path/to/train.csv --out results
# optional: 3-fold CV search over C in {0.01, 0.1, 1, 10, 100} for logistic regression and linear SVM
python baseline_models.py --train path/to/train.csv --out results --tune
```

`train.csv` comes from Kaggle and is not committed.

## Outputs (in `--out`)

- `baseline_comparison.csv` — train AUROC, validation AUROC, and fit time per model
- `*_val_scores.csv` — validation scores per model
- `validation_ids.csv` — IDs and labels of the 2,000 validation rows
- `tuning_comparison.csv` — best `C`, CV AUROC, and validation AUROC (with `--tune`)
- `experiment_config.json` — seed, split, and TF-IDF settings

This work was first committed in `jaminatore/5785-kaggle-group-35` on branch `feature/baseline-models` (commit `3f8f13d`).
