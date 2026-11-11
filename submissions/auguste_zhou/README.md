# Team "Auguste Zhou" Kaggle Competition Submission

**Team members:** Auguste Zhou (zz285), Nishant Ray (nr487), Chenghao Guan (cg777)  
**Kaggle team name:** `auguste-zhou`

## Setup

Use Python 3.10 or newer. Run all commands from `submissions/auguste_zhou/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Kaggle files `train.csv`, `test.csv`, and `sample_submission.csv` in `./data/` (not committed). Only the semi-supervised experiments require `unlabeled.csv`. Training data must contain `text` and `label`; test data must contain `id` and `text`.

Dependencies are listed in `requirements.txt`. XGBoost is used only by the model-comparison experiments.

## Train / validate

Evaluate the final model recipe on an 8,000-row training partition and a stratified 2,000-row holdout:

```bash
python train.py --train data/train.csv --validation --seed 42
```

This prints training and validation AUROC. TF-IDF vocabularies and style scaling are fitted only on the training partition. Validation does not save a model.

Train on all labeled rows and save the model, vectorizers, and scaler for prediction:

```bash
python train.py --train data/train.csv --model-out model.joblib
```

The final model combines up to 26,000 word unigram/bigram TF-IDF features, 35,000 character 3–5-gram TF-IDF features, and 27 stylometric features. Style features are standardized, clipped to `[-5, 5]`, and weighted by `0.1`. The classifier is `SGDClassifier(loss="log_loss", penalty=None, max_iter=5, tol=None, random_state=42, average=False)`.

## Predict (writes Kaggle submission)

After training:

```bash
python predict.py --test data/test.csv --model model.joblib --sample data/sample_submission.csv --out submission.csv
```

This writes `id,label` columns, where `label` is the predicted probability of AI authorship. Prediction uses batches of 100 and checks that IDs match the sample submission and probabilities are finite and between 0 and 1.

## Seeds and notes

- Final classifier seed: `42`. The `--seed` argument to `train.py` controls only the validation split; it does not change the classifier seed.
- Main validation split: stratified 2,000-row holdout with seed `42`. Historical robustness checks also used split seeds `13` and `123`.
- The final model uses labeled training data only. PCA, clustering, and pseudo-labels are separate experiments and are not used for final prediction.
- `experiments/` retains the baseline, model comparisons, regularization, feature ablation, n-gram, clustering, semi-supervised, and stylometric scripts. Run these from the team folder, for example `python experiments/baseline/train_predict.py --train data/train.csv --validation --split-seed 42`.
- Historical stylometric validation can be run with `python experiments/stylometry/verify_style_multiseed.py` (seeds `13` and `123`) and `python experiments/stylometry/verify_style_group_aware.py` (group-aware split). These scripts use L-BFGS-B, whereas the final model uses SGD; their AUROCs are not interchangeable.
- `results/` contains historical aggregate metrics. Experiment scripts write new metrics to `outputs/`.

## AI assistance

ChatGPT helped draft experimental code, debug implementation/runtime issues, and interpret results. ChatGPT also assisted with ideas for feature representation, including stylometric features. We reviewed all code, selected the baseline, explored additional methods, defined the style features, and prepared the report.
