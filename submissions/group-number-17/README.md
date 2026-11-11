# Group Number 17: Human or AI?

**Kaggle team name:** Group Number 17

**Team members:**
- Anisha Chawla (ac3433): repo setup, shared validation split, Model B (word TF-IDF + linear SVM), Q4.3 to Q4.4 error analysis
- Zhuoyang Li (zl2342): Model A (word TF-IDF + logistic regression), Q2 learning curve, Q4.1 to Q4.2
- Phil-Ho Combatir (pc787): Model C (Naive Bayes), Model D (char n-gram SVM), Q3 pseudo-labeling, Q5 ensemble, final submission script

## Final approach

Rank-average ensemble of two linear SVMs: Model B (word unigram + bigram TF-IDF) and
Model D (character 3 to 5 gram TF-IDF). Validation AUROC on our split: **0.9497**, vs
0.9434 for the best single model (B). For Kaggle, both models are retrained on all 10,000
labeled rows. Kaggle public AUROC: 0.86497.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the Kaggle files in `data/` inside this folder (ignored by git, never committed):
`data/train.csv`, `data/unlabeled.csv`, `data/test.csv`, `data/sample_submission.csv`.

Run every command below from `submissions/group-number-17/`.

## Validation protocol

`val_ids.csv` holds 2,000 IDs (1,000 human / 1,000 AI), a stratified split of `train.csv`
made with `python make_val_split.py` (seed 42). Every experiment trains on the other
8,000 rows and scores on these 2,000. Kaggle scores were never used for tuning.

## Train / evaluate on the validation split

```bash
# Model A: logistic regression tuning, Q2 learning curve, length bins (details in LOGISTIC_REGRESSION.md)
python logistic_model.py split \
  --train data/train.csv --text-col text --label-col label --id-col id \
  --positive-label 1 --validation-size 2000 --seed 42 \
  --validation-ids val_ids.csv --manifest data/logistic_split.json
python logistic_model.py experiment \
  --train data/train.csv --text-col text --label-col label --id-col id \
  --positive-label 1 --manifest data/logistic_split.json \
  --c-values 0.1 1 10 --max-features 20000 50000 100000 \
  --train-sizes 500 1000 2000 5000 8000 \
  --output-dir data/logistic_results

# Model B: linear SVM, C sweep
python svm_model.py

# Model C: Naive Bayes grid (run before pseudo_label.py and ensemble.py)
python nb_model.py

# Q3: pseudo-labeling with unlabeled.csv
python pseudo_label.py

# Q5: ensembles of A, B, C, D on the validation split
python ensemble.py

# Q4: B + D val predictions, train vs val AUROC, length bins + figure
python export_bd_val_predictions.py        # writes bd_val_predictions.csv, bd_length_bins.png

# Q4.3: 5 most confident false positives / false negatives at threshold 0.5
python error_analysis.py bd_val_predictions.csv
```

## Write the Kaggle submission

```bash
python make_submission.py
```

Writes `submission.csv` (columns `id,label`, 3,000 rows). Trains B and D on all 10,000
labeled rows and rank-averages their test scores.

## Seeds and config

- Validation split: seed 42 (`make_val_split.py`), committed as `val_ids.csv`
- Model A: `random_state=42`, C=10, max_features=20,000, liblinear
- Models B and D: LinearSVC, C=1 (deterministic on this data)
- Model C: MultinomialNB, alpha=0.001, max_features=10,000
- Small differences (about 0.001 AUROC) can show up across scikit-learn versions

## AI assistance

- Anisha: used Claude to help write the validation split, SVM and error analysis code,
  walk through git/repo setup, and debug environment issues (a pandas install conflict and
  a .gitignore blocking val_ids.csv). Ran every experiment herself, confirmed the split was
  exactly 1,000/1,000, confirmed error_analysis.py reproduced the 0.9497, and read the
  misclassified passages herself before writing the failure reasons.
- Zhuoyang: used OpenAI Codex to help implement and debug Model A, run its experiments,
  and draft parts of the write-up. Reviewed the code himself before sharing it with the team.
- Phil: used Claude to write first versions of nb_model.py, pseudo_label.py, ensemble.py,
  make_submission.py and export_bd_val_predictions.py and to draft report text. Reviewed
  the code, reran the experiments in Google Colab, and checked the numbers against the output.
