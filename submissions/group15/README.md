# Group 15 — Human or AI?

**Team members:** Jingyu Zhang (jz2523), [Xinyu Zhang] ([xz2225]), Yao Xiao (yx222)
**Kaggle team name:** `group15`

## Setup

Python version: 3.12.7

```bash
pip install -r requirements.txt
```

Place the Kaggle data files (`train.csv`, `unlabeled.csv`, `test.csv`, `sample_submission.csv`) in `./data/` (not committed).

Run every command below from this folder (`submissions/group15/`).

## Train / validate

All models use the same fixed validation set: 2,000 rows from `train.csv`, stratified by label, ids saved in `splits/val_ids.txt`. Models are trained on the other 8,000 rows and scored with `common.evaluate` (train AUROC, validation AUROC, fit time). Hyperparameters are picked on validation AUROC only.

```bash
# data checks
python eda.py --data data

# Model A: word TF-IDF + logistic regression
python experiments/model_a.py            # chosen settings
python experiments/model_a.py --tune     # 32-setting grid -> experiments/model_a_tuning.txt

# Model B: character TF-IDF + linear SVM
python -m experiments.model_b_char_svm

# Optional Model B style-feature blend
python experiments/model_b_style_blend.py

# Model C: 34 text-style features + random forest (Yao Xiao)
python experiments/model_c_style.py

# Q2 learning curve -> experiments/learning_curve.txt, figures/learning_curve.png
python experiments/learning_curve.py

# Q3 unlabeled data -> experiments/unlabeled.py
# Q4 error analysis -> experiments/error_analysis.py
```

## Predict (writes Kaggle submission)

```bash
python experiments/ensemble.py --submit AC
```
This writes final submission: submission_ensemble_ac.csv

For now, Model A's submission can be made with:

```bash
python experiments/model_a.py --submit   # retrains on all 10,000 rows, writes submission_model_a.csv
```

For Model B, validation and test SVM scores can be generated with:
```bash
python -m experiments.model_b_char_svm
```
This saves:
- preds/model_b_val.npy
- preds/model_b_test.npy

If the character/style blend improves validation AUROC, a Kaggle submission can be generated with:
```bahsh
python experiments/model_b_style_blend.py
```

Model C's submission can be made with:

```bash
python experiments/model_c_style.py 
```

## Seeds and notes

- Global seed: `20261023` (`common.SEED`), used for the validation split, learning-curve subsampling and every model's `random_state`
- Validation: stratified 2,000-row holdout from `train.csv`, fixed across all comparisons
- Saved predictions: `preds/<model>_val.npy` (validation rows, in `val_ids.txt` order) and `preds/<model>_test.npy` (in `test.csv` order)

## AI assistance

- Jingyu: used Claude to explain course concepts and to draft `eda.py`, `common.py`, `experiments/model_a.py` and `experiments/learning_curve.py`. Ran every script, checked the outputs, and made the design choices (keeping case and punctuation, the fixed validation protocol, choosing C=10).
- [Xinyu]: used ChatGPT to clarify concepts including TF-IDF, character n-grams, linear SVMs, AUROC, validation, and semi-supervised learning; to discuss the experimental design for the character-level SVM model; and to review code structure and model-evaluation logic. She ran and inspected the experiments and verified the submitted implementation and outputs.
- Yao Xiao: used ChatGPT to help debug Model C’s feature-extraction and random-forest code, revise the error-analysis script, interpret experimental results, and improve the report’s wording and LaTeX formatting.[Xinyu's AI use to be added. 

No AI tool or external API was used to label or predict the competition passages.
