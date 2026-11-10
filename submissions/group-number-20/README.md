# Human or AI? — Team code

**Team members:** Zhijie Wang (zw892), Simin Xu (sx333), Yangchen Jin (yj645)
**Kaggle team name:** Group Number 20 (`group-number-20`)

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

This fits the final detector on 8,000 rows, evaluates the fixed validation set in `splits/val_ids.txt`, writes `results/final.json`, then refits on all 10,000 labeled rows and saves `model.joblib`.

## Write submission.csv

```bash
python src/predict.py --test data/test.csv --out submission.csv
```

Format-free fallback:

```bash
python src/train.py --no-format --model-out model_nofmt.joblib --metrics-out results/final_nofmt.json
python src/predict.py --model model_nofmt.joblib --out submission_no_format.csv
```

Masked-word blend (`submission_v2.csv`, added after the public-score check):

```bash
python src/train.py --no-format --dist-weight 0.5 --model-out model_v2.joblib --metrics-out results/final_v2.json
python src/predict.py --model model_v2.joblib --out submission_v2.csv
```

This adds a punctuation-aware word 1-3-gram model with rare words masked (keep the 10,000 most frequent words, C=30) to the format-free blend with weight 0.5. It was chosen by leave-one-pseudo-domain-out AUROC inside the training partition, not by Kaggle scores.

## Report experiments

Run from this folder in this order:

```bash
python src/exp_models.py      # Q1
python src/exp_curve.py       # Q2 and Q3
python src/plot_curve.py      # Q2 figure
python src/exp_shift.py       # Q5
python src/exp_format.py      # Q5 blank-line follow-up
python src/train.py           # final model
python src/exp_errors.py      # Q4
python src/exp_final_lodo.py  # final pseudo-domain scores
python src/inspect_attacks.py # label-free test-shift diagnostics
python src/exp_distort.py     # masked-word model, first grid
python src/exp_distort2.py    # extended grid and blends
python src/exp_distort3.py    # no-masking control
```

Use `python src/train.py --no-format --model-out model_nofmt.joblib --metrics-out results/final_nofmt.json` before `exp_final_lodo.py`.

## Validation protocol and seeds

- Global seed: `20261023`
- Validation: 2,000 rows from `train.csv`, 1,000 per class, with near-duplicate groups (word-trigram TF-IDF cosine >= 0.7) kept on one side.
- `python src/make_split.py` recreates `splits/val_ids.txt`.
- Final models use a stylometric newline feature; every labeled blank-line row is AI, so treat this as a likely dataset artifact.
- `python src/shift_check.py` measures train/unlabeled/test shift from text only.

## AI assistance

This summary matches Q6.3 in the report.

- **Zhijie Wang:** used Claude (Claude Code in a cloud workspace) as the main developer. Claude wrote the code, ran the experiments, drafted the report and pushed commits under Zhijie's account, with Claude listed as co-author. Zhijie set the direction, made the key decisions, submitted to Kaggle, and reran both pipelines on his own laptop.
- **Yangchen Jin:** used Claude (chat) to write two check scripts, add options to `exp_errors.py`, and draft the error review and the Q4–Q5 wording. He ran every script on his own laptop, confirmed that `q4_errors.json` was unchanged, read the full texts of the reviewed errors, and edited the wording.
- **Simin Xu:** used Claude to explain the reproduction steps, run the comparisons, write `check_repro.py`, and draft `results/reproduction_simin.md`. She ran the setup, training and prediction commands herself and confirmed that the results match, with predictions within about 1e-9 of the submitted files and an identical ranking.

No AI tool was used to label test data or to predict test labels directly; all Kaggle predictions come from the local scikit-learn pipeline in `src/`.
