# Group 16

**Team members:** Brandon She (qs249), Youzi Ouyang (yo267), Ved Rajesh (vr382)
**Kaggle team name:** `Group 16`

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

Q2–Q3: run all cells of `q2_q3.ipynb`. Q4 and Q5: see `q4/README.md` and `q5/README.md`.

Topic-shift analysis (leave-topic-out CV on `train.csv` only; used to choose the final model):

```bash
python shift_eval.py
```

## Predict (writes Kaggle submission)

Final submission: equal-weight rank average of four models (word TF-IDF + LR, character n-gram TF-IDF + LR,
function-word/punctuation n-grams + LR, 25 style features + gradient boosting), retrained on all 10,000 labeled rows:

```bash
python predict_ensemble.py --train data/train.csv --test data/test.csv --out submission.csv
```

Validation AUROC 0.981 on the fixed split; Kaggle public AUROC 0.90464.

Earlier single-model submission (Model A only, Kaggle public AUROC 0.868), kept for comparison:

```bash
python predict.py --train data/train.csv --test data/test.csv --out submission_modelA.csv
```

## Seeds and notes

- Global seed: `42` (set in `common.py`)
- Validation: stratified 2,000-row holdout from `train.csv`, fixed across comparisons; ids saved in `splits/val_ids.csv`
- Hyperparameters chosen by 5-fold stratified CV on the 8,000 training rows only
- No test labels used; the final model was chosen by leave-topic-out CV on `train.csv`

## AI Assistance

We used Claude and Codex to help write and debug code and to check our work. We ran the experiments ourselves and verified the reported numbers against our saved results. No hidden test labels were retrieved.
