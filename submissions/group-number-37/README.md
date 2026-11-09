# group number 37

human or ai, cornell tech cs 5785. predicts the probability that a passage was written by ai.

**team:** cris nicoglu (netid: TODO), derrick cardenas (netid: TODO), alexander siladie (netid: TODO)
**kaggle team name:** Group Number 37

## four-method robustness candidate

A separate T4-compatible candidate and validation report are available in [ROBUST_ENSEMBLE.md](ROBUST_ENSEMBLE.md).
Open [Four_Method_Robust_Colab.ipynb](Four_Method_Robust_Colab.ipynb) in Colab and upload the original competition ZIP.
It combines the original reference, dual GPT-2 feature models, DistilRoBERTa, and DeBERTa with frozen training-fitted calibration.
The original commands below remain the team control. No Kaggle submission is performed by the new notebook.

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
the model is a rank blend of the reference model, surprisal features and a distilroberta encoder, weighted 0.55, 0.25, 0.2
(`src/pipeline.py`, chosen in `experiments/21_nested_three_way.py`). the encoder is saved to `--encoder-dir`, default `encoder/`.

each encoder training takes about 20 minutes on an m4. `--validate` trains it twice, so skip the flag to train once. keep the
machine awake during long runs.

## write submission.csv

```bash
python submissions/group-number-37/predict.py --test data/test.csv --model model.joblib --out submission.csv
```

run this from the same folder as training. the saved model finds its encoder through the relative `--encoder-dir` path.

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
