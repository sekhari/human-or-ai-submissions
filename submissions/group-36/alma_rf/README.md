# Random Forest

From `submissions/group-number-36/alma_rf/`, create a separate environment (originally tested with Python 3.12.14):

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put the competition files in `alma_rf/data/`.

```bash
python train.py --train data/train.csv --seed 42 --model-out artifacts/validation_model.joblib --results-dir results/reproduced
python train.py --train data/train.csv --seed 42 --full --model-out artifacts/full_model.joblib --results-dir results/reproduced
python predict.py --test data/test.csv --template data/sample_submission.csv --model artifacts/full_model.joblib --out submission_rf_style.csv
```

Recorded original validation AUROC: 0.989913. During the 2026-11-08 relocation check on Jianing's Python 3.13 environment, the unchanged RF script produced 0.989905 (difference -0.000008). This cross-environment reproduction is recorded separately; the historical result is not overwritten. Candidate CSV: `alma_rf/results/submission_rf_style.csv`. To reproduce exploratory notebooks from the same directory:

```bash
python run_notebook.py Alma_Random_Forest_Exploration.ipynb
python run_notebook.py Alma_RF_Improvements.ipynb
python run_notebook.py Alma_RF_Kaggle_Export.ipynb
```
