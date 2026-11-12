# Team 29

## Team

Kaggle team name: **Team 29**

- Ritvik Thakur (rt556)
- Bowei Huang (bh654)
- Zalan Kaman (zk239)

## Python and dependencies

Python 3.11. Install the pinned dependencies from the repository root:

```sh
cd submissions/group-number-29
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the supplied competition CSVs in `data/`. Run the remaining commands from this team folder.

## Training and validation

```sh
python style_experiment.py --train data/train.csv
```

This selects the feature combination using three grouped folds within the 8,000 training rows, evaluates on the fixed 2,000-row validation set, and refits the selected model on all 10,000 labeled rows for submission.

Other report experiments:

```sh
python experiments.py --train data/train.csv
python improvement.py --train data/train.csv
python improvement.py --train data/train.csv --tune
python unlabeled.py --train data/train.csv --unlabeled data/unlabeled.csv --model word_and_style --out-dir outputs/q3_style
python errors.py --train data/train.csv --model word_and_style --out-dir outputs/q4_style
```

Use empty output directories for fresh runs. For repeat runs, supply a new `--out-dir` and adjust subsequent model and prediction paths accordingly.

## Generate submission.csv

```sh
python predict.py \
  --model outputs/style_experiment/final_model.joblib \
  --test data/test.csv \
  --sample data/sample_submission.csv \
  --out outputs/style_experiment/submission.csv
```

Output: `outputs/style_experiment/submission.csv`, containing AI probabilities in sample-submission order.

## Seeds and configuration

Random seed: **42**. Fixed stratified validation IDs are stored in `validation_ids.json`.
The final model combines word 1–2 gram TF-IDF (`min_df=2`, `max_features=100000`, `sublinear_tf=True`) with 20 standardized writing-style measurements and logistic regression (`C=10`, `solver=liblinear`, `max_iter=2000`). Vocabulary, IDF and scaling are fitted only on training rows during evaluation. Grouped cross-validation uses 12 text clusters and three folds; the clusters approximate topics, not known source domains.

## AI-tool disclosure

We used AI tools, including ChatGPT/Codex, to assist us with code development, debugging and graph creation. They also suggested different features and model settings to test. AI assisted us as the human implementers. We reviewed the code and outputs, checked the reported scores against our results, and decided which changes to keep.
