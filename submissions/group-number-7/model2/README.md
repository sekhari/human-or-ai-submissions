# Model 2 — character TF-IDF + calibrated Linear SVM

Selected final competition model (owner: **Yuqi (Yokie) Zhang, yz3657**). This self-contained script reproduces the selected configuration: character 2–4-grams, `min_df=2`, `max_features=100000`, sublinear TF, `LinearSVC(C=1)`, and 3-fold sigmoid calibration with seed 42. Human = 0; AI = 1.

The adapter follows the public model code and configuration at [yokiezz/cornell-tech-ai-human-model2](https://github.com/yokiezz/cornell-tech-ai-human-model2). That repository attributes an earlier implementation to `JD-TB1/cornell-tech-ai-human-model2` (revision `1b1d379964c8b2ceb75e4eef531bbf90623ad176`); the code and historical search should not be represented as entirely original Group 7 work.

## Input files

Download official Kaggle files and place `train.csv`, `test.csv`, and `sample_submission.csv` together in a local directory (e.g. `/path/to/kaggle-data`). **Do not commit competition data, passwords, or hidden labels.** `train.csv` must keep the original CSV row order to reconstruct the shared split.

## Install

From the group submission directory:

```bash
python -m pip install -r requirements.txt
```

## Reproduce 8,000/2,000 evaluation

```bash
python model2/run.py evaluate --data-dir /path/to/kaggle-data --outdir outputs/model2_eval
```

Outputs: `train_predictions.csv`, `validation_predictions.csv` (ID, truth label, raw margin, P(AI), word count), and `run_info.json`.

## Generate final Kaggle probabilities

```bash
python model2/run.py submit --data-dir /path/to/kaggle-data --outdir outputs/model2_final
```

This refits on all **10,000** labeled examples and writes `outputs/model2_final/submission.csv` with exactly `id,label`, where `label` is P(AI). This script does **not** upload to Kaggle.

## Notes

- Validation membership is recreated from the **original-order** official `train.csv`, using stratified `train_test_split(test_size=0.2, random_state=42)` and preserving the original CSV order.
- Unlabeled data and Kaggle test labels are never used for supervised fitting.
- Calibration is done using stratified training-only folds; TF-IDF is inside the pipeline in each fold.
- The selected configuration was already chosen in an earlier project; our course Q1 tuning and reproduction are documented separately.
