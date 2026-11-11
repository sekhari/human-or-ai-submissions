# Model 1 — Word TF-IDF + Logistic Regression

**Team member:** Yaoan (Ben) Xie, yx749. Source: [Ben's verified baseline repository](https://github.com/yx749-lab/cornell-tech-cs5785-model1). This compact adapter uses the same single **untuned** Model 1 configuration (word unigrams/bigrams, `min_df=2`, 50,000 maximum features, `sublinear_tf=True`, Logistic Regression `C=1`, seed 42). It does not claim that Model 1 carried out the team's later tuning experiments.

From `submissions/group-number-7/`, install `requirements.txt` then place official competition CSV files in a private local directory; do **not** upload `train.csv`, `test.csv`, or per-row predictions to GitHub.

```bash
python -m pip install -r requirements.txt
python model1/run.py evaluate --data-dir /path/to/kaggle-data --outdir outputs/model1_eval
python model1/run.py submit --data-dir /path/to/kaggle-data --outdir outputs/model1_final
```

The evaluation reproduces a fixed stratified 8,000/2,000 split based on original Kaggle `train.csv` row order; labels are `0=Human`, `1=AI`. Outputs include both AUROCs and an ID-aligned `validation_predictions.csv` (columns `id,label,probability`). The final `submit` mode fits all 10,000 labeled examples and writes `outputs/model1_final/submission.csv` (`id,label`, where label is P(AI)).

Original teammate-reported train/validation AUROC: **0.993194 / 0.936066**. A close reproduction in a different software environment was ~0.936077; small numeric differences should be recorded honestly. Model 1 is not the selected final Kaggle submission.
