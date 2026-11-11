# Word learner comparison

Issue #4 runs nine fixed candidates in notebook 04. Each candidate refits its full
word TF-IDF pipeline in three stratified folds within the original 8,000 training
rows. Seed is 42. The frozen 2,000 validation rows stay outside selection.

| Selected family | Training CV mean | Validation AUROC | Final fit seconds |
|---|---:|---:|---:|
| Logistic C4 | 0.936100 | 0.941252 | 2.785 |
| NB alpha .1 | 0.892278 | 0.903867 | 2.442 |
| SVM C1, sigmoid calibrated | 0.937294 | 0.943079 | 9.568 |

SVM leads training CV. Logistic is within .002, so the precommitted simplicity
policy chooses it for the handoff. This is a cost tradeoff. Validation did not
choose the handoff. SVM calibration uses training-only out-of-fold margins from
fresh full-pipeline fits. Sigmoid calibration does not guarantee calibrated
probabilities on new data.

Compared with the C1 baseline, logistic gains .008524 validation AUROC, with a
paired class-stratified bootstrap 95% interval [.006423, .010789]. NB loses .028861
and SVM gains .010351. The notebook keeps all nine candidates, fold scores,
settings, variation, costs, and paired intervals. Bootstrap uses 2,000 draws at
seed 42, conditional on fitted models and reused validation rows. It does not
capture training variation or selection bias.

Search took 59.539 seconds; the full notebook took 92.579 seconds. Fresh-kernel
execution passed, including model reload checks with zero probability difference.
The record is `results/04_word_comparison/20261110T183613126372Z.json`. The separate
selection JSON was saved before validation. All scores are local, not Kaggle.

Reproduce after notebook 03:

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=900 notebooks/04_word_comparison.ipynb
uv run --locked python tools/check_notebooks.py
```

The private pointer `artifacts/04_word_comparison/latest.json` names all three
selected family models and ID-aligned train/validation probabilities. It includes
configurations, CV means, hashes, and the safe result path. Notebook 05 uses the
handoff and selected word logistic arm. Notebook 07 uses the selected logistic C4
and the fixed C1 control. Verify hashes before loading. No automatic upload occurs.
Teammate review is requested in the PR.
