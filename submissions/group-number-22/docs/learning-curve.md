# Learning curve

Issue #6 uses the fixed notebook 03 baseline from PR #21. That PR is merged. Its original run lacked private files; a fresh-kernel run on this machine succeeded with verified data. Baseline validation AUROC was 0.932728, fit time 2.75 seconds, and model reload difference was zero. The private baseline result and executed copy are retained under ignored artifacts.

## Method

Use seed 42 to shuffle each class once inside the manifest training rows. Take balanced prefixes of 500, 1,000, 2,000, 5,000, and 8,000 rows. Sort selected IDs before fitting. Subsets are nested. The same 2,000 validation IDs are used throughout.

Each arm refits lowercase word 1-2 TF-IDF and C1 liblinear logistic regression. Exact baseline settings are in the notebook and result JSON. One thread. No tuning. Training AUROC uses only that arm's fitted rows. The 2,000-row repeat was fixed before measurement.

## Measured results

| Training rows | Training AUROC | Validation AUROC | Fit seconds |
|---|---:|---:|---:|
| 500 | 1.000000 | 0.856337 | 0.108 |
| 1,000 | 0.999872 | 0.884924 | 0.201 |
| 2,000 | 0.999507 | 0.905162 | 0.451 |
| 5,000 | 0.996299 | 0.921710 | 1.161 |
| 8,000 | 0.993341 | 0.932728 | 1.842 |

The complete curve run took 7.30 seconds. Maximum prediction difference was zero for the repeated 2,000-row fit and for the 8,000-row comparison with baseline 03.

- More labeled rows improved held-out AUROC across the measured sizes, by 0.076391 from 500 to 8,000.
- The final training/validation gap was 0.060613. Training remains easier than held-out prediction.
- This is one reused split. No claim about an unmeasured size, split uncertainty, or Kaggle performance.

## Reproduce

Use main after PRs #21 and #22 are merged. Restore verified data and audit components as described in docs/split-metrics.md.

```bash
uv sync --locked
uv run --locked jupyter nbconvert --execute --to notebook --inplace notebooks/03_word_baseline.ipynb
uv run --locked jupyter nbconvert --execute --to notebook --inplace notebooks/06_learning_curve.ipynb
```

Notebook 06 requires the successful private baseline handoff at `artifacts/03_word_baseline/latest.json`. It verifies the result, settings, data/split identity, and artifact hashes before fitting. Run notebook 03 locally to create that handoff. Never fabricate it from the reported score.

Safe records and a plot are under `results/06_learning_curve/`. Ordered subset ID hashes, validation ID hash, settings, source revision, notebook source hash, seed, versions, timings, all arms, repeat metrics, and artifact hashes are recorded. Subset membership and row predictions stay under ignored `artifacts/06_learning_curve/`. Completed arms are preserved there if a later arm fails; a safe failure record is written too.

PR #21 records the successful baseline execution separately. The user waived further teammate review; merge after local validation passes.
