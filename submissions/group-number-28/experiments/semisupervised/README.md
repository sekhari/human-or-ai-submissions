# Learning from unlabeled examples — Q3

Owner: Gabriel (C). Also coordinates Q6 environment/README/reproduction,
collecting each author's actual commands, relevant commits, and AI-use notes.

## Q3 result (c_q3_20261109_v1): no measurable gain

| Method | Validation AUROC | Diff vs. baseline [95% CI] |
|---|---:|---:|
| Supervised only (8,000 labeled rows) | 0.9691 | — |
| **Self-training, threshold 0.95, 1 round (primary, predeclared)** | **0.9693** | +0.0001 [−0.0003, +0.0006] |
| Self-training, 0.95, round 2 / round 3 | 0.9693 / 0.9693 | +0.0002 / +0.0002, CIs include 0 |
| Self-training, 1 round, threshold 0.99 / 0.90 / 0.80 / 0.50 | 0.9691 / 0.9694 / 0.9696 / 0.9694 | all CIs include 0 |
| TF-IDF vocabulary fit on train + unlabeled (no pseudo-labels) | 0.9690 | −0.0001 [−0.0004, +0.0002] |

Training AUROC is 1.0 for every run. Figure:
`report/figures/q3_c_q3_20261109_v1_selftraining.png`.

**Setup.** Character (3,5) TF-IDF, `min_df=2`, 60,000 features, sublinear TF
(the same representation as Lin's Q5 C60), plus L2 logistic regression
(liblinear). C was chosen from {1, 3, 10, 30, 100} by 3-fold CV on the 8,000
training rows only (C=30), then frozen for every run. Seed 20261109.

**Self-training.** The baseline (the teacher) scores the 3,000 unlabeled passages.
Passages with max(p, 1−p) ≥ threshold get the predicted label (weight 1.0). The
vectorizer and classifier are refit on 8,000 labeled + pseudo-labeled rows.
Multi-round runs re-label all 3,000 rows with the latest student. Selected
pseudo-label IDs, labels and confidences are in each run's `pseudo_labels.csv`.

**Pseudo-label noise**, estimated by applying the same teacher and thresholds
to the labeled validation set:

| Threshold | Share kept | Teacher error on kept rows |
|---:|---:|---:|
| 0.99 | 26.6% | 0.6% |
| 0.95 | 53.6% | 1.1% |
| 0.90 | 66.6% | 1.9% |
| 0.80 | 79.8% | 3.1% |
| 0.50 | 100% | 9.2% |

**Leakage controls.** Only training rows and unlabeled text fit the vectorizer,
the classifier or the teacher. Validation text and labels are used only for
scoring and for the noise table above. `run_q3.py` never reads `test.csv`.

## Formatting-shortcut and shift check (c_rob_20261109_v1)

This supports Q3.3 and is offered to Q4/Q5. Normalization is NFKC, removing
zero-width characters, and collapsing all whitespace.

- No human training passage contains a blank line; 65% of AI ones do.
  **Blank-line count alone gives validation AUROC 0.844.**
- The raw-text model drops from **0.969 to 0.936** when validation formatting is
  stripped. A model trained and evaluated on normalized text scores **0.954** and
  is insensitive to formatting.
- Share of predictions with confidence ≥ 0.95 (raw model): validation 53.6%,
  unlabeled 50.5%, **test 44.5%**. The unlabeled set looks like the training
  distribution, while the test set is harder or shifted. Test is used here for
  prediction only.

## Reproduce

From the repository root, after `python3 -m human_ai.prepare_data`.
Python 3.13, `pip install -r experiments/semisupervised/requirements.txt`:

```sh
python3 -m experiments.semisupervised.run_q3 --prefix c_q3_reproduction_001       # ~8 min, 1 CPU
python3 -m experiments.semisupervised.plot_q3 --prefix c_q3_reproduction_001
python3 -m experiments.semisupervised.robustness_check --prefix c_rob_reproduction_001  # ~1 min
```

Run directories are immutable; the scripts refuse to overwrite an existing prefix.
