# Group 30 — Day 2 Results & Handoff (Q3, Q5, Q6)

> Historical daily record. Pending tasks, branch names and report paths below describe that day. For the completed submission and current report, use [README.md](README.md) and [FINAL_HANDOFF.md](FINAL_HANDOFF.md).

## Executive summary
Q3 pseudo-labeling selected 95/3,000 unlabeled passages (12 predicted human, 83 AI) and did not improve validation AUROC: 0.970763 vs 0.970822. Q5's 50/50 character + word/style blend reached AUROC 0.971276 (+0.000454) but log loss worsened to 0.329447 from 0.281667. This is a marginal ranking gain, not an unambiguous improvement.

## Data and environment
- `train.csv`: 10,000 rows, balanced 5,000/5,000.
- `unlabeled.csv`: 3,000 rows.
- `test.csv`: 3,000 rows.
- `sample_submission.csv`: 3,000 rows.
- `train_starter_2000.csv`: subset of train.csv, not extra training data.
- Stratified split: 8,000 train (4,000/class), 2,000 validation (1,000/class), seed `20261023`. Recreated split because the saved Day 1 `split.json` was not supplied.
- Environment: Python 3.13.18, NumPy 2.4.6, pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.6.1.

## Q3 results
| Method | Val AUROC | Val log loss | Fit seconds |
|---|---:|---:|---:|
| Supervised character TF-IDF LR C=3 | 0.970822 | 0.281667 | 36.910948 |
| Character LR + confidence pseudo-labels | 0.970763 | 0.281552 | 46.489525 |

Pseudo-labeling retained 95 rows: 12 human and 83 AI. AUROC decreased by 0.000059. Log loss changed slightly in a favorable direction. Only a small subset passed the strict thresholds; incorrect pseudo-labels may reinforce model errors. Conclusion: no meaningful ranking improvement under this setup.

## Q5 results
| Configuration | Val AUROC | Val log loss | Fit seconds |
|---|---:|---:|---:|
| Baseline: character TF-IDF LR C=3 | 0.970822 | 0.281667 | 36.910948 |
| 50/50 character + word/style blend | 0.971276 | 0.329447 | 37.778574 |

AUROC change: +0.000454. Runtime increase: +0.867626 seconds. The blend slightly improved ranking, directionally supporting complementary representations, but log loss worsened. This is a mixed, marginal result on one recreated split with one fixed blend weight and one compact handcrafted style-feature design. It is not an exact reproduction of the POS-based Day 1 blend.

## Q6 reproducibility
- Entrypoint: `Group30_Day2_Q3_Q5_Q6_Colab.ipynb`.
- Seed: `20261023`.
- `day2_run_record.json` records environment, data hashes, split IDs, metrics and pseudo-label counts.
- `submission_day2_blend.csv` generated with shape `(3000, 2)`. This confirms local file creation, not Kaggle upload.
- Q6.1: add actual course PR URL.
- Q6.2: add 2–3 real commit/PR links and explain genuine changes.
- Q6.3: disclose AI use honestly and state what was personally reviewed and verified.

## Remaining actions
Add PR and commit links, confirm NetIDs/contributions, review the submission CSV before upload, and do not claim Kaggle submission unless it was actually uploaded.
