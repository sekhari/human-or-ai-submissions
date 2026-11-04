# Applied ML competition experiments

The task is to rank AI-written passages above human-written passages. AUROC is the scoring metric. The course supplies 10,000 balanced labeled examples, 3,000 additional unlabeled examples and 3,000 test passages. The private 2,100-row leaderboard split determines final placement. The competition closes November 11, 2026 at 11:59 p.m. Eastern. Course deliverables also include a structured report and a GitHub pull request. [Competition overview](https://www.kaggle.com/competitions/cornell-tech-cs-5785-combined-2026-ai-vs-human-text/overview)

## Model selection

The notebook kept 2,000 stratified labeled rows aside for an audit. It formed topic clusters on the remaining 8,000 rows and selected the CPU recipe using three folds that hold out whole clusters. Topic clusters are approximate domains. They do not identify generators. Feature vocabularies and classifier weights were fitted within each training fold.

| model | mean | min |
| --- | --- | --- |
| pseudo_blend | 0.996193 | 0.993537 |
| blend | 0.996175 | 0.993526 |
| style | 0.994553 | 0.990994 |
| lexical | 0.952386 | 0.934549 |

The supervised blend combined lexical logistic regression and a CatBoost style classifier. The self-training variant added 900 high-confidence pseudo-labeled examples per fold. Its mean gain was below the preselected 0.001 threshold for keeping it. The supervised blend scored **0.998243 AUROC** on the untouched audit. The default CSV refits that recipe on all 10,000 labeled examples.

## Training size

The experiment kept the same topic-held-out validation rows and used nested training subsets. Only starter rows eligible for training on that fold were included.

| rows | blend_auroc |
| --- | --- |
| 1223 | 0.995841 |
| 4000 | 0.997639 |
| 6035 | 0.997917 |

## Generalization and error analysis

| length_bin | rows | auroc |
| --- | --- | --- |
| (1.999, 162.0] | 501 | 0.996334 |
| (162.0, 202.0] | 506 | 0.997913 |
| (202.0, 245.0] | 494 | 0.999319 |
| (245.0, 371.0] | 499 | 0.99869 |

The shortest quartile had the lowest audit AUROC. Whitespace normalization was also a meaningful stress test. The backup blends original style features with a second style model trained on normalized text. Its audit AUROC was **0.997631**. These stress scores describe a simulated preprocessing shift.

| condition | mean | min |
| --- | --- | --- |
| backup | 0.994109 | 0.991013 |
| backup_shifted | 0.892026 | 0.826865 |
| original | 0.994553 | 0.990994 |
| original_shifted | 0.80043 | 0.730973 |

Kaggle's hidden labels and generator metadata were unavailable. The local AUROCs cannot establish leaderboard placement. The optional DeBERTa experiment was prepared for Colab and was not run on a GPU here.

## Reproduction

Open the notebook in Colab and upload the competition ZIP. Run all cells for the CPU recipes. For the optional GPU experiment, select a GPU runtime and enable `RUN_ENCODER` after confirming the course rules on pretrained weights and cloud processing. The notebook writes prediction and experiment files to `results/`. Both provided CSVs were checked against the supplied sample for exact IDs and ordering, 3,000 rows, and finite probabilities in the required range. No prediction was uploaded to Kaggle.
