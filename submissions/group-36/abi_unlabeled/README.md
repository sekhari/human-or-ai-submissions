# Unlabeled vocabulary learning in Abishek’s reconstructed stack

The controlled comparison uses Abishek’s four NB-weighted logistic text classifiers and a gradient boosting model on 67 writing-style features. A second-stage logistic regression is fitted on five-fold out-of-fold predictions. The supervised condition is reproduced with the sibling group36_evidence/rebuild.py script. Its historical leaderboard counterpart cannot be verified from the available source.

In the unlabeled condition, each text vectorizer sees the labeled training portion of its fold plus the 3,000 passages from unlabeled.csv. It then transforms only the labeled fold before fitting supervised NB ratios and regression coefficients. All four original binary vectorizers use use_idf=False. Additional text affects vocabulary membership, including document-frequency cutoffs and feature caps. The style features, estimators, C values, fold assignments and random split seed are held fixed. The fixed validation set contains 2,000 passages. No pseudo-labels are generated.

## Reproduction

Install the dependencies listed in ../group36_evidence/requirements.txt and use the project environment. From this directory:

~~~sh
python experiment.py --data /path/to/competition-data --split /path/to/seed42_stratified_8000_2000.csv
python compare.py
python experiment.py --submission --data /path/to/competition-data --split /path/to/seed42_stratified_8000_2000.csv
~~~

The validation command writes results/validation.csv and results/train.csv. The comparison aligns the saved probabilities by ID and true label and writes results/comparison.json. The submission command refits the model on all 10,000 labeled passages using the additional unlabeled corpus for vocabulary fitting. It writes results/stack_unlabeled.csv in sample-submission order.

## Fixed validation results

| Vocabulary fitting corpus | Labeled classifier training rows | Validation AUROC |
| --- | ---: | ---: |
| Training fold only | 8,000 across the training partition | 0.988825 |
| Training fold + 3,000 unlabeled passages | 8,000 across the training partition | 0.988813 |

The difference is -0.000012. A paired, class-stratified bootstrap over 2,000 validation passages (1,000 replicates, seed 42) gives a 95% percentile interval of [-0.000085, 0.000061]. This interval describes sampling variation on this fixed validation set. At the final validation fit, the word vocabulary grew from 167,947 to 218,886 entries. Additional vocabulary offered essentially no ranking benefit in this experiment. The historical stack_nbsvm_stylo.csv public score belongs to a separate submission.

[Paired validation results](results/comparison.json) · [Candidate Kaggle CSV](results/stack_unlabeled.csv)
