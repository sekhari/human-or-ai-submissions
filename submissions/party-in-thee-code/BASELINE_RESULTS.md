# Baseline result

| Model | Training AUROC | Validation AUROC | Fit time |
|---|---:|---:|---:|
| Word TF-IDF + logistic regression | 0.993344 | 0.932705 | 4.653 s |

The run uses a fixed stratified split with seed 42: 8,000 training rows and
2,000 validation rows. The split ID files are included. TF-IDF uses lowercase
word unigrams and bigrams, min_df=2 and at most 50,000 features. It is fitted
only on the training texts. Logistic regression uses C=1 and liblinear.
No hyperparameter search was done for this starting model.

The training AUROC is about 0.0606 higher than validation AUROC. That gap
suggests some overfitting, but one split does not establish the cause. The
validation score is not a Kaggle score. At threshold 0.5 there are 69 false
positives and 197 false negatives, so there are enough cases for Q4 analysis.
The notebook was executed end to end. The saved model reproduces the exported
probabilities, and the stored IDF values match document counts from the 8,000
training texts. Runtime covers the complete pipeline fit and varies by machine.

This handoff is one baseline. Remaining Q1 model comparisons, tuning, Q2 learning
curves, Q3/Q5 experiments, final Q4 analysis and the final Kaggle pipeline are
separate steps. The final team report should use the actual later results.
