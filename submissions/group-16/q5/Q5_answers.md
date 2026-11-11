# Q5. Open-ended improvements and insights

## Q5.1

Can reducing sensitivity to punctuation conventions improve Model A without changing its classifier? Q4 identified human passages whose AI scores were increased by list punctuation and conventional phrasing, while curly apostrophes contributed toward human predictions in an AI passage. These examples suggested that surface conventions might sometimes outweigh authorship evidence. We therefore proposed mapping curly quotes to their ASCII equivalents, preserving words and all other punctuation. A separate word-only ablation tested the stronger idea of discarding punctuation tokens entirely. The hypothesis was higher AUROC, not merely fewer errors at one threshold. This experiment follows inspection of the existing validation set.

## Q5.2

The baseline was the selected Q1 model: lowercase, punctuation-aware word TF-IDF with unigrams and bigrams, sublinear TF, min_df=5, and logistic regression with C=30 and the LIBLINEAR solver. The proposed variant maps U+2018/U+2019 to the ASCII apostrophe and U+201C/U+201D to the ASCII double quote before tokenization. The diagnostic ablation instead retains original text but extracts only word tokens, including single-character words. We used the saved Q4 split: 8,000 training rows and 2,000 validation rows. Hyperparameters were fixed. Identical five-fold stratified CV partitions (seed 42) compared configurations within training; vocabulary and IDF were refit inside every fold. Before validation scoring, the rule selected quote normalization as the best proposed variant but retained the baseline because neither variant exceeded its mean CV AUROC.

## Q5.3

| Configuration | CV AUROC | Validation AUROC | Fit time (s) | Relevant difference |
|---|---:|---:|---:|---|
| Baseline: Model A | 0.961877 | 0.966246 | 1.270 | Reference; FP 64, FN 145 |
| Proposed: quote normalization | 0.960905 | 0.965481 | 1.305 | Delta AUROC -0.000765; FP 54, FN 146 |
| Diagnostic: word-only | 0.945498 | 0.947583 | 1.064 | Delta AUROC -0.018663; FP 90, FN 155 |

Median of three TF-IDF + classifier fit times with a two-thread limit, excluding quote mapping, scoring, CV, and bootstrap. FP/FN use threshold 0.5.

## Q5.4

The results did not support improved ranking. Quote normalization changed validation AUROC by -0.000765; the descriptive paired-bootstrap 95% interval [-0.002589, +0.001044] includes zero and does not establish either improvement or equivalence. At threshold 0.5, false positives fell from 64 to 54 while false negatives rose from 145 to 146: fewer threshold errors can coexist with slightly worse AUROC. None of the ten inspected Q4 errors became correct. Removing punctuation lowered AUROC by 0.018663, suggesting those features carry useful signal in this dataset. We therefore keep Model A. These conclusions are limited to this split: Q4 already used validation examples to motivate the experiment, and the bootstrap resamples validation rows without capturing training uncertainty or domain shift. Next, we would test a punctuation perturbation stress set and add sentence-level repetition features, choosing settings using training-only CV and seeking fresh labeled evaluation data before claiming broader robustness.
