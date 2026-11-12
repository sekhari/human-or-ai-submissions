> Alma’s Random Forest experiments on the shared 2,000-article validation set.

# Random Forest improvement results

Run: rf_improvement_20261107T200156_163772Z
Split: acf0a70e863835c0 (8,000 training / 2,000 validation).

## Protocol
Representation selection used 3-fold stratified CV within the training partition. Vocabulary and IDF were refitted inside each fold. Four RF settings were compared for the CV-selected representation. Validation IDs remained fixed.
Baseline validation errors informed the hypotheses; therefore holdout results are exploratory.

## Representation comparison
| Experiment | Train AUROC | Validation AUROC | Gap | Fit seconds |
| --- | ---: | ---: | ---: | ---: |
| word_count | 0.999762 | 0.944298 | 0.055464 | 3.935 |
| word_tfidf | 0.999748 | 0.932584 | 0.067164 | 2.341 |
| style_only | 0.999968 | 0.989913 | 0.010055 | 0.882 |
| count_plus_style | 0.999979 | 0.981946 | 0.018033 | 1.402 |
| binary_word | 0.999600 | 0.931258 | 0.068342 | 1.538 |
| char_tfidf | 0.999982 | 0.961610 | 0.038371 | 5.520 |

## Selection and uncertainty
Training CV selected **style_only**.
Validation AUROC change versus original word-count RF: +0.045615.
Paired bootstrap 95% percentile interval: [+0.035675, +0.055108] (1,000 resamples).
This interval does not account for all development/selection decisions and does not establish Kaggle generalization.

## Interpretation boundaries
- Preserve all candidate results, including regressions. Do not present only the best validation score.
- Style features and character features are hypotheses about complementary information; improved scores do not establish a causal mechanism.
- Length-bin AUROC and error IDs are saved separately. Failure hypotheses require reading the examples in context.
- No test-label access, external labels, or Kaggle submission was used.
- At the time of this RF experiment, teammate code was unavailable. Jianing's code and matching validation IDs are now integrated under ../../jianing_svm/; the cause of the local-to-Kaggle gap remains unproven.

## Additional observations

The strongest representation was style-only (21 features), with training CV AUROC 0.985673 +/- 0.000598 (fold standard deviation). The baseline RF settings won the four-setting CV comparison; tuning did not improve them. Counts+style scored lower than style-only.

| Validation word count | Count baseline AUROC | Style-only AUROC | Rows |
| --- | ---: | ---: | ---: |
| <=150 | 0.883867 | 0.982366 | 413 |
| 151-250 | 0.953816 | 0.992830 | 1056 |
| >250 | 0.961903 | 0.988700 | 531 |

At threshold 0.5, false positives rose from 39 to 49, while false negatives fell from 206 to 47. Threshold-specific errors must be distinguished from AUROC.

The most confident baseline errors included non-English passages, recipes, and a technical abstract. The style extractor uses English-word regexes and heuristic sentence splitting; multilingual text is an important limitation to investigate. These selected examples do not establish population-level failure patterns.

All six AUROCs were independently recomputed from saved predictions; sample alignment, split membership, CV isolation and saved notebook outputs passed verification. At the time of this validation report, no submission.csv had been generated and no Kaggle upload had occurred. Subsequent export and team-selection updates are documented in ../EXPERIMENT_HISTORY.md.
