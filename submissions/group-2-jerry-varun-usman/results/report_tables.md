## Q1.2 Model comparison (fixed 2,000-row validation)
| model            |   train_auc |   val_auc |   train_time_s |
|:-----------------|------------:|----------:|---------------:|
| word_tfidf_lr    |      1.0000 |    0.9423 |         7.1240 |
| char_tfidf_lr    |      0.9998 |    0.9422 |        24.8236 |
| weird_char_lr    |      0.7103 |    0.7118 |         0.0561 |
| deberta_v3_small |      0.9988 |    0.9927 |       341.3845 |

### Q1.3 grid: word_tfidf_lr (3-fold CV AUROC on training partition)
| params                                          |   cv_auc |
|:------------------------------------------------|---------:|
| {'clf__C': '0.3', 'vec__ngram_range': '(1, 1)'} |   0.8946 |
| {'clf__C': '0.3', 'vec__ngram_range': '(1, 2)'} |   0.9120 |
| {'clf__C': '1', 'vec__ngram_range': '(1, 1)'}   |   0.9082 |
| {'clf__C': '1', 'vec__ngram_range': '(1, 2)'}   |   0.9257 |
| {'clf__C': '3', 'vec__ngram_range': '(1, 1)'}   |   0.9133 |
| {'clf__C': '3', 'vec__ngram_range': '(1, 2)'}   |   0.9336 |
| {'clf__C': '10', 'vec__ngram_range': '(1, 1)'}  |   0.9122 |
| {'clf__C': '10', 'vec__ngram_range': '(1, 2)'}  |   0.9371 |
Chosen: `{'clf__C': 10, 'vec__ngram_range': [1, 2]}`

### Q1.3 grid: char_tfidf_lr (3-fold CV AUROC on training partition)
| params                                          |   cv_auc |
|:------------------------------------------------|---------:|
| {'clf__C': '0.3', 'vec__ngram_range': '(2, 4)'} |   0.9127 |
| {'clf__C': '0.3', 'vec__ngram_range': '(3, 5)'} |   0.9038 |
| {'clf__C': '1', 'vec__ngram_range': '(2, 4)'}   |   0.9277 |
| {'clf__C': '1', 'vec__ngram_range': '(3, 5)'}   |   0.9205 |
| {'clf__C': '3', 'vec__ngram_range': '(2, 4)'}   |   0.9338 |
| {'clf__C': '3', 'vec__ngram_range': '(3, 5)'}   |   0.9276 |
| {'clf__C': '10', 'vec__ngram_range': '(2, 4)'}  |   0.9345 |
| {'clf__C': '10', 'vec__ngram_range': '(3, 5)'}  |   0.9284 |
Chosen: `{'clf__C': 10, 'vec__ngram_range': [2, 4]}`

### Q1.3 grid: weird_char_lr (3-fold CV AUROC on training partition)
| params            |   cv_auc |
|:------------------|---------:|
| {'clf__C': '0.1'} |   0.7076 |
| {'clf__C': '1'}   |   0.7079 |
| {'clf__C': '10'}  |   0.7079 |
Chosen: `{'clf__C': 10}`

## Q2 Learning curve (DeBERTa-v3-small)
|      rows |    n_used |   train_auc |   val_auc |   train_time_s |
|----------:|----------:|------------:|----------:|---------------:|
|  500.0000 |  500.0000 |      0.9285 |    0.8785 |        21.8555 |
| 1000.0000 | 1000.0000 |      0.9912 |    0.9740 |        43.0431 |
| 2000.0000 | 2000.0000 |      0.9933 |    0.9797 |        86.1089 |
| 5000.0000 | 5000.0000 |      0.9980 |    0.9935 |       215.1547 |
| 8000.0000 | 8000.0000 |      0.9988 |    0.9927 |       341.3845 |

## Q3 Unlabeled data (confidence pseudo-labeling)
| method                         |   val_auc |
|:-------------------------------|----------:|
| Supervised only                |    0.9927 |
| + 2802 pseudo-labels (p>=0.95) |    0.9931 |

## Q5 Ensemble (equal-weight rank average)
| config             |   val_auc |
|:-------------------|----------:|
| deberta            |    0.9927 |
| deberta+char       |    0.9847 |
| deberta+char+weird |    0.9718 |
Final config by validation AUROC: **deberta**

## Q4.1 Train vs validation (DeBERTa)
train AUROC 0.9988 | val AUROC 0.9927

## Q4.2 Subgroups
| group           |   count |   ai_frac |   auc_deberta |   auc_final |
|:----------------|--------:|----------:|--------------:|------------:|
| 1-150 words     |     413 |    0.6223 |        0.9910 |      0.9910 |
| 151-250 words   |    1056 |    0.4138 |        0.9918 |      0.9918 |
| 251-10000 words |     531 |    0.5763 |        0.9954 |      0.9954 |
| has weird chars |      29 |    0.5172 |        1.0000 |      1.0000 |
| no weird chars  |    1971 |    0.4997 |        0.9927 |      0.9927 |

## Q4.3 Errors at threshold 0.5: 139 FP, 12 FN (top 10 each in results/errors.csv)
|   id | error   |   p_ai |   n_words | weird   |
|-----:|:--------|-------:|----------:|:--------|
| 3326 | FP      | 0.9999 |       114 | False   |
| 4667 | FP      | 0.9998 |       181 | False   |
| 5482 | FP      | 0.9998 |       175 | False   |
| 7180 | FP      | 0.9998 |       181 | False   |
| 2406 | FP      | 0.9998 |       236 | False   |
| 3364 | FP      | 0.9998 |       184 | False   |
| 6681 | FP      | 0.9998 |       126 | False   |
| 4986 | FP      | 0.9998 |       162 | False   |
| 6880 | FP      | 0.9997 |       235 | False   |
| 1360 | FP      | 0.9997 |       190 | False   |
| 7995 | FN      | 0.0016 |       156 | False   |
| 1041 | FN      | 0.0027 |       194 | False   |
| 7018 | FN      | 0.0052 |       203 | False   |
| 6331 | FN      | 0.0078 |       260 | False   |
| 7837 | FN      | 0.0083 |       108 | False   |
| 1524 | FN      | 0.0089 |       124 | False   |
| 9490 | FN      | 0.0212 |       288 | False   |
| 9505 | FN      | 0.0541 |       255 | False   |
|  424 | FN      | 0.0630 |       264 | False   |
| 3062 | FN      | 0.0892 |       170 | False   |
