# Q1 results (fixed split seed 42: 8,000 train / 2,000 validation)

Settings chosen by 5-fold CV inside the 8,000 training rows; validation scored once.

## Q1.2 models

| model     | text representation                | chosen setting                       |   CV AUROC |   train AUROC |   val AUROC |   fit seconds |
|:----------|:-----------------------------------|:-------------------------------------|-----------:|--------------:|------------:|--------------:|
| tfidf_lr  | Word n-gram + char 3-5gram TF-IDF  | ngram_range=(1, 3), C=16             |     0.9486 |        1      |      0.9544 |        9.4889 |
| masked_lr | Masked function-word n-gram TF-IDF | C=64, ngram_range=(1, 5)             |     0.9956 |        0.9998 |      0.9947 |        4.1417 |
| style_gb  | 24 style features                  | learning_rate=0.3, max_leaf_nodes=63 |     0.9911 |        1      |      0.9924 |        2.7768 |

## Q1.3 grid: tfidf_lr

| setting                    |   CV AUROC (mean) |   CV AUROC (std) |
|:---------------------------|------------------:|-----------------:|
| ngram_range=(1, 3), C=16   |            0.9486 |           0.0056 |
| ngram_range=(1, 3), C=256  |            0.9474 |           0.0051 |
| ngram_range=(1, 3), C=4    |            0.9471 |           0.0053 |
| ngram_range=(1, 3), C=64   |            0.947  |           0.0054 |
| ngram_range=(1, 2), C=64   |            0.9452 |           0.0055 |
| ngram_range=(1, 2), C=16   |            0.945  |           0.0055 |
| ngram_range=(1, 2), C=4    |            0.9447 |           0.005  |
| ngram_range=(1, 2), C=256  |            0.9446 |           0.0056 |
| ngram_range=(1, 3), C=1    |            0.9429 |           0.0057 |
| ngram_range=(1, 2), C=1    |            0.94   |           0.0051 |
| ngram_range=(1, 1), C=4    |            0.9327 |           0.006  |
| ngram_range=(1, 1), C=16   |            0.9315 |           0.0063 |
| ngram_range=(1, 1), C=64   |            0.9308 |           0.0059 |
| ngram_range=(1, 1), C=256  |            0.9307 |           0.0059 |
| ngram_range=(1, 1), C=1    |            0.93   |           0.0054 |
| ngram_range=(1, 3), C=0.25 |            0.9286 |           0.0061 |
| ngram_range=(1, 2), C=0.25 |            0.9252 |           0.0054 |
| ngram_range=(1, 1), C=0.25 |            0.9169 |           0.0056 |

## Q1.3 grid: masked_lr

| setting                    |   CV AUROC (mean) |   CV AUROC (std) |
|:---------------------------|------------------:|-----------------:|
| C=64, ngram_range=(1, 5)   |            0.9956 |           0.0008 |
| C=256, ngram_range=(1, 5)  |            0.9954 |           0.001  |
| C=64, ngram_range=(1, 4)   |            0.9954 |           0.0008 |
| C=256, ngram_range=(1, 4)  |            0.9954 |           0.0008 |
| C=16, ngram_range=(1, 4)   |            0.9953 |           0.0008 |
| C=4, ngram_range=(1, 4)    |            0.9953 |           0.0008 |
| C=16, ngram_range=(1, 5)   |            0.9953 |           0.0013 |
| C=4, ngram_range=(1, 5)    |            0.9952 |           0.001  |
| C=16, ngram_range=(1, 3)   |            0.9952 |           0.0008 |
| C=256, ngram_range=(1, 3)  |            0.995  |           0.0009 |
| C=4, ngram_range=(1, 3)    |            0.995  |           0.001  |
| C=64, ngram_range=(1, 3)   |            0.9949 |           0.0007 |
| C=4, ngram_range=(1, 2)    |            0.9943 |           0.0009 |
| C=16, ngram_range=(1, 2)   |            0.9943 |           0.0007 |
| C=1, ngram_range=(1, 4)    |            0.9943 |           0.0012 |
| C=1, ngram_range=(1, 5)    |            0.9942 |           0.0013 |
| C=1, ngram_range=(1, 3)    |            0.9941 |           0.0012 |
| C=64, ngram_range=(1, 2)   |            0.9938 |           0.0007 |
| C=256, ngram_range=(1, 2)  |            0.9935 |           0.0008 |
| C=1, ngram_range=(1, 2)    |            0.9932 |           0.0014 |
| C=0.25, ngram_range=(1, 4) |            0.9925 |           0.0016 |
| C=0.25, ngram_range=(1, 5) |            0.9923 |           0.0016 |
| C=0.25, ngram_range=(1, 3) |            0.9921 |           0.0016 |
| C=0.25, ngram_range=(1, 2) |            0.99   |           0.002  |

## Q1.3 grid: style_gb

| setting                                |   CV AUROC (mean) |   CV AUROC (std) |
|:---------------------------------------|------------------:|-----------------:|
| learning_rate=0.3, max_leaf_nodes=63   |            0.9911 |           0.0025 |
| learning_rate=0.3, max_leaf_nodes=127  |            0.9908 |           0.0028 |
| learning_rate=0.1, max_leaf_nodes=63   |            0.9905 |           0.0028 |
| learning_rate=0.3, max_leaf_nodes=31   |            0.9905 |           0.0026 |
| learning_rate=0.6, max_leaf_nodes=63   |            0.9905 |           0.0032 |
| learning_rate=0.1, max_leaf_nodes=127  |            0.9905 |           0.003  |
| learning_rate=0.6, max_leaf_nodes=127  |            0.9903 |           0.0025 |
| learning_rate=0.1, max_leaf_nodes=31   |            0.9901 |           0.0034 |
| learning_rate=0.6, max_leaf_nodes=31   |            0.99   |           0.0032 |
| learning_rate=0.3, max_leaf_nodes=15   |            0.9897 |           0.0032 |
| learning_rate=0.6, max_leaf_nodes=15   |            0.9885 |           0.0029 |
| learning_rate=0.1, max_leaf_nodes=15   |            0.9877 |           0.0036 |
| learning_rate=0.03, max_leaf_nodes=127 |            0.9863 |           0.0039 |
| learning_rate=0.03, max_leaf_nodes=63  |            0.9859 |           0.004  |
| learning_rate=0.03, max_leaf_nodes=31  |            0.9841 |           0.0039 |
| learning_rate=0.03, max_leaf_nodes=15  |            0.9785 |           0.0038 |

