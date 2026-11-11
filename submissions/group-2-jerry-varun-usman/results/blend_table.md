## Q5 Transformer variants and blends (equal-weight rank average, fixed validation)
| config             |   val_auc | has_test   |
|:-------------------|----------:|:-----------|
| base512+deberta    |    0.9946 | True       |
| base512            |    0.9943 | True       |
| deberta            |    0.9927 | True       |
| base512+deberta+lm |    0.9881 | True       |
| deberta+lm         |    0.9718 | True       |
| base512+lm         |    0.9637 | True       |
| lm                 |    0.8586 | True       |
