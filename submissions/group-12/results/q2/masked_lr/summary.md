# Q2 results: masked_lr (fixed validation split, subset seed 42)

|   training rows |   train AUROC |   validation AUROC |   fit seconds |   AI share |   subset seed |
|----------------:|--------------:|-------------------:|--------------:|-----------:|--------------:|
|             500 |        1      |             0.9869 |        0.154  |        0.5 |            42 |
|            1000 |        1      |             0.9903 |        0.2505 |        0.5 |            42 |
|            2000 |        1      |             0.9926 |        0.5051 |        0.5 |            42 |
|            5000 |        0.9998 |             0.9944 |        1.2276 |        0.5 |            42 |
|            8000 |        0.9998 |             0.9949 |        1.9803 |        0.5 |            42 |

## Q2.3 draft

Using one fixed masked n-gram logistic-regression configuration, training AUROC is 1.0000 with 500 examples and 0.9998 with 8,000, while validation AUROC changes from 0.9869 to 0.9949. The training-validation gap therefore changes from 0.0131 to 0.0049. High training performance together with a lower validation score at small sample sizes indicates overfitting rather than underfitting: the model can fit the available rows, but its estimates generalize less reliably. The largest validation gain occurs between 500 and 1,000 examples. The final increment is small, indicating diminishing returns from additional labels under this representation. These conclusions apply to the fixed in-distribution split and may not fully predict performance on shifted Kaggle test data.
