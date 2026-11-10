# Q3 results: masked_lr (fixed split seed 42, val = 2,000 rows)

## Q3.2 supervised vs self-training

| method                                            |   threshold |   round |   pseudo rows added |   val AUROC |   val AUROC (fiction rows) |   val error@0.5 |   fit seconds |   delta vs supervised |
|:--------------------------------------------------|------------:|--------:|--------------------:|------------:|---------------------------:|----------------:|--------------:|----------------------:|
| supervised only (8,000 labeled)                   |      nan    |       0 |                   0 |      0.9949 |                     0.9921 |          0.026  |       11.0861 |                0      |
| self-training (pseudo-label if P>=0.6 or <=0.40)  |        0.6  |       1 |                2906 |      0.9948 |                     0.9921 |          0.0255 |       31.5252 |               -0.0001 |
| self-training (pseudo-label if P>=0.6 or <=0.40)  |        0.6  |       2 |                2927 |      0.9948 |                     0.9921 |          0.025  |       30.0825 |               -0.0001 |
| self-training (pseudo-label if P>=0.8 or <=0.20)  |        0.8  |       1 |                2604 |      0.9948 |                     0.9921 |          0.027  |       29.3297 |               -0.0001 |
| self-training (pseudo-label if P>=0.8 or <=0.20)  |        0.8  |       2 |                2657 |      0.9947 |                     0.9921 |          0.027  |       18.3814 |               -0.0002 |
| self-training (pseudo-label if P>=0.9 or <=0.10)  |        0.9  |       1 |                2152 |      0.9948 |                     0.9921 |          0.0265 |        6.7451 |               -0.0001 |
| self-training (pseudo-label if P>=0.9 or <=0.10)  |        0.9  |       2 |                2208 |      0.9948 |                     0.9921 |          0.0265 |        7.0967 |               -0.0001 |
| self-training (pseudo-label if P>=0.95 or <=0.05) |        0.95 |       1 |                1637 |      0.9947 |                     0.9921 |          0.0275 |        7.1643 |               -0.0002 |
| self-training (pseudo-label if P>=0.95 or <=0.05) |        0.95 |       2 |                1686 |      0.9948 |                     0.9921 |          0.026  |       10.6246 |               -0.0001 |

## Pseudo-label noise proxy (teacher confidence on validation rows)

|   threshold |   val rows kept |   val share kept |   accuracy of kept rows |   unlabeled rows kept |   unlabeled AI share (pseudo) |
|------------:|----------------:|-----------------:|------------------------:|----------------------:|------------------------------:|
|        0.6  |            1941 |           0.9705 |                  0.9845 |                  2906 |                        0.4721 |
|        0.8  |            1747 |           0.8735 |                  0.9937 |                  2604 |                        0.48   |
|        0.9  |            1512 |           0.756  |                  0.9967 |                  2152 |                        0.4967 |
|        0.95 |            1180 |           0.59   |                  0.9975 |                  1637 |                        0.5412 |
