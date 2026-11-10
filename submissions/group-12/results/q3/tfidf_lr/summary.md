# Q3 results: tfidf_lr (fixed split seed 42, val = 2,000 rows)

## Q3.2 supervised vs self-training

| method                                            |   threshold |   round |   pseudo rows added |   val AUROC |   val AUROC (fiction rows) |   val error@0.5 |   fit seconds |   delta vs supervised |
|:--------------------------------------------------|------------:|--------:|--------------------:|------------:|---------------------------:|----------------:|--------------:|----------------------:|
| supervised only (8,000 labeled)                   |      nan    |       0 |                   0 |      0.9511 |                     0.8783 |          0.1125 |       58.6823 |                0      |
| self-training (pseudo-label if P>=0.6 or <=0.40)  |        0.6  |       1 |                2683 |      0.951  |                     0.881  |          0.115  |       32.9075 |               -0      |
| self-training (pseudo-label if P>=0.6 or <=0.40)  |        0.6  |       2 |                2787 |      0.9502 |                     0.8757 |          0.1135 |       33.3294 |               -0.0009 |
| self-training (pseudo-label if P>=0.8 or <=0.20)  |        0.8  |       1 |                1869 |      0.9503 |                     0.8836 |          0.1115 |       30.0475 |               -0.0008 |
| self-training (pseudo-label if P>=0.8 or <=0.20)  |        0.8  |       2 |                1976 |      0.9509 |                     0.8836 |          0.111  |       30.1531 |               -0.0001 |
| self-training (pseudo-label if P>=0.9 or <=0.10)  |        0.9  |       1 |                1186 |      0.9504 |                     0.8783 |          0.1115 |       28.0392 |               -0.0007 |
| self-training (pseudo-label if P>=0.9 or <=0.10)  |        0.9  |       2 |                1254 |      0.9505 |                     0.881  |          0.1125 |       27.1761 |               -0.0005 |
| self-training (pseudo-label if P>=0.95 or <=0.05) |        0.95 |       1 |                 714 |      0.9508 |                     0.8783 |          0.112  |       25.5821 |               -0.0002 |
| self-training (pseudo-label if P>=0.95 or <=0.05) |        0.95 |       2 |                 734 |      0.951  |                     0.8862 |          0.109  |       25.8288 |               -0.0001 |

## Pseudo-label noise proxy (teacher confidence on validation rows)

|   threshold |   val rows kept |   val share kept |   accuracy of kept rows |   unlabeled rows kept |   unlabeled AI share (pseudo) |
|------------:|----------------:|-----------------:|------------------------:|----------------------:|------------------------------:|
|        0.6  |            1802 |           0.901  |                  0.9212 |                  2683 |                        0.4379 |
|        0.8  |            1266 |           0.633  |                  0.9676 |                  1869 |                        0.4724 |
|        0.9  |             845 |           0.4225 |                  0.9882 |                  1186 |                        0.5388 |
|        0.95 |             525 |           0.2625 |                  0.9943 |                   714 |                        0.6092 |
