# Q3: learning from unlabeled data

## Protocol

- Code revision: `e4250aa62795c4f7bfda5edafbbc56dbb9d67d37` (dirty: False); seed: `42`; Python: `3.11.13`; scikit-learn `1.4.2`.
- Model: word 1-2 TF-IDF + 20 standardized style features, logistic regression C=10 (`--model word_and_style`), unchanged across rows.
- Teacher: fit on the 8,000 training rows only; it scores all 3,000 unlabeled passages.
- Pseudo-labels: one round. For each fraction, the k most human-like rows get label 0 and the k most AI-like get label 1 (k = fraction x 3,000 / 2), keeping the added rows balanced.
- Student: a fresh pipeline (vocabulary, IDF and classifier) fit on training + pseudo-labeled rows, scored on the same fixed 2,000-row validation set.
- Leakage guards: validation rows are never fit or pseudo-labeled; the script checks that unlabeled IDs and normalized texts do not overlap the labeled/validation data. `test.csv` is not used.
- Estimated pseudo-label error: the teacher's error rate on validation rows that are at least as confident as each fraction's cutoffs. The true unlabeled error is unknown because those labels are hidden.
- The teacher labels 47.1% of unlabeled passages as AI at threshold 0.5.

## Results

| Method | Pseudo-labeled rows | Teacher cutoffs (human <= / AI >=) | Est. pseudo-label error | Validation AUROC | Change vs supervised (95% CI) | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| Supervised only | 0 | - | - | 0.988061 | - | 9.6 |
| Pseudo-labels, top 10% | 300 | 0.003 / 1.000 | 1.44% (209 val rows) | 0.988084 | +0.000023 [-0.00004, +0.00009] | 10.2 |
| Pseudo-labels, top 25% | 750 | 0.007 / 1.000 | 0.70% (572 val rows) | 0.988075 | +0.000014 [-0.00008, +0.00010] | 10.6 |
| Pseudo-labels, top 50% | 1,500 | 0.019 / 1.000 | 0.94% (1,067 val rows) | 0.988074 | +0.000013 [-0.00010, +0.00012] | 11.4 |
| Pseudo-labels, top 100% | 3,000 | 0.266 / 0.267 | 4.00% (1,999 val rows) | 0.988036 | -0.000025 [-0.00041, +0.00033] | 13.8 |

Fit seconds use one native thread and include vectorization. The 95% interval is a paired bootstrap (2,000 resamples of the validation rows, seed 42); an interval containing 0 means the change is within validation noise. Choosing a fraction by these scores uses the validation set.

## Reproduction

```sh
.venv/bin/python unlabeled.py --train data/train.csv --unlabeled data/unlabeled.csv --model word_and_style --out-dir outputs/q3_style
```

Local outputs: `outputs/q3_style/` (results, teacher probabilities, validation predictions; not committed).
