# Q4: generalization and error analysis

## Protocol

- Code revision: `93271af271cfb1ec255912b19f6c0aab4d339ac2` (dirty: False); seed: `42`; Python: `3.11.13`; scikit-learn `1.4.2`.
- Selected model: word 1-2 TF-IDF + 20 standardized style features, logistic regression C=10 (`--model word_and_style`), fit on the 8,000 training rows only (100,020 features).
- All numbers use the fixed 2,000-row validation set; hidden test labels are not used.

## Q4.1: training vs validation

The selected model scores **0.999980** training AUROC and **0.988061** validation AUROC (gap 0.011919).

To check whether the gap is regularization-driven, the same features were refit with other L2 strengths (smaller C = stronger regularization). Only C=10 is the selected model; the other rows are diagnostics, not a new selection.

| C | Train AUROC | Validation AUROC | Gap | Train / val accuracy at 0.5 | Train / val log loss | Confident (<=0.01 or >=0.99) train / val | Fit seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.990305 | 0.984606 | 0.005699 | 0.9679 / 0.9580 | 0.1129 / 0.1341 | 42.2% / 43.2% | 8.1 |
| 3 | 0.996676 | 0.986734 | 0.009942 | 0.9839 / 0.9645 | 0.0732 / 0.1215 | 46.0% / 47.5% | 8.3 |
| 10 | 0.999980 | 0.988061 | 0.011919 | 0.9964 / 0.9670 | 0.0345 / 0.1123 | 55.8% / 57.0% | 9.2 |
| 30 | 1.000000 | 0.988527 | 0.011473 | 1.0000 / 0.9680 | 0.0156 / 0.1113 | 69.2% / 68.0% | 10.2 |

Log loss and the confident share describe how extreme the probabilities are; AUROC depends only on ranking.

Formatting check: with every whitespace run in the validation passages collapsed to one space (no line breaks or paragraph gaps), the selected model's validation AUROC is **0.868740** (vs 0.988061 on the original text).

## Q4.2: validation AUROC by text length

Words are whitespace-separated tokens. Bins are equal-count terciles of the validation set's word counts, chosen so each AUROC uses about the same number of passages. Intervals are percentile bootstraps (2,000 resamples within each bin, seed 42).

| Bin (words) | Count | Human / AI | Validation AUROC (95% CI) | Accuracy at 0.5 |
|---|---:|---:|---:|---:|
| Short (75-175) | 667 | 319 / 348 | 0.9812 [0.9711, 0.9896] | 0.9370 |
| Medium (176-239) | 669 | 382 / 287 | 0.9976 [0.9939, 0.9997] | 0.9910 |
| Long (240-350) | 664 | 299 / 365 | 0.9870 [0.9759, 0.9954] | 0.9729 |

Median words, human vs AI: overall 204 vs 210; short 153 vs 137; medium 206 vs 209; long 270 vs 280.

## Q4.3: most confident validation errors

Threshold 0.5: a false positive (FP) is a human passage with P(AI) >= 0.5; a false negative (FN) is an AI passage with P(AI) < 0.5. At this threshold the model makes **10 FPs** (of 1,000 human) and **56 FNs** (of 1,000 AI). Listed are the five most confident of each: FPs with the highest P(AI), FNs with the lowest. Passage text is not reproduced here; failure hypotheses are written in the report after reading each passage.

| ID | Error type | P(AI) | Words |
|---|---|---:|---:|
| 8448 | FP | 0.9999 | 159 |
| 7376 | FP | 0.9691 | 229 |
| 4085 | FP | 0.8150 | 106 |
| 3291 | FP | 0.7528 | 112 |
| 8267 | FP | 0.7289 | 128 |
| 4114 | FN | 0.0011 | 300 |
| 1429 | FN | 0.0013 | 320 |
| 8444 | FN | 0.0013 | 327 |
| 5762 | FN | 0.0060 | 334 |
| 1810 | FN | 0.0115 | 277 |

## Reproduction

```sh
.venv/bin/python errors.py --train data/train.csv --model word_and_style --out-dir outputs/q4_style
```

Local outputs: `outputs/q4_style/` (gap.json, train/validation predictions; not committed).
