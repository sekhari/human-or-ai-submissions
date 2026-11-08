# Neural-network experiment

## Question and result

Can a small CPU neural network beat the team's existing style-feature gradient
booster, and does it improve on logistic regression with identical input features?

On the team's fixed 2,000-row validation set (1,000 human and 1,000 AI passages):

| Model | Training AUROC | Validation AUROC | Validation AUROC after removing line breaks |
| --- | ---: | ---: | ---: |
| Neural network, combined features, average of three seeds | 0.999960 | 0.997156 | 0.877494 |
| Existing full-layout gradient booster | 1.000000 | 0.996604 | 0.811113 |
| Logistic regression, same combined features as the network | 0.994782 | 0.992586 | 0.873544 |
| Existing tuned word TF-IDF logistic regression | 1.000000 | 0.949403 | 0.949403 |

The network is numerically ahead of the booster by **0.000552 AUROC**. The paired
95% bootstrap interval for this difference is **[-0.001693, 0.002980]**, which
includes zero. This does not establish a reliable improvement over boosting.
On the same five training CV folds, the booster averaged 0.997213 and the selected
network averaged 0.997001, reinforcing the conclusion that they are competitive.

Against logistic regression using the same combined representation, the network
gains **0.004570 AUROC**, with paired 95% interval **[0.002345, 0.007385]**.
Comparing against the existing word-only LR combines a classifier change with a
representation change; it cannot isolate the contribution of the network.

Removing line breaks is a synthetic sensitivity diagnostic. The network loses
performance too, so this experiment does not demonstrate robustness to every
formatting change or establish better performance on Kaggle's hidden test labels.
The user uploaded `submissions/neural_combined_mlp.csv` and reported a Kaggle
public AUROC of approximately **0.95**. The exact displayed score has not been
recorded, so 0.95 must not be treated as an exact measurement. This is below the
previously reported team-best public AUROC of 0.96397. The small local-validation
gain therefore did not translate into a higher reported public score. The
submission scripts themselves do not upload files.

## What the network learns

1. Word unigram/bigram TF-IDF, capped at 60,000 features, represents vocabulary.
2. TruncatedSVD compresses this into 128 numeric components, then standardizes them.
3. The team's existing 35 style features, including layout, are transformed with
   log1p and standardized, then appended to the word components.
4. Each network takes these 163 inputs through hidden layers of 128 and 64 ReLU
   units, then a sigmoid output estimates P(AI). There are 29,313 trainable
   parameters per network.
5. The final candidate averages the probabilities of networks trained with seeds
   42, 43, and 44. This is an ensemble of three small neural networks.

The neural classifier uses Adam, batch size 128, learning rate 0.001, L2 alpha=1,
and at most 200 epochs. Early stopping monitors accuracy on an internal 15% of
the fitting partition, with patience 15; it never uses the official validation
set. Thus the network's effective fitting subset is smaller than LR's.

## Experimental protocol

- The committed `splits/val_ids.csv` was reused without modification. Its SHA-256
  is `72e7c7b8840b41beca47b73fabeba8db8039b5e4c562d849d218ab5d35da603a`.
- Only the remaining 8,000 labeled passages were used for model selection.
- The initial search compared text, style, and combined representations using
  three training folds. Those results are in this directory and were not
  evaluated on the official validation set.
- Based on training CV only, the combined representation was refined using five
  stratified folds, seed 42. Settings were frozen in `refined/selected.json`
  before official validation evaluation.
- Refined search: LR C in [0.01, 0.1, 1, 10, 100]; single networks with hidden
  layers [64], [128, 64], or [256, 128] and alpha in [0.1, 1, 10]; three-seed
  averages with [64] or [128, 64] and alpha in [1, 10].
- TF-IDF, SVD, and scaling were fitted separately inside each training fold.
  LR and the neural classifiers received identical features within each fold.
- The existing booster's saved settings were also checked on those five folds.
  Its earlier tuning used these same 8,000 examples, so this is not nested CV.
- Training on the 8,000 rows and final validation followed model selection.
  No test or unlabeled texts were used to fit features or tune this experiment.
- Bootstrap comparisons used 2,000 stratified, paired resamples of the same
  validation examples. The intervals condition on this split and fitted models;
  they do not include training or model-selection uncertainty. The validation
  split had already been used for the team's earlier model comparisons.

## Reproduce

Run from the repository root using Python 3.11 and the existing virtual environment:

```bash
source .venv/bin/activate
python -m pip install -r experiments/neural_requirements.txt
```

Reproduce the complete refined experiment in a **new** directory to preserve
the original evidence:

```bash
VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m experiments.neural_comparison --profile refined --representations combined --folds 5 --output results/neural_reproduction
```

The experiment records package versions, seeds, individual CV folds, chosen
settings, warnings, runtime, validation probabilities, paired intervals, length
subgroups, training curves, and a comparison figure. Reported fit times include
feature extraction for every model; bootstrap and diagnostic scoring are excluded.

To refit the frozen neural candidate on all 10,000 labeled passages and generate
a new local submission file:

```bash
python -m experiments.neural_submission --output submissions/neural_reproduction.csv
```

The original prepared candidate is `submissions/neural_combined_mlp.csv`.
It was checked for exactly 3,000 rows, matching unique test IDs, columns
`id,label`, and finite probabilities in [0,1]. Its JSON manifest records the
selected configuration and file hashes. There is no automatic upload step.

## Sources and AI assistance

Implementation references: scikit-learn's [MLP documentation](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
and [data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Codex implemented and ran this experiment, generated the comparison artifacts,
and checked prediction validity and preprocessing isolation. The student still
needs to review the code, inspect the results, understand the modeling choices,
and record their own contributions before submitting the work. No pretrained
model, external labeled data, or commercial LLM classification API was used.
