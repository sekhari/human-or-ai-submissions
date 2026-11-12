# Multinomial Naive Bayes models

Run all commands below from the repository root. This package uses the shared
`preprocess.py` and the existing `data/train.csv`, `data/test.csv`, and
`data/sample_submission.csv`; it does not keep separate dataset copies.

## Environment

The recorded environment is Python 3.10.8 with the versions pinned in this
folder's `requirements.txt`. To use an optional isolated environment, first
select Python 3.10.8, then run:

```sh
python3 -m venv models/multinomial_nb/.venv
source models/multinomial_nb/.venv/bin/activate
python3 -m pip install -r models/multinomial_nb/requirements.txt
```

## Run the selected model

```sh
python3 -m models.multinomial_nb.multinomial_nb_blend
```

The fixed recipe's recorded development AUROC is **0.939743**. To also reproduce
its recorded five-fold training CV before the development evaluation:

```sh
python3 -m models.multinomial_nb.multinomial_nb_blend --verify-cv
```

The recorded mean CV AUROC is **0.9357384375**. This verification refits the
shared TF-IDF independently within each fold and checks the recorded fold scores.
It takes substantially longer than fitting once.

## Generate a submission

```sh
python3 -m models.multinomial_nb.multinomial_nb_submit
```

This uses the selected ensemble. For the controlled single weighted NB model:

```sh
python3 -m models.multinomial_nb.multinomial_nb_submit --model single
```

Both commands fit only the original 8,000 training rows, check their local
validation result against the packaged reference, and write a 3,000-row
`submission.csv` with columns `id,label` in sample-submission ID order. Upload
that CSV manually; these commands do not upload to Kaggle or refit on all 10,000
labeled examples. The single model's recorded development AUROC is **0.900036**;
its Kaggle public score is unknown.

The ordinary `MultinomialNB(alpha=1)` baseline is also available:

```sh
python3 -m models.multinomial_nb.multinomial_nb
```

Its recorded development AUROC is **0.879705**. It evaluates locally and does not
write a Kaggle submission.

## Model and evaluation

The shared preprocessing uses the fixed stratified 8,000/2,000 split with seed
12, text cleaning, and up to 10,000 TF-IDF unigram/bigram features. Models share
this feature representation; the vectorizer is fitted only on the fitting rows.

The selected model combines a 1,600-learner, SAMME.R-style NB ensemble
(`alpha=1500`, learning rate 1, equal probability averaging) with a weighted
single NB (`alpha=0.3`, sample power 1, feature-prior power -96, weight cap 2).
Each component's probability logits are divided by their fitting-row population
standard deviation without centering, then blended at weights **0.75 / 0.25**
and passed through a sigmoid. This score scaling is not probability calibration.
The separate `--model single` control uses its own frozen weighted-NB recipe.

The 2,000-row development holdout was inspected across multiple experiments and
influenced retention of this recipe. Its **0.939743** AUROC is not an untouched
test estimate. The ensemble's user-reported Kaggle public score, shown in the
submission screenshot, is **0.82148**. These are different evaluations; neither
the local score nor CV guarantees a Kaggle score.

## References and outputs

`references/` contains compact reproduction records: `selected_blend/` holds
`candidate_results.csv` and `metadata.json`, `high_smoothing/metadata.json`
records the ensemble selection, and `baseline.json` holds fingerprints and
expected numerical results.
Historical root-level `outputs/` directories are not needed to run this package.

The submission command checks hashes of the shared preprocessing and input data.
Changing `preprocess.py` intentionally makes this package's submission check fail;
the check does not prevent teammates from running their own models.

Probability fingerprints use SHA-256 of probabilities rounded to 10 decimal
places. They are strict reference-environment checks, not absolute-tolerance
comparisons: tiny floating-point differences near rounding boundaries can change
the hash. On a mismatch, review the pinned versions and numerical parity before
accepting a new baseline; do not blindly replace `baseline.json`. Full numerical
parity verification compares actual prediction vectors against the original
snapshot with `rtol=0, atol=1e-12` separately from these hash checks.

New runs write only under this package:

- `outputs/experiments/<UTC timestamp>/`: local metrics, predictions, split IDs,
  and model provenance.
- `outputs/kaggle_submissions/<UTC timestamp>/`: submission CSV, validation
  evidence, split IDs, and metadata.

Generated outputs, caches, virtual environments, and `.DS_Store` are ignored by
this folder's `.gitignore`.

## Checks

```sh
python3 -m unittest discover -s models/multinomial_nb/tests -v
python3 -m unittest discover -s tests -v
```

The first command checks this package; the second checks the shared repository
code. Full-data score reproduction requires the run commands above and the
original CSV files.

## AI assistance

AI tools assisted with coding, parameter exploration, debugging, and
documentation. The authors remain responsible for inspecting the implementation,
verifying results, and describing the assistance accurately in their submission.
