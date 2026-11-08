# `all_blend_roberta_rank10_10000` (public 0.99132): exact reproduction

Run commands from `submissions/group-number-9/`.

## Recreate the exact uploaded CSV

After installing the team's `requirements.txt`:

```bash
python -m experiments.all_blend_roberta_rank --verify-original --out submission.csv
```

This combines the committed `submissions/improve_all_blend.csv` with the committed
`results/all_blend_roberta_rank/roberta_test_predictions.csv`. The latter contains
only test IDs and predicted probabilities, not text, ground-truth labels or model
weights. These artifacts make exact score replay possible without distributing a
checkpoint. The script checks input hashes, IDs, ties, finite/bounded predictions
and the output checksum. Expected SHA-256:

`c398ae317ba46233e5a90e8867cbff6c242e9bf49274001bf757cef39255700a`

The original submitted CSV is also committed under `submissions/`.

## Model and weight choice

`rank10` means 10% RoBERTa percentile rank, not ten seeds or ten neural models:

```
score = 0.90 * improve_all_blend_score
      + 0.10 * scipy.stats.rankdata(roberta_probability, method="average") / 3000
```

The baseline already averages boosting and ExtraTrees ranks. It is preserved on
that scale without ranking it again or treating it as a calibrated probability.
The effective component weights are 45% boosting, 45% ExtraTrees, 10% RoBERTa.

RoBERTa-base (`FacebookAI/roberta-base`, revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`) was supervised on all 10,000 labels.
Seed 42, checkpoint after epoch 1, 625 optimizer updates. Maximum 512 tokens;
batch 4, gradient accumulation 4, evaluation batch 8, FP32, encoder LR 2e-5,
classifier LR 1e-4, AdamW, weight decay .01, gradient clipping 1, linear warmup
for 10% of the TWO-epoch scheduler horizon. Training stopped after epoch 1, but
the scheduler horizon must remain 2. No pseudo-labels or MLM adaptation were used.
See `config.json` and the existing `src/roberta_experiment.py` for full settings.

The 10% weight was predeclared as a conservative addition before the submission
was evaluated on Kaggle. It was not selected by optimizing validation or Kaggle
scores. The seed-42/epoch-1 neural component was reused from Kenny's earlier
`curvature_roberta_single_10000` submission. That earlier experiment selected the
epoch on training-only inner/topic holdouts; its 35% log-odds blend is a different
recipe from this 10% rank blend.

## Scores and limits

| Model | Fixed validation AUROC | Public Kaggle AUROC |
|---|---:|---:|
| This 10% rank blend | 0.999941 | 0.99132 |
| Teammate `improve_all_blend` | 0.9998655 | 0.99134 |
| RoBERTa alone, seed 42/epoch 1, trained on 8000 | 0.998853 | not separately submitted |
| Earlier 35% RoBERTa + old-curvature log-odds blend | 0.999911 | 0.98660 |

The newest baseline's per-example held-out scores were unavailable when this
blend was built, so it was submitted without offline validation. They were
computed afterwards: `python -m experiments.rank10_validation` refits
`improve_all_blend` on the 8,000-row training partition, combines its validation
scores with the RoBERTa (8,000-row fit) validation probabilities in
`results/curvature_roberta/single_validation_predictions.csv` using the same fixed
10% formula, and writes `validation.json` (0.999941, 95% CI 0.99987-0.99999).
The weight was not tuned on this result. Do not report 0.999911 as this blend's
validation score; it belongs to the earlier 35% log-odds blend. Validation had
been repeatedly reused, and private Kaggle scores are unknown.
The public blend is .00002 below the reference, with no observed public gain.

`protocol.json`, `verification.json`, `config.json` and the submission's `.json`
manifest are historical records written before the validation check above (hence
their null validation fields) from the original working checkout, including hashes of the
original scripts/caches. The portable script replaces those checkout-specific
dependencies with score artifacts. `kaggle_result.json` records the subsequent
user-reported public score. `earlier_curvature_blend_validation.json` supports
the component and earlier-blend scores; it is not validation of this rank blend.

## Retrain the neural component from competition data

Exact score replay above has been verified. Retraining uses the original helper
and settings, but is not guaranteed bitwise identical across hardware/backends.
It downloads the pinned public pretrained model and creates local checkpoints;
these are ignored and must not be pushed. Put Kaggle `train.csv` and `test.csv`
in local `Data/` first. To also recompute the baseline instead of using its
committed predictions, run the team's existing `python predict.py` as below.

The teammate pipeline uses Transformers 5.22.1; RoBERTa originally used 4.57.6.
Use a separate environment for this neural experiment:

```bash
python predict.py --out submissions/recomputed_all_blend.csv
python3.11 -m venv .venv-rank10
source .venv-rank10/bin/activate
python -m pip install -r experiments/rank10_requirements.txt
python -m experiments.all_blend_roberta_rank --stage retrain --baseline submissions/recomputed_all_blend.csv --out submission.csv
```

To retrain only RoBERTa and reuse the committed baseline predictions, omit the
first command and omit `--baseline submissions/recomputed_all_blend.csv` from
the last command. The retraining entrypoint runs training and scoring in fresh
processes and leaves the original committed score artifacts untouched.

## Verification

```bash
python -m unittest tests.test_rank_blend tests.test_portable_rank10
```

AI assistance: Codex/ChatGPT helped implement the neural experiments, rank-blend
script, provenance checks, tests and this reproduction package under Kenny's
direction. Kenny ran Kaggle submissions and reported the scores. This addition
contains no competition text CSVs, hidden test labels, model weights or API keys.
