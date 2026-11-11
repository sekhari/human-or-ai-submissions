# Contract for independent experiments

## Shared inputs

Use the original CSVs plus the committed split. From the repository root:

```python
from human_ai.data import load_partitions
train_records, validation_records = load_partitions()
```

Or read the materialized `data/processed/train.csv` and `validation.csv` after
running the preparation command. These contain the same IDs, in canonical order.
Convert string labels to integers in your modeling code as needed.

Only training records may fit vectorizers, scalers, feature selection, calibration,
or pseudo-label teachers. In Q3 an explicitly documented feature-learning step may
use `unlabeled.csv`, but never validation text/labels for fitting. Do not silently
use test text in training. The starter subset supplies no additional labeled rows.

## Directory ownership

- `experiments/supervised/`: A, Q1 + Q5, comparisons/tuning/feature improvements.
- `experiments/generalization/`: B, Q2 + Q4, own baseline/learning curves/errors.
- `experiments/semisupervised/`: C, Q3, own baseline/unlabeled experiments.
- C coordinates Q6 and clean-environment reproduction; everyone provides their
  own commands, dependencies, relevant commits, and AI-use notes.
- Shared utilities and canonical split changes get teammate review. Independent
  model implementations need not share classes or wait on each other's code.

## One immutable directory per run

Use `results/<track>/<run_id>/` with a globally distinct run ID such as
`a_word_lr_001`. Do not overwrite a recorded run. Save:

- `config.json`: actual parameters, all seeds, training IDs/subset selection,
  preprocessing, dependency versions, split version, and executing Git commit
  (including whether there were uncommitted changes).
- `metrics.json`: train/validation AUROC, fit time with units, evaluation size,
  and relevant experiment details. Use numbers from actual runs only.
- `validation_predictions.csv`: exactly `id,label,p_ai`; `label` is true class,
  `p_ai` is predicted probability of class **1 (AI)**. One row per canonical
  validation ID. Never use the Kaggle schema here without changing the checker.
- A local model artifact if needed; model binaries are ignored by Git.

Candidate submission CSV: `submissions/<run_id>.csv`, exactly `id,label`, where
`label` is probability, not the true class. Test IDs must all appear exactly once.
Align by ID when combining predictions; never assume two files have equal order.

Validate artifacts:

```sh
python3 -m human_ai.check_predictions results/supervised/RUN_ID/validation_predictions.csv --kind validation
python3 -m human_ai.check_predictions submissions/RUN_ID.csv --kind submission
```

Commit small configurations, metrics, prediction files, tables, and figures after
checking them. Do not commit passages, serialized models, credentials, or caches.

## Controlled experiments and final model

- Q1: >=3 approaches; >=2 tuned approaches; same held-out validation IDs.
- Q2: one fixed model/hyperparameter configuration; stratified training sizes
  500/1,000/2,000/5,000/8,000 drawn only from the 8,000 training IDs. Prefer
  reproducible nested subsets. Refit feature extraction for each subset. Record
  subset IDs, train AUROC, validation AUROC, and the learning-curve figure.
- Q3: supervised and unlabeled variants share training IDs, validation IDs,
  feature/model settings except the intended semi-supervised intervention.
  Record pseudo-label counts, thresholds, and weights if used.
- Q4: record subgroup boundaries and class counts; subgroup AUROC is undefined
  if only one class is present. List >=5 FP and >=5 FN with threshold and IDs.
  Final report must analyze the finally selected model's held-out predictions.
- Q5: state a hypothesis, change one factor per ablation, and hold others fixed.
- After selection, retrain the chosen approach on all 10,000 labeled rows for
  Kaggle. Keep the 8,000-trained checkpoint/results for honest Q1/Q4 reporting;
  do not evaluate the 10,000-trained model on the old validation set as holdout.
- An optional ensemble is an extra candidate, never a prerequisite for a member
  to produce a complete standalone candidate. Choose weights without test labels.

## Dependencies and reproduction

Data preparation/checking require only Python >=3.11 standard library. Each track
records exact tested modeling dependencies in its own `requirements.txt` once
experiments exist, plus commands runnable from the repository root. C consolidates
the selected final approach into root `requirements.txt` and a documented final
training/prediction entrypoint. Those team-wide final files await selection;
Q1 now provides a runnable supervised candidate in its own track, with measured
results and a separate full-data refit. Disclose pretrained weights and verify
permission before use.

