# Split and metric helpers

Issue #2 builds on #1 in PR #19. PRs #19 and #20 are merged. The user waived further teammate review; merge future work after local validation passes.

## Reproduce

Run notebook 01 first. Keep its private component artifact available. Then run:

```bash
uv sync --locked
RUN_SPLIT_SETUP=1 uv run --locked jupyter nbconvert --execute --to notebook --inplace notebooks/02_split_metrics.ipynb
uv run --locked python tools/check_split_helpers.py
```

The first setup run took 2.09 seconds. It made a seed-42 split with 8,000 training rows and 2,000 validation rows, balanced in each partition. All 3,000 unlabeled rows are eligible. No model was fitted. Competition AUROC is null.

The split checksum is `308f829721138d05495a6f188d0b8b0daae094787c224f7a99a2f62e2bd466e9`. The manifest stores ordered IDs, data hashes, audit identity, duplicate artifact hash, policy, and seed. Re-running setup checks equality. It never replaces a different split.

## Use in later notebooks

```python
%run ./02_split_metrics.ipynb
parts = load_partitions()
train = parts['train']
validation = parts['validation']
unlabeled = parts['unlabeled']
```

Plain `%run` defines helpers only. It does not download, fit, create a split, or write a run. The setup flag is consumed once. Loaders verify raw bytes and return rows in manifest order. IDs select rows; never feed IDs to a classifier.

The original audit points to an ignored component artifact. On another machine, run notebook 01 and copy its matching `duplicate_components.json` to the path named in the pinned audit record. Its bytes must match the pinned component hash. `load_partitions(component_path=...)` also accepts an explicit local copy. Keep the committed audit record and input manifest unchanged.

Use `class1_probabilities(model, texts)` to select class 1 by the model's class labels. Pass truth columns `id,label` and prediction columns `id,p_ai` to `evaluate_predictions`. It joins by ID and rejects missing, extra, duplicate, nonnumeric, nonfinite, or out-of-range predictions. Single-class AUROC is null. Accuracy uses threshold .5.

Use `record_run` with the command, exact config, seed, fit/eval counts, metrics, runtime, and local artifact paths. It saves a new JSON in the named result folder, with revision, dirty-worktree flag, versions, lock hash, data/split/config hashes, and artifact hashes. Call it in an exception handler with `status='failed'` and the exception type to preserve failed model runs. Keep any detailed traceback private. Full-data refits require null validation AUROC and a separate held-out confirmation reference.

## Checks and limits

Fresh kernels check prediction alignment, reversed model classes, malformed probabilities, single-class metrics, feasible grouped quotas, impossible grouped quotas, and unlabeled exclusion. The separate helper check verifies import has no writes, row order matches the manifest, and changed inputs are rejected. It uses synthetic changes and does not alter raw files.

Current data has no duplicate components. The grouped fallback uses a binary integer solver to meet exact class quotas. If it cannot solve within 60 seconds, it stops without saving. A timeout is not proof that no feasible split exists.

The split is reused for model comparison. Repeated decisions based on its score make it a selection set, not an untouched final test.
