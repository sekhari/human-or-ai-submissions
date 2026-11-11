# Data audit

Notebook 01 checks the five files before a split is made. Run it from the repo root:

```bash
uv sync --locked
uv run --locked jupyter nbconvert --execute --to notebook --inplace notebooks/01_data_setup.ipynb
```

Keep the Kaggle token in private `.env`. The notebook loads it without printing it.
KaggleHub may reuse its cache. `download_checked_utc` records when that call and byte checks ran, not the original server upload date.

The first run checked 10,000 training rows, 3,000 unlabeled rows, 3,000 test rows, 2,000 starter rows, and 3,000 sample rows. Training has 5,000 rows per class. The starter matches training exactly. Sample and test IDs agree.

The duplicate audit checked 108,271 candidate pairs. It found no exact-text pairs and no normalized five-word-shingle pairs at Jaccard >= .85 across train, unlabeled, and test, including within-file pairs. This does not rule out paraphrases below that threshold. Audit time was 2.68 seconds; full recorded runtime was 3.67 seconds. No model was fitted. AUROC is null.

`configs/data_manifest.json` pins raw byte hashes. Existing local data or downloaded data with different hashes stop the run. No silent overwrite. Each run writes a new JSON under `results/01_data_setup/`, with source revision, notebook source hash, versions, lock hash, settings, timings, and artifact hashes. The length plot contains aggregates only.

Duplicate component membership lives under ignored `artifacts/01_data_setup/<run-id>/duplicate_components.json`. Notebook 02 should use the component artifact named in the chosen successful audit, verify its hash, and verify the raw input hashes. Re-run notebook 01 to recreate private artifacts on another machine. The component file contains the same data/config identity on each matching run.

The prefix search is checked against brute force on synthetic sets, including the .85 boundary. Failed load/audit runs save a safe failure record. Inspect failures locally; do not commit a traceback containing passages or credentials.

PR #19 is merged and #1 is closed. The user waived further teammate review on November 10, 2026.
