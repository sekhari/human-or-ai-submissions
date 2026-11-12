# group-number-5 — Human or AI? (Cornell Tech AML)

**Kaggle team name:** `group-number-5`  
**Team members:**

| Name | NetID |
|---|---|
| `Shifeng Hong` | `sh2769` |
| `Fangfei Dong` | `fd284` |
| `Nathan Wolf` | `ntw25` |

## Task summary

Binary classification of passages as human-written (`0`) or AI-generated (`1`).
Each passage is embedded into a 768-dimensional vector with
`embeddinggemma-2:270m` served by [Ollama](https://ollama.com); the classifier is
an SVC with a polynomial kernel (`C=10`, Platt-calibrated probabilities).

The final submission model is trained on **all 10,000 labelled rows — no
held-out split**. A 2,000-row stratified holdout is used only to report the
validation metrics in the report.

## Repository layout

```text
pipeline.py       # embedding (with .npy cache), model factory, shared constants
train.py          # optional holdout evaluation + full-data fit -> model.joblib
predict.py        # score test.csv -> submission.csv
requirements.txt  # numpy, pandas, scikit-learn<1.11, joblib, ollama
data/             # Kaggle data, not committed
```

## 1. Environment

- Python 3.11 or newer (developed on CPython 3.12)
- CPU only; no GPU required
- [Ollama](https://ollama.com) running locally with the embedding model pulled

```bash
cd submissions/group-number-5

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the Ollama server if it is not already running (desktop app also works)
ollama serve &

# One-time: download the embedding model (~380 MB)
ollama pull embeddinggemma-2:270m
```

`requirements.txt`:

```text
numpy>=1.24,<3
pandas>=2.0,<4
scikit-learn>=1.3,<1.11
joblib>=1.3
ollama>=0.4
```

## 2. Data

Competition data is **not** committed. Download `train.csv`, `test.csv` and
`sample_submission.csv` from the Kaggle competition page and place them here:

```text
submissions/group-number-5/data/
  train.csv              # 10,000 labelled rows: id, text, label
  test.csv               # 3,000 rows to score: id, text
  sample_submission.csv  # id, label (all 0.5)
```

`data/unlabeled.csv` is optional and **not used** by this pipeline. `data/` and
`*.csv` are git-ignored, so nothing from the competition appears in the PR.

## 3. Reproduce `submission.csv` (exact commands)

From `submissions/group-number-5`, with the data in place:

```bash
# 1) Fit the classifier on all 10,000 labelled rows (no held-out split)
python train.py --train data/train.csv --holdout 0 --model-out model.joblib

# 2) Score the 3,000 test rows and write the Kaggle file
python predict.py --test data/test.csv --model model.joblib --out submission.csv
```

Expected output (CPU, cached embeddings):

```text
Loaded cached embeddings data/train_embeddinggemma2_270m.npy (10000, 768)
data: 10000 rows x 768 dims, labels [5000, 5000]
holdout disabled (--holdout 0): fitting on all 10000 labelled rows
final model trained on 10000 rows in 58.6s -> model.joblib
metrics -> metrics.json

Loaded cached embeddings data/test_embeddinggemma2_270m.npy (3000, 768)
wrote submission.csv (3000 rows) | mean P(AI)=0.5112 | predicted AI rate=0.5047
```

Produced files (all git-ignored):

| File | Size | Content |
|---|---|---|
| `model.joblib` | ~28 MB | SVC + metadata bundle, fit on 10,000 rows |
| `metrics.json` | ~0.6 KB | config and (if enabled) holdout metrics |
| `submission.csv` | ~75 KB | `id,label` with `label = P(AI)` for 3,000 rows |

### Verify the output

```bash
python -c "import pandas as pd; s = pd.read_csv('submission.csv'); t = pd.read_csv('data/sample_submission.csv'); print(len(s), list(s.columns), s.id.equals(t.id), bool(s.label.between(0, 1).all()))"
# 3000 ['id', 'label'] True True

shasum -a 256 submission.csv
# 28799713692d18ab6d9353bce4bede8d7de61440b65ee156d9d2ce29e1356a3b
```

The SHA-256 identifies the file used for the Kaggle submission: it is identical
to the output of the original `predict.ipynb` pipeline (verified, maximum
absolute difference `0.0` on all 3,000 rows) and to every re-run with the same
data and library versions. `submission.csv` is git-ignored (root `.gitignore`
has `*.csv`), so a `git diff` will not show it — use the checks above instead.

`predict.ipynb` is kept as the original experiment notebook; `train.py` and
`predict.py` are the reproducible entrypoints that replace it.

### Why `--holdout 0`

`train.py` always refits the saved model on every labelled row, whether or not
evaluation runs. `--holdout 0` simply skips the evaluation step, so the script
does one fit on all 10,000 rows — exactly how the leaderboard model was trained.
Running the default `--holdout 2000` instead produces the same `model.joblib`
content (and the same `submission.csv`) but additionally records the validation
metrics; that is the run used for the numbers in the report.

### First run without embedding caches

If `data/*_embeddinggemma2_270m.npy` is missing (fresh clone), the scripts embed
the text through Ollama first and write the cache, which takes a few minutes for
10,000 rows; later runs load the cache and train in about one minute. The cache
is keyed by row order and validated against the CSV row count; to force a
re-embed, add `--no-cache` or delete the `.npy` file.

## 4. Train / evaluate on the labelled validation protocol

The validation split is a stratified 2,000-row holdout of `train.csv` (80/20 of
the 10,000 labelled rows), drawn with `random_state=42`. Model selection only
ever looks at this holdout; `test.csv` is never used for tuning. After reporting
the metrics, the script refits on all labelled rows and saves the artifact:

```bash
python train.py \
  --train data/train.csv \
  --holdout 2000 \
  --seed 42 \
  --model-out model.joblib \
  --metrics-out metrics.json
```

Expected output:

```text
holdout (2000 rows, seed=42): accuracy=0.9200 f1(AI)=0.9195 roc_auc=0.9722 [41.2s]
final model trained on 10000 rows in 58.1s -> model.joblib
```

`metrics.json`:

```json
{
  "config": {
    "train_file": "data/train.csv",
    "n_rows": 10000,
    "embedding_model": "embeddinggemma-2:270m",
    "embedding_dims": 768,
    "seed": 42,
    "holdout": 2000,
    "model": {"type": "SVC", "kernel": "poly", "C": 10.0, "probability": true},
    "sklearn_version": "1.9.1"
  },
  "holdout": {
    "n_train": 8000,
    "n_holdout": 2000,
    "accuracy": 0.92,
    "f1_ai": 0.9195171026156942,
    "roc_auc": 0.9722115,
    "fit_seconds": 41.2
  },
  "final_model": {
    "path": "model.joblib",
    "trained_on": 10000,
    "trained_on_full_data": true,
    "fit_seconds": 58.1
  }
}
```

Numbers for the report: **accuracy 0.9200, F1(AI) 0.9195, ROC-AUC 0.9722** on
the 2,000-row holdout (seed 42). Training accuracy on the full data is 1.000,
so the holdout is the only meaningful estimate of generalisation.

To evaluate without the final refit (faster when tuning):

```bash
python train.py --train data/train.csv --holdout 2000 --skip-final-fit
```

All flags:

| Flag | Default | Purpose |
|---|---|---|
| `--train` | required | Labelled CSV path |
| `--holdout` | `2000` | Held-out rows; `0` disables evaluation (full-data fit) |
| `--seed` | `42` | Holdout split and SVC `random_state` |
| `--C` | `10.0` | SVC inverse regularisation |
| `--kernel` | `poly` | SVC kernel |
| `--embeddings` | `data/<stem>_embeddinggemma2_270m.npy` | Embedding cache path |
| `--no-cache` / `--no-cache-write` | off | Ignore / skip writing the cache |
| `--embed-model` | `embeddinggemma-2:270m` | Ollama embedding model |
| `--batch-size` | `32` | Ollama batch size |
| `--text-col` / `--label-col` / `--id-col` | `text` / `label` / `id` | Column names |
| `--skip-final-fit` | off | Evaluate only; save no model |
| `--model-out` / `--metrics-out` | `model.joblib` / `metrics.json` | Output paths |

## 5. Predict options

Default (probability scores, what we submitted):

```bash
python predict.py --test data/test.csv --model model.joblib --out submission.csv
```

Hard 0/1 labels instead:

```bash
python predict.py --test data/test.csv --model model.joblib \
  --out submission.csv --hard --threshold 0.5
```

| Flag | Default | Purpose |
|---|---|---|
| `--test` | required | Test CSV path |
| `--model` | `model.joblib` | Artifact written by `train.py` |
| `--out` | `submission.csv` | Output path |
| `--hard` / `--threshold` | off / `0.5` | Emit 0/1 labels instead of probabilities |
| `--embeddings` | `data/<stem>_embeddinggemma2_270m.npy` | Embedding cache path |
| `--no-cache` / `--no-cache-write` | off | Ignore / skip writing the cache |
| `--embed-model` | model's embedding model | Override the embedding model |
| `--summary-out` | none | Optional JSON summary of the prediction run |
| `--text-col` / `--id-col` | `text` / `id` | Column names |

`--model` must point at an artifact produced by `train.py`; `model.joblib` is
git-ignored, so it has to be generated locally before predicting.

## 6. Seeds and configuration

| Item | Value |
|---|---|
| Global seed | `42` |
| Validation split | stratified 2,000 rows (20%), `random_state=42` |
| Embedding model | `embeddinggemma-2:270m` via Ollama, 768 dims |
| Classifier | `SVC(kernel="poly", C=10.0, probability=True, random_state=42)` |
| Final fit | all 10,000 labelled rows, no holdout |
| Score | `predict_proba(X)[:, 1]` = P(AI) |
| Label convention | `1 = AI-generated`, `0 = human-written` |

SVC fitting is deterministic given the same inputs; the Platt calibration uses
seeded internal CV shuffling (`random_state=42`). Same data, caches and library
versions therefore reproduce `model.joblib`, `metrics.json` and
`submission.csv` bit-for-bit.

**scikit-learn note:** `SVC(probability=True)` is deprecated in scikit-learn
1.9 (removal in 1.11). It is kept because it is the estimator that produced the
submitted file; `requirements.txt` pins `scikit-learn<1.11`. On a newer release,
replace it with `CalibratedClassifierCV(SVC(...), ensemble=False)`, which is the
documented equivalent — probabilities may differ in the last digits.

## 7. Runtime

| Step | Time (CPU, cached embeddings) |
|---|---|
| Embed 10,000 train rows (first run only) | a few minutes |
| Embed 3,000 test rows (first run only) | ~1 minute |
| Holdout fit (8,000 rows) | ~41 s |
| Full fit (10,000 rows) | ~58 s |
| Predict 3,000 test rows | seconds |

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `connection refused` / `ollama: command not found` | Start `ollama serve` (or install Ollama) |
| `model "embeddinggemma-2:270m" not found` | `ollama pull embeddinggemma-2:270m` |
| `model artifact not found: model.joblib` | Run `train.py` first; the artifact is not committed |
| `Ignoring stale cache ... has N rows` | CSV and cache disagree; the script re-embeds and overwrites the cache |
| `--skip-final-fit with --holdout 0` | Nothing would be trained; drop one flag |
| `column 'label' not in ...` | Wrong file or custom schema; pass `--label-col` / `--text-col` / `--id-col` |
| Probabilities differ in the last digits | Different scikit-learn version; install `scikit-learn<1.11` |

## 9. Status

| Item | State |
|---|---|
| `train.py` / `predict.py` / `pipeline.py` | implemented; runs end-to-end on the competition files |
| Holdout metrics (2,000 rows, seed 42) | accuracy `0.9200`, F1(AI) `0.9195`, ROC-AUC `0.9722` |
| Submission model | SVC fit on all 10,000 labelled rows (`--holdout 0`) |
| `submission.csv` reproduced from the commands above | confirmed; 3,000 rows, SHA-256 `28799713692d18ab6d9353bce4bede8d7de61440b65ee156d9d2ce29e1356a3b` |

## 10. AI assistance

Written with the help of AI coding assistants (Claude and DeepSeek coding
agents). AI help was used for repository/pipeline scaffolding (splitting
`predict.ipynb` into `pipeline.py` / `train.py` / `predict.py`), CLI and caching
boilerplate, and this README. All modelling decisions, the validation protocol,
the reported numbers and the final submission were run and checked manually;
nothing in `submission.csv` is produced by an external LLM labelling service.

This disclosure is the basis for the Q6.3 answer in the report and is updated
whenever the tooling changes.
