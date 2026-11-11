# AI vs Human Text — Cornell Tech CS5785 Kaggle Competition (Fall 2026)

Predict the probability that a ~210-word passage was written by an AI model rather than a human.
Metric: AUROC. Competition page: https://www.kaggle.com/competitions/cornell-tech-cs-5785-combined-2026-ai-vs-human-text

Project context, model list and results so far: **[AGENTS.md](AGENTS.md)**.

## In the course repo (`submissions/<team>/pretrained/`)

This folder is the research repo imported with its history and trimmed to the pipeline behind the three
uploaded files (`kaggle_submissions/final_{A,B,C}_*.csv`). Differences from the sections below:

- **Data** stays outside git. `data/...` paths resolve through `--data-dir PATH`, `$HOA_DATA_DIR`, `./data`,
  or the team default in `../common/data.py`.
- **Fixed validation split** (shared with `../classical/`): `--set cv.scheme=fixed` trains on 8,000 rows and
  scores the 2,000 in `../common/val_ids.csv`; `cv.scheme=inner` is 5-fold CV inside the 8,000 for tuning.
- **Report experiments**:
  ```bash
  python run_all.py --profile fixed                          # every model of blends A/B/C on the fixed split
  python scripts/report_experiments.py collect --session outputs/sessions/<session>
  python scripts/report_experiments.py hpsearch              # also: curve, semisup, errors, debias
  python scripts/report_tables.py                            # -> reports/tables, reports/figures
  ```
  Results are in `results/`; the report is `reports/kaggle_pretrained_report.tex`.
- **Uploaded files**: `python run_all.py --profile blends` trains the 5-fold models (seed 42, ~4 h on two
  12 GB GPUs). Then:
  ```bash
  python make_submission.py --run <runs> [--weights ...] --name <file>
  ```
  Blend B, the final pretrained selection, is lm_gpt2, lm_gpt2_medium, binoculars_qwen05b and binoculars_qwen15b
  with equal weights. Blend C adds ft_distilroberta. Blend A's weights come from greedy selection on topic-held-out
  CV (`--profile topic_cv`, then `scripts/select_blend.py`; weights in `../predict.py`).
  Run against the saved models, this re-creates all three uploaded files exactly (`results/reproduction.csv`).
- Removed in the trim: the smoke/pc/mac profiles and 12 model configs not used by the final blends.
  `experiments/results.csv` and `experiments/sessions/` keep the record of everything that was run.

## Setup

### 1. Get the data
Download the competition files from the Kaggle **Data** tab into `data/`
(`train.csv`, `test.csv`, `unlabeled.csv`, `train_starter_2000.csv`, `sample_submission.csv`). `data/` is git-ignored.

### 2a. MacBook (Apple silicon)
```bash
conda activate cs-aml                       # or: conda env create -f environment.yml && conda activate cs-aml
conda install -c conda-forge lightgbm       # bundles libomp on macOS
pip install -r requirements.txt
python scripts/check_setup.py               # should report "Apple MPS GPU ... ok"
```

### 2b. Windows / Linux PC with NVIDIA GPUs (RTX 5070 + RTX 3060)
The RTX 50xx (Blackwell, `sm_120`) needs a PyTorch build for **CUDA 12.8 or newer**; older builds fail with
"no kernel image is available". Install that build *before* the requirements:
```bash
conda env create -f environment.yml
conda activate cs-aml
pip install torch --index-url https://download.pytorch.org/whl/cu128     # or a newer cu12x index
pip install -r requirements.txt
python scripts/check_setup.py               # both GPUs should show "matmul + bf16 autocast test passed"
```
A recent NVIDIA driver (R570 or newer) is required for CUDA 12.8. `check_setup.py` prints the exact fix if the
installed PyTorch doesn't support a card.

### 3. Pretrained weights (optional – otherwise downloaded on first use)
```bash
python scripts/download_models.py                 # all models, into the Hugging Face cache (~16 GB total)
python scripts/download_models.py --profile mac   # only what the mac profile needs (~4 GB)
python scripts/download_models.py --local         # also keep copies in ./pretrained/ (git-ignored)
python scripts/check_setup.py --models            # load each model once and run 2 texts through it
```

## Run everything

```bash
python run_all.py --profile smoke    # ~20-40 min: every model on a tiny budget -> "does it all run?"
python run_all.py --profile pc       # full benchmark on the desktop (both GPUs + CPU in parallel)
python run_all.py --profile mac      # what is practical on the MacBook
```
You get a live progress bar, one line per finished job, and at the end:

- `experiments/sessions/<session>/report.md` — results table, ensembles, learning-curve / semi-supervised /
  robustness comparisons, graphs (model comparison, ROC incl. low-FPR, score distributions, per-topic AUROC,
  model agreement, calibration, cost vs accuracy) and sanity checks. Tracked in git.
- `outputs/submissions/<session>__best.csv` — the recommended Kaggle upload.
- `outputs/sessions/<session>/logs/<job>.log` — full log of every job.

Useful options: `--only a,b`, `--skip a,b`, `--set cv.n_folds=3`, `--gpus 0`, `--cpu-workers 2`, `--dry-run`,
`--resume outputs/sessions/<session>` (after Ctrl+C or a crash), `--report outputs/sessions/<session>`.

### Which models run on the MacBook?

| Runs well on the Mac | Possible but slow (hours) | PC (12 GB NVIDIA) only |
|---|---|---|
| `tfidf_logreg`, `tfidf_svm`, `tfidf_nb`, `stylometric_gbm`, `stylometric_gbm_ext`, `lm_gpt2`, `embed_bge_base`, `embed_e5_base`, `binoculars_qwen05b`, `ft_distilroberta`, `ft_deberta_v3_small` (hold-out split) | `lm_gpt2_medium`, `ft_roberta_base`, `ft_electra_base`, `ft_modernbert_base`, `ft_openai_detector`, `ft_deberta_v3_base` | `ft_deberta_v3_large`, `ft_openai_detector_large`, `binoculars_qwen15b(_zeroshot)` |

## Train a single model

Each model has its own file in `src/models/` and one or more configs in `configs/models/`.
Pick a config with `--model`; change any parameter with `--set` (no code edits needed).

```bash
python train.py --model tfidf_logreg                      # 5-fold CV, then fit all + write submission
python train.py --model tfidf_logreg --quick              # 2k-example smoke test (~30 s)
python train.py --model tfidf_logreg --set model.params.C=10 model.params.tfidf.char.ngram_range=[2,6]
python train.py --model stylometric_gbm_ext --set model.params.estimator=random_forest
python train.py --model ft_deberta_v3_base --device cuda --set model.params.epochs=4 model.params.lr=1.5e-5
python train.py --model binoculars_qwen05b --device mps
```

| What | How |
|---|---|
| Out-of-domain validation | `--set cv.scheme=cluster` |
| Learning curve | `--set data.train_frac=0.25` |
| Pseudo-labelling / self-training | `--set semi_supervised.enabled=true semi_supervised.pool=[unlabeled,test]` |
| Text cleaning ("debiasing") | `--set preprocessing.collapse_blank_lines=true preprocessing.map_homoglyphs=true` |
| CV only, no submission | `--cv-only` |
| Torch device | `--device auto/cuda/cuda:1/mps/cpu`, config `device:`, or env `AIVH_DEVICE` |
| Note in the results log | `--notes "char 2-6 grams"` |

Every run logs to `experiments/results.csv` and refreshes the results table in `AGENTS.md`.
Model files, OOF predictions and submissions go to `outputs/` (git-ignored). Progress bars (tqdm) are shown for
preprocessing, feature extraction, CV folds, boosting rounds, training epochs and prediction.

## Submit

Upload a CSV from `outputs/submissions/` on Kaggle, then record the score:

```bash
python scripts/record_kaggle_score.py --run <run_name or outputs/submissions/<file>.csv> --score 0.9xx
python scripts/blend.py --runs outputs/runs/<runA> outputs/runs/<runB> --method greedy   # custom blends
```

## No external labelled data

Competition rule 4(b): "No additional externally sourced labeled datasets may be used to train, validate, or
determine competition predictions." Supervised labels come only from `train.csv`. `data.external_files` must stay
empty (`src/data.py` refuses anything else).

## Repository layout

See [AGENTS.md](AGENTS.md#directory-layout).
