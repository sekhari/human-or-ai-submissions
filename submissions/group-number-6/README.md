# Group Number 6: Human or AI?

**Kaggle team name:** Group Number 6 (folder/branch slug `group-number-6`)
**Team members:** Tejas Mutneja (tm784), Jack Li (jl4865), Sahil Dhumale (sd2329)

**Report:** [`report.pdf`](report.pdf)

**Final model:** an equal-weight logit average of three views of Unicode-normalised text:

- **A:** word 1–3-gram TF-IDF + logistic regression
- **B:** character 2–5-gram TF-IDF + logistic regression
- **C:** 39 stylometric features + histogram gradient boosting

It scores **0.9984 validation AUROC** on the fixed 2,000-row split, compared with 0.9968 for the best single model. It is also more robust when the formatting shortcut is removed (0.987 vs 0.960), when the text is adversarially edited (0.998 vs 0.992) and when the topic is unseen (0.993 vs 0.989). CPU only; the full pipeline trains in about 80 s.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python 3.10+; tested on 3.13
```

Put the Kaggle files in `./data/`: `train.csv`, `test.csv` and `unlabeled.csv`. They are not committed.

## Reproduce the Kaggle submission

```bash
python train.py                          # fit configs/final.json on all 10,000 rows -> models/final.joblib (~80 s)
python predict.py                        # -> submission.csv (3,000 rows: id, label = P(AI))
```

Check the selected model on the fixed validation protocol (train on 8,000 rows, score the 2,000 held-out rows):

```bash
python train.py --validate               # expected: train AUROC 1.0000, val AUROC 0.9984
```

Comparison submission using the single best model (C) on its own:

```bash
python train.py --config configs/baseline_style_gbm.json --model-out models/baseline_style_gbm.joblib
python predict.py --model models/baseline_style_gbm.joblib --out submission_baseline_style_gbm.csv
```

## Reproduce every experiment in the report

`./run_all.sh` runs everything in order (about 1.5 h on 2 cores). Individual steps:

| Step | Command | Outputs |
|---|---|---|
| Split and leakage check | `python experiments/00_make_split.py` | `splits/val_ids.txt`, `results/split_leakage_check.json` |
| Q1: model comparison | `python experiments/01_compare_models.py` | `results/q1_cv_grid.json`, `results/q1_comparison.json` |
| Q2: learning curve | `python experiments/02_learning_curve.py --model C_style_gbm` (also `B_char_lr`) | `results/q2_*.json`, `figures/q2_*.png` |
| Q3: unlabeled data | `python experiments/03_unlabeled.py --model B_char_lr` (also `C_style_gbm`) | `results/q3_*.json` |
| Q5: improvement | `python experiments/05_improvement.py` | `results/q5_improvement.json`, `figures/q5_robustness.png` |
| Q4: error analysis | `python experiments/04_error_analysis.py --config ensemble+norm` | `results/q4_*`, `figures/q4_subgroups.png` |

Q4 runs after Q5 because it analyses the model that Q5 selects. Run the steps one at a time. Running two together on a small machine oversubscribes the CPU, because gradient boosting uses OpenMP, and that distorts the timings.

## Protocol, seeds and configuration

- **Seed:** `20261102` everywhere: the split, the CV folds, the GBM, the SVD/KMeans clustering and the perturbations. Learning-curve subsamples use seeds 0, 1 and 2.
- **Validation:** a stratified 2,000-row hold-out from `train.csv` (1,000 human, 1,000 AI). Its ids are saved in `splits/val_ids.txt` and used for every comparison.
- **Hyperparameters:** chosen by 5-fold stratified CV inside the 8,000-row training partition, with vectorisers re-fit per fold. The validation set is never used for tuning.
- **Leakage:** a near-duplicate check (word TF-IDF cosine similarity) found 2 validation rows above 0.8 against train. The final model is retrained on all 10,000 labeled rows. `unlabeled.csv` and `test.csv` are never used for training the final model.
- **Final configuration:** `configs/final.json`.

| Member | Settings |
|---|---|
| A | word 1–3-grams, min_df=2, sublinear TF, C=256 (liblinear) |
| B | `char` 2–5-grams, min_df=3, sublinear TF, max 300k features, C=256 |
| C | HistGradientBoosting, lr=0.1, 63 leaves, 500 iterations, L2=1 |

Normalisation applies NFKC, removes zero-width characters, maps Cyrillic/Greek homoglyphs to Latin inside Latin words, and collapses runs of spaces.

## Code layout

```
src/data.py        loading and fixed split          src/features.py  normalisation and stylometric features
src/models.py      model factories A–E              src/ensemble.py  logit-average ensemble
src/perturb.py     synthetic adversarial edits (stress tests only)
experiments/       one script per report section    results/ figures/  outputs used in the report
train.py, predict.py  entrypoints                   configs/           final and baseline configs
```

## AI-tool disclosure

We used Claude (Anthropic) as a coding assistant. It wrote the first drafts of the pipeline, the experiment scripts and the plotting code, and it proposed the robustness stress tests. The team ran the scripts and checked every reported number against `results/*.json`. We read the false positives and false negatives in `results/q4_errors.md` ourselves before writing the failure hypotheses. The team also confirmed that `submission.csv` is regenerated exactly by the commands above. No LLM API was used to label or score competition passages, and no hidden labels were retrieved from RAID or any other source.
