# Group_Number_19 — Human or AI?

**Kaggle team name:** `Group_Number_19`
**Team:** Chris Viets (cv329), Daniel Xing (yx753), Yuchen Gao (yg673)

| Model | Public LB AUROC |
|---|---|
| **Final submission:** CatBoost (style + TF-IDF/LR + GPT-2 + Qwen3.5 features, self-training), rank-blended with a fine-tuned RoBERTa-base (weight 0.1) | **0.99162** |
| Team baseline: LightGBM on 53 style features + TF-IDF/LR score | 0.9754 |

The final pipeline is described below; every experiment, parameter and result table is in
[`YUCHEN_EXPERIMENTS.md`](YUCHEN_EXPERIMENTS.md). The team baseline is documented in the
second half of this README.

---

## Final submission

### Pipeline

1. **Features (193 per text).** 126 hand-crafted style features (layout and blank lines,
   sentence-length burstiness, punctuation, word rarity, compressibility, rhyme), the
   cross-fitted score of a TF-IDF (word 1–3 + char 3–5-gram) logistic regression, 35
   token-probability statistics from frozen GPT-2 small/medium (perplexity, log-prob spread,
   Fast-DetectGPT), and 31 Binoculars-style features from frozen Qwen3.5-2B Base/Instruct.
2. **Classifier.** CatBoost (depth 6, learning rate 0.05, up to 1,500 trees, early stopping
   on a 10% holdout), 3 seeds averaged.
3. **Self-training.** One round: rows of `unlabeled.csv` and `test.csv` (text only) with
   P > 0.95 or P < 0.05 are added with hard pseudo-labels, then CatBoost is refit.
4. **Blend.** Rank average of the CatBoost scores (weight 0.9) and a fine-tuned RoBERTa-base
   classifier (weight 0.1).

Model selection used **leave-one-topic-cluster-out CV** (8 k-means topic clusters, AUROC
weighted to the test cluster mix), because test is ~36% poetry/fiction versus 1.5% of train
and random CV scores every model at ~0.999. GPT-2 and Qwen3.5 are only used as frozen local
feature extractors; no LLM labels or classifies competition texts.

### Environment

- CPU part (features, CatBoost, blending): Python 3.11 and the pinned
  `requirements-yuchen.lock.txt` (catboost 1.2.10, lightgbm 4.7.0, scikit-learn 1.9.1,
  wordfreq 3.1.1, ...). Runs on a laptop.
- GPU part (GPT-2 / Qwen3.5 features, RoBERTa): SLURM jobs in `*.sbatch`. They build their own
  environments with `uv`; tested with PyTorch 2.11 (CUDA 12.8) and transformers 5.22 on RTX
  2080 Ti (GPT-2, RoBERTa) and RTX A6000 (Qwen3.5, needs >= 24 GB and bf16).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r submissions/Group_Number_19/requirements-yuchen.lock.txt
# put train.csv, test.csv, unlabeled.csv in data/ at the repository root
```

All `yuchen_*.py` commands run from the **repository root**; outputs go to `results/yuchen/`.

### Reproduce `submission.csv`

```bash
S=submissions/Group_Number_19

# 1. Style features and cross-fitted TF-IDF/LR scores (CPU, ~10 min); also writes clusters.csv
python $S/yuchen_v2.py --stage features
python $S/yuchen_v2.py --stage lr

# 2. LM features and RoBERTa (GPU). Run from a cluster folder holding data/, the $S/*.py and
#    $S/*.sbatch files, and clusters.csv. Run bino_feats first: it creates the
#    environment that deberta.sbatch reuses.
sbatch gpt2_feats.sbatch                                          # -> out/gpt2, out/gpt2-medium (~18 min)
MODELS="--splits test train unlabeled" sbatch bino_feats.sbatch   # -> out/qwen35_2b (~24 min)
TAG=rob_a ARGS="--model FacebookAI/roberta-base --nl-marker off --bs 8 --accum 2 --folds 0 1 2 3 --skip-full" sbatch deberta.sbatch
TAG=rob_b ARGS="--model FacebookAI/roberta-base --nl-marker off --bs 8 --accum 2 --folds 4 5 6 7" sbatch deberta.sbatch
# copy out/gpt2* and out/qwen35_2b to results/yuchen/lm_features/, out/rob_a and out/rob_b to results/yuchen/roberta/

# 3. CatBoost with self-training (v4) and its leave-cluster-out predictions (CPU, ~5 min)
python $S/yuchen_semi.py --stage prep
python $S/yuchen_semi.py --stage predict --tag v4 --bino results/yuchen/lm_features/qwen35_2b --out results/yuchen/v4_bino
python $S/yuchen_v3.py --stage cv --bino results/yuchen/lm_features/qwen35_2b --out results/yuchen/v4_bino

# 4. Blend with RoBERTa -> final submission
python $S/yuchen_blend.py --cat-oof results/yuchen/v4_bino/lco_v2+lm+bino+lr+pseudo.csv \
    --deb-dir results/yuchen/roberta/rob_a results/yuchen/roberta/rob_b \
    --cat-test results/yuchen/v4_bino/submission_v4_unl_target.csv
# -> results/yuchen/blend/submission_blend_w0.1.csv  (id,label; submitted file, public 0.99162)
```

Step 3 also writes `submission_v4_unl.csv` (self-training on `unlabeled.csv` only, no test
text), the backup submission.

### Validation and report experiments

```bash
# Report template (fixed 2,000-row validation = fold 0 of folds.py): Q1-Q4 tables and learning curve
python $S/yuchen_report.py --stage all          # also writes folds.csv (copy it to the GPU folder)
TAG=rob_fixed ARGS="--model FacebookAI/roberta-base --nl-marker off --bs 8 --accum 2 --fixed-split folds.csv" sbatch deberta.sbatch
python $S/yuchen_report.py --stage q4blend      # Q4 for the final blend (after copying out/rob_fixed to results/yuchen/roberta/fixed)

# Leave-cluster-out experiments
python $S/yuchen_v3.py --stage cv                                  # LM-feature ablation
python $S/yuchen_semi.py --stage semi                              # self-training pools
python $S/yuchen_semi.py --stage curve                             # learning curve
python $S/yuchen_models.py --stage cv                              # 6 classifiers on the same features
python $S/yuchen_lco_hparams.py                                    # CatBoost depth / learning rate
```

### Seeds

- `SEED = 20261109` (`folds.py`) for folds, topic clusters, inner splits and TF-IDF cross-fitting.
- CatBoost: seeds 20261109, 20261110, 20261111, averaged.
- RoBERTa: seed 0 + fold index (100 for the full fit, 200 for the fixed-split fit).
- `folds.csv` / `clusters.csv` are cached on first run, so all experiments use identical splits.

### Results

| Version | Change | Leave-cluster-out AUROC (with self-training from v2) | Public LB |
|---|---|---|---|
| Team baseline | LightGBM, 53 style + TF-IDF/LR | 0.9939 | 0.9754 |
| v2 | CatBoost, 126 style + TF-IDF/LR, self-training | 0.9975 | 0.9788 |
| v3 | + GPT-2 features | 0.9992 | 0.9887 |
| v4 | + Qwen3.5 Binoculars features | 0.99939 | 0.99146 |
| **Final** | + 10% RoBERTa rank blend | **0.99953** | **0.99162** |

On the fixed 2,000-row validation split the final blend scores 0.99928 (train 0.99998).

### Files

| File | Purpose |
|---|---|
| `yuchen_features.py`, `yuchen_features_v2.py` | 37 + 36 extra style features (word rarity, compressibility, rhyme, ...) |
| `yuchen_v2.py` | Feature / TF-IDF-LR caches, leave-cluster-out CV, CatBoost + self-training |
| `yuchen_lm_features.py`, `gpt2_feats.sbatch` | GPT-2 token-probability features (GPU) |
| `yuchen_bino_features.py`, `bino_feats.sbatch` | Qwen3.5 Binoculars features (GPU) |
| `yuchen_v3.py` | Leave-cluster-out CV with LM features |
| `yuchen_semi.py` | Self-training pools, learning curve, **v4 CatBoost submission** |
| `yuchen_deberta.py`, `deberta.sbatch` | Fine-tuned RoBERTa (DeBERTa-v3 failed with NaN loss) |
| `yuchen_blend.py` | Rank blend with RoBERTa, **final submission** |
| `yuchen_models.py`, `yuchen_lco_hparams.py` | Classifier comparison, CatBoost hyperparameter check |
| `yuchen_report.py` | Report-template experiments on the fixed validation split |
| `yuchen_experiments.py`, `verify_yuchen_results.py`, `analyze_yuchen_results.py` | v1 CatBoost runner and checks |
| `requirements-yuchen.lock.txt` | Pinned CPU environment |

---

## Team baseline (LightGBM)

### Setup

Tested with Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, lightgbm 4.7.0.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data files (`train.csv`, `test.csv`) are not committed. Run every command from this
directory and point `DATA` at the folder that holds them (scripts default to `../../../data/`):

```bash
cd submissions/Group_Number_19
DATA=/path/to/data
```

### Predict using LightGBM model

```bash
python predict.py --train $DATA/train.csv --test $DATA/test.csv --out submission.csv
```

Fits the full pipeline on all of `train.csv` (~1 min on a laptop CPU) and writes
`id,label` with P(AI) for each test row.

### Validation of LightGBM

```bash
# LightGBM model, fixed stratified 5-fold CV (also reports logreg alone and style-only LightGBM)
python lgbm_stack.py --train $DATA/train.csv

# Same, leave-one-topic-cluster-out CV (8 k-means topic clusters)
python lgbm_stack.py --train $DATA/train.csv --test $DATA/test.csv --cv cluster

# TF-IDF baselines and hyperparameter sweeps
python baseline_tfidf.py --train $DATA/train.csv
python sweep_c.py --train $DATA/train.csv --svd-dims 0 --models logreg svm --Cs 0.3 1 3 10 30
```

Each script writes per-fold metrics and out-of-fold predictions as CSVs in this directory.

### Seeds and config

- Global seed `SEED = 20261109` (`folds.py`), used for folds, SVD, k-means, LightGBM and inner splits.
- Fold and cluster assignments are cached to `folds.csv` / `clusters.csv` on first run, so
  every experiment is scored on identical splits; delete them to rebuild from the seed.
- Leakage control: TF-IDF and models are refit inside each fold; LightGBM early-stops on a
  10% holdout of the training rows; the stacked LR score is out-of-fold within each training split.

### Files

| File | Purpose |
|---|---|
| `folds.py` | Fixed stratified 5-fold split |
| `clusters.py` | Topic clusters for leave-cluster-out CV |
| `baseline_tfidf.py` | TF-IDF features (tf/idf/n-gram options) + logistic regression or linear SVM |
| `sweep_c.py` | Sweeps C over models, tf/idf schemes, n-gram ranges, SVD |
| `style_features.py` | Layout, burstiness, punctuation and vocabulary features |
| `layout.py` | Optional whitespace/markdown normalization (`--strip-layout`) |
| `lgbm_stack.py` | CV of the LightGBM stack |
| `predict.py` | Fits the LightGBM model and writes `submission.csv` |

### Results (AUROC)

| Model | Random 5-fold CV | Leave-cluster-out (test-mix weighted) | Public LB |
|---|---|---|---|
| TF-IDF logistic regression | 0.9739 | 0.9160 | — |
| LightGBM, style features only | 0.9979 | 0.9858 | — |
| **LightGBM, style + LR score** | **0.9990** | **0.9939** | **0.9754** |
| Same, layout stripped (`--strip-layout`) | 0.9826 | — | 0.919 |

Lessons:
- In train, almost no human text contains a blank line (17 of 5,000) while 66% of AI texts
  do; layout features exploit this. Removing them lowered the leaderboard score, so they help
  on test as well.
- Test is ~36% poetry versus ~1.5% in train. Wording-based TF-IDF transfers poorly to unseen
  genres (leave-cluster-out mean AUC 0.920), which random CV hides.
- CV still overestimates the leaderboard, suggesting test poetry differs from train poetry.

---

## AI assistance

No AI model or commercial LLM API was used to label or classify competition texts. GPT-2 and
Qwen3.5 appear only as frozen, open-weight models run locally to compute token-probability
features (see `YUCHEN_EXPERIMENTS.md`).

**Team baseline (TF-IDF, style features, LightGBM stack).** Claude Code (Claude Opus 5.5) was used to write the pipeline code, run
cross-validation experiments, and analyze results (TF/IDF variants, C sweeps, n-gram ranges,
linear SVM, style features, LightGBM stacking, train/test shift diagnostics). We reviewed the
code, chose which experiments to run, and checked results against the Kaggle public
leaderboard.

**Yuchen's CatBoost line (`yuchen_*.py`).** Codex (v1) and Claude Code (Claude Opus 5.5,
v2 onward) proposed ideas, wrote the code and ran the experiments. Yuchen chose among the proposals and suggested
alternatives (e.g. using the lab GPU cluster, Qwen3.5 instead of Qwen2.5, adding
`unlabeled.csv`, trying a fine-tuned transformer and other classifiers), and decided what to
submit to Kaggle.
