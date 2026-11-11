# Yuchen's experiments: overview

Every version is a classifier on top of a feature table; **all versions include the
TF-IDF + logistic-regression score** as one input feature (team configuration: lower-cased
word 1-3-grams + case-sensitive char 3-5-grams, binary tf, smoothed idf, C=30; cross-fitted
with 3 inner folds that each refit vocabulary/idf, so training rows get out-of-fold scores).

| Version | Inputs to the classifier | Classifier | Unlabeled / self-training | Model selection | Public LB |
|---|---|---|---|---|---|
| Team baseline | 53 team style features + TF-IDF/LR score (54) | LightGBM | none | random 5-fold CV | 0.97543 |
| v1 | 53 team + 37 new stylometry features + TF-IDF/LR score (91) | CatBoost (+ style/stack blend) | compared, not used in final | fixed fold-0 holdout + inner split | not recorded |
| v2 | 90 above + 36 topic-free features (word rarity, compressibility, rhyme, line breaks) + TF-IDF/LR (127) | CatBoost, 3 seeds | one round on test texts, thr 0.95 | leave-one-topic-cluster-out (LCO), weighted to test mix | 0.97880 |
| v3 | v2 + 35 GPT-2 small/medium token-probability features (162) | CatBoost, 3 seeds | same | LCO | 0.98869 |
| v4 | v3 + 31 Qwen3.5-2B Base/Instruct Binoculars features (**193** = 126 style + 1 TF-IDF/LR + 35 GPT-2 + 31 Qwen) | CatBoost, 3 seeds | one round on `unlabeled.csv` + test texts, thr 0.95 | LCO | 0.99146 |
| **v4 + RoBERTa (final)** | rank blend 0.9 x v4 + 0.1 x fine-tuned RoBERTa-base | - | - | LCO (0.99953) | **0.99162** |

Validation protocols used below:
- **LCO** (v2 onward, model selection): 8 k-means topic clusters of train+test text; hold out
  one cluster at a time; AUROC weighted by each cluster's test/train share, plus the same
  AUROC on texts without a blank line. Tracks the test genre shift (poetry 35.8% of test vs
  1.5% of train).
- **Fixed 2,000-row validation** (v1 and the report section at the end): fold 0 of
  `folds.py` (SEED 20261109), 8,000 training rows. Required by the report template; every
  feature model scores ~0.999 here, so it cannot rank them.

| Script | Purpose | Output (`results/yuchen/...`) |
|---|---|---|
| `yuchen_experiments.py` | v1 CatBoost runner | `catboost_v1/` |
| `yuchen_features_v2.py`, `yuchen_v2.py` | v2 features, TF-IDF/LR cross-fit caches, LCO CV, submissions | `v2_shift/` |
| `yuchen_lm_features.py`, `gpt2_feats.sbatch` | GPT-2 features (GPU) | `lm_features/gpt2*/` |
| `yuchen_bino_features.py`, `bino_feats.sbatch` | Qwen3.5 Binoculars features (GPU) | `lm_features/qwen35_2b/` |
| `yuchen_v3.py` | v3/v4 LCO CV and submissions (`--bino` adds Qwen) | `v3_lm/`, `v4_bino/` |
| `yuchen_semi.py` | self-training pools, LCO learning curve, **final v4 submission** | `semi/`, `v4_bino/` |
| `yuchen_deberta.py`, `deberta.sbatch`, `yuchen_blend.py` | fine-tuned RoBERTa (DeBERTa failed) and blend | `roberta/`, `blend/` |
| `yuchen_models.py` | 6 classifiers on the v4 features (LCO) | `models/` |
| `yuchen_lco_hparams.py` | CatBoost depth / learning-rate check under LCO | `lco_hparams/` |
| `yuchen_report.py` | report-template experiments on the fixed 2,000-row split | `report/` |

Final submission file: `results/yuchen/blend/submission_blend_w0.1.csv` (public 0.99162), written by
`yuchen_blend.py --cat-oof results/yuchen/v4_bino/lco_v2+lm+bino+lr+pseudo.csv --deb-dir results/yuchen/roberta/rob_a results/yuchen/roberta/rob_b --cat-test results/yuchen/v4_bino/submission_v4_unl_target.csv`.
Its v4 part, `results/yuchen/v4_bino/submission_v4_unl_target.csv` (public 0.99146), is written by
`yuchen_semi.py --stage predict --tag v4 --bino results/yuchen/lm_features/qwen35_2b --out results/yuchen/v4_bino`.

AI tools: v1 was implemented with Codex (see below); v2 onward with Claude Code
(Claude Opus 5.5). In both cases the assistant proposed and implemented; Yuchen chose
directions, redirected them and decided what to submit.

---

# v1: CPU CatBoost experiments

These experiments add CatBoost without modifying the team's original LightGBM pipeline.
The original 53 style features are attributed to the existing team implementation.
New work: 37 additional stylometry features, CatBoost tuning, strict cross-fitted LR
meta-features, CatBoost blending, pseudo-labeling comparisons and seed averaging.
Check the measured feature count in `run.log` rather than relying on this description.

## Run

From the repository root, using Python 3.11:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r submissions/Group_Number_19/requirements-yuchen.lock.txt
python -u submissions/Group_Number_19/yuchen_experiments.py --stage all
python submissions/Group_Number_19/verify_yuchen_results.py
python submissions/Group_Number_19/analyze_yuchen_results.py
```

Place the official `train.csv`, `unlabeled.csv`, and `test.csv` in `data/`.
By default, outputs are written to `results/yuchen/catboost_v1/`.
Alternative locations: `--data /path/to/data --out /path/to/new/run`.
Use the same output directory to resume completed stages; use a NEW directory if
data, source code, environment, thread count or iteration cap changes.
No Kaggle upload or external API classification is performed.

Stages can also be run individually:

```bash
python submissions/Group_Number_19/yuchen_experiments.py --stage style
python submissions/Group_Number_19/yuchen_experiments.py --stage stack
python submissions/Group_Number_19/yuchen_experiments.py --stage pseudo
python submissions/Group_Number_19/yuchen_experiments.py --stage stability
python submissions/Group_Number_19/yuchen_experiments.py --stage final
```

## Protocol

- Same `SEED=20261109` and 5-fold stratification as `folds.py`.
- Fold 0 is the fixed outer validation set (2,000 rows). The other 8,000 are the training pool.
- A fixed stratified 20% internal holdout divides that pool into 6,400 fitting and 1,600 tuning rows.
- CatBoost grid: depth `{4,6,8}` × L2 `{3,10}`, learning rate `0.03`, iteration cap 1,600, patience 150.
- Select hyperparameters and representation using INTERNAL AUROC. Fix the selected
  number of trees, refit on all 8,000 rows, then evaluate the 2,000 outer rows.
- Two representations: the existing style table and that table extended by
  token/character entropy, word/phrase repetition, function-word usage, case,
  contractions, punctuation patterns, and sentence/line patterns.
- LR stacking uses the team's TF-IDF + LR configuration (`C=30`), with THREE inner
  folds. Each fold fits its own vocabulary/IDF and classifier. These stricter LR
  meta-features differ from the team's original five-fold classifier-only cross-fit.
- For early stopping of the stacked CatBoost, LR cross-fitting is confined to the
  6,400 fitting rows. For the final outer fit, it is redone on the 8,000-row pool.
- CatBoost style/stack blend weights `{0,.25,.5,.75,1}` are chosen internally.
- The LightGBM comparison uses the original `fit_lgbm` function unchanged. Its LR
  feature uses the new stricter cross-fit, so it is a controlled local comparator,
  not an exact reproduction of the historical public leaderboard submission.
- LightGBM/CatBoost blends at weights `.1,.2,.3,.5` are exploratory outer-validation
  comparisons. They are not automatically chosen as final submissions.
- Pseudo-label experiments use the internally selected style CatBoost as teacher,
  confidence thresholds `.98,.95`, sample weight `.25`, one round, and the same
  hyperparameters/tree count as the supervised baseline. They use only the supplied
  unlabeled set. Deduplication checks validation text hashes only to exclude overlap;
  validation text/labels are never used for feature fitting or pseudo-label training.
- Three-seed stability holds the split, LR features and hyperparameters fixed.
- Final candidates retrain on all 10,000 labels and average three seeds. They do not
  use pseudo-labels automatically, even when an exploratory comparison improves.

The fixed validation set has already appeared in the team's previous CV work; it
is not a never-seen final test. Tiny differences near AUC 1 should not be treated
as proof of a leaderboard improvement. Public/private performance remains unknown
until an actual competition submission is evaluated.

## Outputs

- `manifest.json`: hashes of input data and relevant code; package versions; seed/config.
- `split.csv`: exact row membership in fitting/tuning/validation partitions.
- `data_audit.json`: missing/duplicate checks and normalized exact-text overlap counts.
- `*_selection.json`: complete CatBoost grid and selected configuration.
- `metrics.json`, `RESULTS.md`: measured local AUROC/log loss and runtime.
- `val_*.csv`: validation ID, ground truth and predicted probability.
- `subgroups.json`: length-bin and blank-line-group metrics; one-class AUROC is null.
- `importance_*.csv`: CatBoost feature importance.
- `cat_pseudo_*_selection.csv`: unlabeled ID, teacher probability and inclusion flag.
- `submission_*.csv`: full-data predictions; `id,label`, finite probabilities in `[0,1]`.
- `*.cbm` and `cache/`: local models and cached cross-fitting outputs; do not commit.

The repository ignores CSV/model data, and this checkout locally excludes `results/`.
Small measured summaries should be copied into the team directory for submission.

`analyze_yuchen_results.py` additionally creates paired stratified bootstrap
intervals (1,000 resamples), a validation comparison plot and up to five false
positives/negatives per main CatBoost variant for manual review. Bootstrap
intervals describe uncertainty on this validation sample; they do not capture
model-selection bias or train/test domain shift.

## Attribution / AI assistance

Codex helped implement the new experimental runner and features, download the official
competition data through the user's logged-in Kaggle session, run experiments, and
check outputs. Team-authored TF-IDF, style features and LightGBM code are reused with
attribution. This note does not claim the user has reviewed all results yet; the user
should inspect the comparisons, understand the selected model, and record their own
modeling decisions before submitting the assignment.

---

# v2: shift-aware CatBoost (Yuchen, 2026-11-10)

## Why v1 needed a different validation

v1 picked models on a random holdout where every candidate scores ~0.999, yet the
team's LightGBM got 0.9754 on the public leaderboard. The gap is distribution shift:
topic cluster 4 (poetry / fiction / book summaries) is 35.8% of test but 1.5% of train.
Within that cluster the v1 ranking is essentially "blank line => AI": all 444 blank-line
test texts in cluster 4 get P(AI)>0.5, and 630 non-blank texts are ranked by weaker
cues. Human poems in test have no blank lines (same formatting as train humans), so the
blank-line rule still holds; the loss comes from ranking the non-blank AI texts.

v2 therefore selects with **leave-one-topic-cluster-out CV (LCO)**, AUROC weighted to the
test cluster mix (the team's proxy; LightGBM style+LR scored 0.9939 here vs 0.9754 LB),
plus the same AUROC restricted to non-blank-line texts.

## What changed

1. `yuchen_features_v2.py`: 36 label-free, topic-independent features: word rarity from
   a general-English unigram model (`wordfreq` Zipf: mean/std/percentiles, rare/OOV
   share), zlib/lzma compressibility, poem structure (end-rhyme rate, line-initial case,
   line-end punctuation, enjambment, archaic words), layout-free punctuation habits.
2. Test-time self-training: CatBoost teacher (3 seeds) labels test rows with
   P>0.95 or P<0.05 (2,652 of 3,000; 1,224 as AI); the student retrains on
   train + those rows (weight 1). Uses test **text only**, never labels. In LCO this is
   simulated by self-training on the held-out cluster.
3. CatBoost depth 6, L2 3, lr 0.05, early stopping on a 10% holdout, 3-seed average;
   LR meta-feature uses the strict 3-fold cross-fit (vocabulary refit per inner fold).

Tried and dropped: augmenting train by collapsing blank lines in AI texts (LCO weighted
AUC 0.9951 -> 0.9934 for LightGBM; it erases a signal that still holds in test);
3 self-training rounds (no gain over 1 for CatBoost); LightGBM blend (+0.0002, noise).

## Results (leave-cluster-out, measured by `yuchen_v2.py --stage cv`)

| Model | LCO test-weighted AUC | non-blank weighted AUC | pooled AUC |
|---|---:|---:|---:|
| v1 features (90) + LR, CatBoost | 0.99637 | 0.99090 | 0.99486 |
| v2 features (126) + LR, CatBoost | 0.99639 | 0.99127 | 0.99487 |
| **v2 + LR + test self-training** | **0.99749** | **0.99369** | **0.99656** |

Earlier LightGBM screening (same LCO): style 53 -> 0.9851, +v1 -> 0.9899, +v2 -> 0.9918
(no LR); with LR 0.9936 (53) -> 0.9951 (126) -> 0.9967 (+ self-training).
With CatBoost the v2 features add little once the LR score is present; most of the
gain comes from self-training. LCO is still optimistic (train cluster 4 is mostly
wiki-style text, not poems), so only the leaderboard can confirm.

## Run

```bash
pip install -r submissions/Group_Number_19/requirements-yuchen.lock.txt   # now includes wordfreq
python submissions/Group_Number_19/yuchen_v2.py --stage features
python submissions/Group_Number_19/yuchen_v2.py --stage lr        # ~15 min
python submissions/Group_Number_19/yuchen_v2.py --stage predict   # writes the two submissions
python submissions/Group_Number_19/yuchen_v2.py --stage cv        # optional, LCO table above
```

Outputs in `results/yuchen/v2_shift/`: `submission_v2_cat.csv` (inductive) and
`submission_v2_cat_pseudo.csv` (with test self-training), `lco_results.csv`,
per-row LCO predictions, `predict_info.json`.

Note for the report: self-training on test text is transductive. It doesn't use
labels or external models, but confirm with course staff that it is allowed before
you pick it as a final submission.

AI assistance (v2): Claude (Cowork) diagnosed the shift, implemented v2 features and
runner, and ran the experiments; results above are copied from the run output.

## v3: frozen GPT-2 token-probability features (`yuchen_v3.py`)

Added zero-shot detector signals from two frozen causal LMs, gpt2 (124M) and gpt2-medium
(355M): per-token log-prob mean/std/percentiles, entropy, Fast-DetectGPT analytic score,
single-model Binoculars-style ratio, rank shares (`yuchen_lm_features.py`, 15 per model),
plus 5 cross-model ratios. No fine-tuning and no labels from the LM. Computed on one RTX
2080 Ti on Unicorn (`gpt2_feats.sbatch`, `BS=8`, ~18 min for both models, all splits).

Individually the LM features are weak (best: `lm_logp_std` 0.837 weighted AUC; perplexity
and Fast-DetectGPT ~0.72 and ~0.52 on no-blank-line text), but CatBoost combines them well:

| Model (same LCO proxy as v2) | test-weighted AUC | non-blank weighted AUC | pooled AUC |
|---|---|---|---|
| LM features only (35) | 0.98553 | 0.98479 | 0.97777 |
| v2 + LR (rerun, this env) | 0.99617 | 0.99090 | 0.99499 |
| v2 + LM + LR | 0.99884 | 0.99703 | 0.99793 |
| **v2 + LM + LR + test self-training** | **0.99922** | **0.99804** | **0.99864** |

```bash
sbatch -p lil --gres=gpu:1 -c 4 --mem=24G gpt2_feats.sbatch   # on Unicorn, from ~/hai; BS=8 on 11 GB cards
# copy out/{gpt2,gpt2-medium} to results/yuchen/lm_features/, then:
python submissions/Group_Number_19/yuchen_v3.py --stage cv
python submissions/Group_Number_19/yuchen_v3.py --stage predict
```

Outputs in `results/yuchen/v3_lm/`: `submission_v3_cat.csv`, `submission_v3_cat_pseudo.csv`.
Note for the report: GPT-2 is an external pretrained model used only as a frozen feature
extractor (no generation, no labeling). Confirm with course staff that this is allowed.

Public LB: `submission_v3_cat_pseudo.csv` = **0.98869** (v2_cat_pseudo was 0.97880, team baseline 0.97543).

## v4: + Qwen3.5-2B Binoculars features, and learning from unlabeled data

**Binoculars features** (`yuchen_bino_features.py`, `bino_feats.sbatch`): observer
Qwen3.5-2B-Base, performer Qwen3.5-2B (instruct), same tokenizer. Per text: Binoculars
score log-PPL_obs / X-PPL(perf->obs), cross-model Fast-DetectGPT, per-model perplexity /
entropy / rank stats, instruct-vs-base log-prob gap (31 features). Frozen open-weight
models used locally as feature extractors (no commercial API, no generation, no labeling).
One RTX A6000, bf16, ~11 min for train (torch built for CUDA 12.8; cu13x wheels fail on
nodes with a 12.8 driver).

Zero-shot the Qwen Binoculars score is weak here (weighted AUC 0.689, ~0.50 on text
without blank lines), weaker than GPT-2's `logp_std` (0.837). Inside CatBoost it still helps:

| Model (LCO, 1 seed) | test-weighted AUC | non-blank weighted AUC | pooled AUC |
|---|---|---|---|
| Binoculars features only | 0.96598 | 0.97243 | 0.96394 |
| GPT-2 + Binoculars features only | 0.99216 | 0.99073 | 0.98885 |
| v3: v2 + GPT-2 + LR | 0.99884 | 0.99703 | 0.99793 |
| v4: v3 + Binoculars | 0.99906 | 0.99757 | 0.99855 |
| v3 + self-training (target) | 0.99922 | 0.99804 | 0.99864 |
| **v4 + self-training (target)** | **0.99939** | **0.99842** | **0.99903** |

**Learning from unlabeled data** (`yuchen_semi.py --stage semi`, v3 features): one round
of self-training, pseudo-labels for P>0.95 / P<0.05, pool varied:

| Pseudo-label pool | test-weighted AUC | non-blank | mean rows pseudo-labeled |
|---|---|---|---|
| none (supervised) | 0.99884 | 0.99703 | 0 |
| unlabeled.csv | 0.99897 | 0.99737 | 2851 |
| target texts (held-out cluster / test.csv) | 0.99922 | 0.99804 | 1138 |
| unlabeled.csv + target | 0.99923 | 0.99802 | 3989 |

Why unlabeled.csv helps little: it looks like train, not test. A TF-IDF classifier
trained to separate train from test texts (5-fold AUC 0.738) calls 6.8% of train, 44.6%
of test and only 11.1% of unlabeled rows "test-like". Self-training only helps where it
adds the shifted domain, which only the target texts provide.

**Error analysis** (v4 + self-training, LCO OOF, `results/yuchen/v4_bino/error_analysis_oof.csv`):
- Blank lines are almost a label in train (P(AI | blank line) = 0.995 vs 0.254 without).
  AUC is 0.99986 on texts with a blank line and 0.99722 without; 103 of the 107 errors at
  threshold 0.5 are on texts without blank lines. Error rate is highest for short texts
  without blank lines (<100 words: 6.3%) and for long ones (>300 words: 2.9%).
- 69 false negatives (AI called human) vs 38 false positives. The confident misses are
  AI text written in Wikipedia-biography / encyclopedic style (cluster 3, "born,
  university, american" -- the weakest cluster, AUC 0.99732) and dry technical
  abstracts; the confident false positives are short, enthusiastic human movie reviews.

Public LB: `submission_v4_unl_target.csv` (v4 features, self-training on unlabeled.csv +
test) = **0.99146** (v3 0.98869, v2 0.97880, team baseline 0.97543).

## Effect of training-set size (`yuchen_semi.py --stage curve`)

Each LCO training fold is subsampled (stratified) to a fraction of its ~8,750 texts; the
TF-IDF/LR is refit on the subsample; 2 repeats below 100%. Test-weighted LCO AUC
(non-blank-line subset in parentheses). Plot: `results/yuchen/semi/learning_curve.png`.

| train texts | TF-IDF + LR | style feats (CatBoost) | GPT-2 LM feats (CatBoost) | combined v3 |
|---|---|---|---|---|
| 438 (5%) | 0.782 (0.852) | 0.964 (0.912) | 0.931 (0.906) | 0.9807 (0.954) |
| 875 (10%) | 0.829 (0.896) | 0.974 (0.932) | 0.971 (0.963) | 0.9907 (0.977) |
| 2,188 (25%) | 0.870 (0.929) | 0.986 (0.964) | 0.983 (0.981) | 0.9972 (0.994) |
| 4,375 (50%) | 0.900 (0.945) | 0.990 (0.973) | 0.985 (0.984) | 0.9985 (0.996) |
| 8,750 (100%) | 0.916 (0.953) | 0.992 (0.979) | 0.986 (0.985) | 0.9988 (0.997) |

- TF-IDF + LR is the most data-hungry and the weakest under topic shift (it learns topic
  words); it is still rising at 100%.
- LM features saturate by ~25% of the data (they measure predictability, not topic), and
  on texts without blank lines they beat the style features from 10% of the data on.
- The families are complementary: the combined model with 25% of the data (0.9972) beats
  every single family trained on all of it.

## Fine-tuned transformer (RoBERTa-base) and blend (`yuchen_deberta.py`, `yuchen_blend.py`)

End-to-end fine-tuning of a pretrained encoder, same LCO protocol (8 fits on 7 clusters,
predict the held-out one) + one fit on all data for test. 2 epochs, lr 2e-5, batch 8 x 2
accumulation, max 512 tokens, fp32, one RTX 2080 Ti per job (~9 min per fit).
DeBERTa-v3-base was tried first: with transformers 5.22 its loss became NaN at the second
optimizer step in both bf16 and fp32 (first loss ~0.70), so it was dropped for
FacebookAI/roberta-base, whose byte-level BPE also keeps newlines as tokens (no marker needed).

RoBERTa alone: LCO test-weighted AUC 0.99520 (non-blank 0.98927) -- below v4 CatBoost
(0.99939) and below every CatBoost variant with LM features, despite seeing raw text.
Per held-out cluster: 0.992 recipes, 0.997 science, 0.994 forum, 0.993 bios, 0.999
poetry, 1.000 CV papers, 0.986 movie reviews, 0.996 news. On test its ranking agrees
only moderately with v4 (Spearman 0.857), i.e. it makes different mistakes.

Rank-average blend with v4 (weight = RoBERTa share), LCO:

| w | test-weighted AUC | non-blank | pooled |
|---|---|---|---|
| 0.0 (v4 alone) | 0.99939 | 0.99842 | 0.99903 |
| **0.1** | **0.99953** | **0.99879** | **0.99915** |
| 0.2 | 0.99945 | 0.99861 | 0.99899 |
| 0.3 | 0.99932 | 0.99831 | 0.99875 |
| 0.5 | 0.99886 | 0.99721 | 0.99794 |
| 1.0 (RoBERTa alone) | 0.99520 | 0.98927 | 0.99212 |

A small RoBERTa share helps (w = 0.1); w was picked on the same OOF, so the gain is
slightly optimistic. Submission: `results/yuchen/blend/submission_blend_w0.1.csv`
(v4_unl_target ranks x 0.9 + RoBERTa full-fit ranks x 0.1; Spearman 0.9989 with v4).
Public LB: **0.99162** vs 0.99146 for v4 alone; this blend is the final submission.

## Classifier comparison on the same v4 features (`yuchen_models.py --stage cv`)

Same 193 inputs (126 style + LR score + 35 GPT-2 + 31 Qwen Binoculars), same LCO folds,
no self-training, 1 seed. Only the learner changes.

| Classifier | test-weighted AUC | non-blank | pooled | held-out poetry cluster (c4) |
|---|---|---|---|---|
| Logistic regression (standardized) | 0.99841 | 0.99626 | 0.99754 | - |
| MLP 256-128 (quantile-normalized) | 0.99886 | 0.99705 | 0.99827 | 1.0000 |
| Random forest (600 trees) | 0.99795 | 0.99477 | 0.99746 | - |
| Extra trees (800 trees) | 0.99843 | 0.99602 | 0.99794 | - |
| LightGBM (early-stopped) | 0.99399 | 0.98993 | 0.99797 | 0.9976 |
| **CatBoost (v4)** | **0.99906** | **0.99757** | **0.99855** | **1.0000** |

- CatBoost is the best single learner; an MLP on quantile-normalized features is a close second.
- LightGBM is as good as CatBoost on the in-distribution clusters (pooled 0.99797) but drops
  on the held-out poetry cluster (0.9976 vs 1.0); because poetry is 36% of test, the
  test-weighted score falls to 0.994. CatBoost's symmetric (oblivious) trees extrapolate
  more smoothly to a genre missing from training than LightGBM's leaf-wise trees.
- Linear LR is already 0.998 on these features: most of the gain is in the features, not the learner.
- Rank-average blends: best pair MLP + CatBoost 0.99936 (vs 0.99906). Adding to v4 with
  self-training: v4 0.6 + MLP 0.3 + RoBERTa 0.1 = 0.99956 vs v4 + RoBERTa 0.1 = 0.99953 vs
  v4 0.99939 -- differences of 3e-5 are within noise, so the MLP is not added to the submission.

## Report protocol: fixed 2,000-row validation set (`yuchen_report.py`)

As required by the report template: validation = fold 0 of the team split (`folds.py`,
SEED 20261109; 2,000 rows, 1,000 per class, the same rows as v1); training partition = the
other 8,000 rows. Hyperparameters and early stopping use an inner 20% split of the
training partition only. Outputs: `results/yuchen/report/`.

**Q1.2 Model comparison**

| Model | Representation | Train AUROC | Val AUROC | Train time |
|---|---|---|---|---|
| TF-IDF + logistic regression (C=30) | word 1-3 + char 3-5 n-gram TF-IDF | 1.0000 | 0.97149 | ~15 s |
| LightGBM (team baseline) | 53 style features + LR score | 0.99993 | 0.99926 | 3 s |
| CatBoost (v2) | 126 style features + LR score | 0.99988 | 0.99874 | 3 s |
| CatBoost (v4) | 126 style + LR + 35 GPT-2 + 31 Qwen3.5 | 0.99997 | 0.99925 | 6 s |
| MLP 256-128 | same 193 inputs, quantile-normalized | 0.99996 | 0.99900 | 1 s |

(The LM features themselves cost ~18 min (GPT-2) and ~15 min (Qwen) of GPU time once.)
On a random in-distribution split every feature-based model is at 0.999+ and the team
LightGBM ties v4 CatBoost, yet on the public leaderboard they score 0.9754 vs 0.9915. The
random split cannot see the test genre shift; that is why model selection used the
leave-cluster-out (LCO) protocol above, where v4 CatBoost leads (0.99906 vs 0.99399 for LightGBM).

**Q1.3 Hyperparameters** (selected on the inner split; validation shown only for reference)

| Model | Grid | Best on inner split | Inner AUROC | Val AUROC |
|---|---|---|---|---|
| CatBoost (v4 features) | depth {4, 6, 8} x learning rate {0.03, 0.05, 0.1}, early stopping (150 rounds) | depth 4, lr 0.1 | 0.99925 | 0.99928 |
| MLP (v4 features) | hidden {(128), (256,128)} x L2 alpha {1e-4, 1e-3, 1e-2} | (256,128), alpha 1e-4 | 0.99953 | 0.99949 |

All 9 CatBoost settings fall within 0.9985-0.9992 inner AUROC, so the model is insensitive
to them; the final model keeps depth 6 / lr 0.05, which was chosen earlier under LCO.

**Q2.1 Training-set size** (CatBoost, v4 features; TF-IDF/LR refit on each subsample;
plot `results/yuchen/report/learning_curve_fixed_val.png`)

| Training rows | Train AUROC | Val AUROC | TF-IDF + LR val AUROC |
|---|---|---|---|
| 500 | 0.99798 | 0.98873 | 0.91045 |
| 1,000 | 0.99877 | 0.99527 | 0.93464 |
| 2,000 | 0.99998 | 0.99883 | 0.95191 |
| 5,000 | 0.99998 | 0.99903 | 0.96411 |
| 8,000 | 0.99997 | 0.99925 | 0.97149 |

The train/validation gap closes from 0.009 (500 rows) to 0.0007 (8,000): mild overfitting
at small sizes, saturation after ~2,000 rows. TF-IDF + LR keeps rising, so it would benefit
from more labels; the feature model would not, in distribution.

**Q3 Unlabeled data** (self-training on `unlabeled.csv`; validation rows are never in
the teacher's training data or in the pseudo-label pool)

| Method | Val AUROC | Pseudo-labeled rows | Teacher accuracy on equally confident val rows |
|---|---|---|---|
| Supervised baseline | 0.99925 | 0 | - |
| Self-training, threshold 0.90 | 0.99906 | 2,913 / 3,000 | 99.80% |
| Self-training, threshold 0.95 | 0.99904 | 2,878 / 3,000 | 99.85% |
| Self-training, threshold 0.98 | 0.99895 | 2,817 / 3,000 | 99.84% |

It did not help in distribution (-0.0002): the confident pseudo-labels are ~99.8% right
(so ~5 wrong labels), but `unlabeled.csv` looks like train (adversarial check above),
so they repeat what the labeled data already says. Under LCO, where the held-out genre is
new, unlabeled self-training helped slightly (0.99884 -> 0.99897) and self-training on
the target texts helped more (-> 0.99922).

**Q4 Final model** (v4 features, CatBoost 3 seeds, self-training on unlabeled + test texts,
refit on the 8,000-row partition): train AUROC 0.99999, val AUROC 0.99909. At threshold 0.5:
3 false positives, 12 false negatives.

| Subgroup | n | AI share | Val AUROC | Errors at 0.5 |
|---|---|---|---|---|
| < 150 words | 384 | 0.59 | 0.99978 | 3 |
| 150-250 words | 1,033 | 0.43 | 0.99955 | 5 |
| > 250 words | 583 | 0.57 | 0.99817 | 7 |
| no blank line | 1,324 | 0.25 | 0.99728 | 14 |
| has blank line | 676 | 0.99 | 1.00000 | 1 |

Errors (`q4_errors.csv`; only 3 FPs exceed 0.5, so the next-highest-scored human texts are listed):

| ID | Type | P(AI) | Hypothesis |
|---|---|---|---|
| 3326 | FP | 0.850 | Polished, superlative movie review ("Academy Award winning war epic") reads like LLM prose |
| 46 | FP | 0.759 | Wikipedia biography lead with dates and alternate spellings, a format AI bios in train imitate |
| 8218 | FP | 0.506 | Human news text that contains a blank line, the strongest AI cue in train |
| 3199 | FP (0.46) | 0.465 | Forum help request with very regular sentences |
| 2290 | FP (0.43) | 0.431 | Short, encyclopedic military-unit stub |
| 5010 | FN | 0.001 | AI Wikipedia-style actress bio, incl. a citation artifact ("California1") |
| 1810 | FN | 0.001 | AI infobox-like list ("Dates of existence:", "Capital city") with no prose cues |
| 5401 | FN | 0.002 | AI news story in plain BBC style, no blank lines |
| 8444 | FN | 0.044 | AI Wikipedia-style politician bio |
| 9767 | FN | 0.097 | AI casual tech-forum post with slang ("gone AWOL") |
| 2857 | FN | 0.173 | AI math abstract in dense technical register |

Observed pattern: 14 of 15 errors are on texts without a blank line; false negatives are
mostly AI text that imitates encyclopedic or news formats (factual, low-emotion, list-like),
false positives are polished or formulaic human text. Speculation: the model has learned
"plain encyclopedic register = human" from Wikipedia/news human text in train. A fix to try:
genre-matched negatives (AI-written bios/news) or features measuring factual density
rather than style.

## CatBoost depth / learning rate under LCO (`yuchen_lco_hparams.py`)

The fixed-split grid (report Q1.3) was flat and slightly favoured depth 4 / lr 0.1, so the
candidates were re-checked under LCO with the v4 features (1 seed; self-training pool =
`unlabeled.csv` + held-out cluster texts, as in the final recipe):

| depth | lr | LCO weighted AUC | non-blank | + self-training: weighted | non-blank |
|---|---|---|---|---|---|
| **6** | **0.05** (final) | 0.99906 | 0.99757 | 0.99932 | 0.99823 |
| 4 | 0.1 | 0.99901 | 0.99746 | 0.99930 | 0.99819 |
| 4 | 0.05 | 0.99894 | 0.99727 | - | - |
| 8 | 0.05 | 0.99888 | 0.99714 | - | - |
| 6 | 0.1 | 0.99919 | 0.99791 | 0.99939 | 0.99844 |

Depth 4 / lr 0.1 is not better under LCO. All settings lie within 0.0003 (0.0001 with
self-training), about the size of seed-to-seed variation, so the final model keeps
depth 6 / lr 0.05 rather than re-tuning on noise.

### Final blend on the fixed split (`yuchen_deberta.py --fixed-split folds.csv`, `yuchen_report.py --stage q4blend`)

RoBERTa-base trained on the same 8,000 rows (same settings as the LCO runs, one RTX 2080
Ti, 9 min): train AUROC 0.99951, val AUROC **0.99928**, the best single model on this
random split (CatBoost final recipe 0.99909), but only 0.99520 under LCO. It is strong on
familiar genres and weaker on held-out ones, which is why the blend gives it weight 0.1.

Final blend (0.9 x CatBoost final recipe + 0.1 x RoBERTa, ranks within each set):
train AUROC 0.99998, val AUROC **0.99928**. The blend output is a rank score, so errors use
the threshold that labels the same share AI as CatBoost at P=0.5 (score 0.5055): 1 FP, 10 FN.

| Subgroup | n | Blend AUROC | CatBoost alone | Errors |
|---|---|---|---|---|
| < 150 words | 384 | 0.99989 | 0.99978 | 3 |
| 150-250 words | 1,033 | 0.99964 | 0.99955 | 3 |
| > 250 words | 583 | 0.99869 | 0.99817 | 5 |
| no blank line | 1,324 | 0.99785 | 0.99728 | 11 |
| has blank line | 676 | 1.00000 | 1.00000 | 0 |

The components fail on different passages: RoBERTa gives P(AI) > 0.89 to the AI
biographies/lists 5010, 8444, 1810 that CatBoost scores ~0, while CatBoost is right on the
human list page 8925 where RoBERTa is wrong. Only 5401, 3721 (FN) and 3326, 8218 (FP) fool
both. At weight 0.1 RoBERTa cannot overturn a confident CatBoost score; a stacker that
learns when to trust it is a possible next step. Files: `results/yuchen/report/q4b_*`.

## Is GPT-2 redundant with Qwen3.5? (LCO ablation)

| Features (CatBoost, + TF-IDF/LR score) | LCO weighted | non-blank | + self-training |
|---|---|---|---|
| v2 style only | 0.99617 | 0.99090 | - |
| v2 + Qwen3.5 (no GPT-2) | 0.99860 | 0.99638 | 0.99898 |
| v2 + GPT-2 (no Qwen3.5) | 0.99884 | 0.99703 | - |
| v2 + GPT-2 + Qwen3.5 (v4) | 0.99906 | 0.99757 | 0.99939 |

No: dropping GPT-2 raises the remaining error by about two thirds, and GPT-2 alone is
slightly stronger than the Qwen features. Both measure predictability, but the two model
families err on different passages. Results: `results/yuchen/v4_bino/ablation_gpt2_vs_qwen.csv`.
