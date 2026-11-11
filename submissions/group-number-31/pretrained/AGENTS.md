# AGENTS.md

Context for AI assistants (and humans) working in this repository: the problem,
how the code is laid out, how to run things, and what has been learned so far.
It is not a set of instructions about *how* to solve the problem — any approach
is welcome.

## The task

Kaggle competition **Cornell Tech CS5785 (Fall 2026) — AI vs Human text**.
Binary classification: predict the probability that a passage is AI-generated.

- Metric: **AUROC** on 3,000 test passages (public LB = 30%, private LB = 70%, final ranking uses private).
- Data (`data/`, not in git — download from Kaggle):
  `train.csv` (10,000 labeled, 50/50; `label` 1 = AI, 0 = human), `unlabeled.csv` (3,000),
  `test.csv` (3,000), `train_starter_2000.csv` (subset of train), `sample_submission.csv`.
- Passages are 75–350 words (≈210 avg) across many domains (recipes, abstracts, news, wiki bios, poems, reddit, …).
- The test set contains **writing styles, domains and generators not present in train**, so generalisation matters.
- Submission: `id,label` with exactly 3,000 rows, label in [0, 1]. Limit: 5 submissions/team/day. Deadline Nov 11, 2026.
- Course grade: 25% leaderboard; the rest is methodology — model comparison, effect of training-set size,
  use of unlabeled data, generalisation/error analysis, open-ended improvements, and a clean GitHub PR history.

## Machines

| | MacBook Pro (M1 Pro, 16 GB) | Desktop PC (Ryzen 5 9600X, 32 GB) |
|---|---|---|
| Accelerator | Apple GPU via MPS | RTX 5070 12 GB + RTX 3060 12 GB (CUDA, bf16) |
| Env | conda `cs-aml` | conda `cs-aml` with a CUDA 12.8+ PyTorch build (RTX 50xx needs it) |
| Profile | `python run_all.py --profile mac` | `python run_all.py --profile pc` |

- Torch device is one setting: `device:` in `configs/default.yaml` (`auto` | `cuda` | `cuda:<i>` | `mps` | `cpu`),
  `--device` on train/predict, or env var `AIVH_DEVICE`. Priority: `--device` / non-auto config > `AIVH_DEVICE` >
  auto-detect (cuda → mps → cpu). An unavailable device fails immediately. Saved models don't store the device.
- `python scripts/check_setup.py` verifies packages, data and GPUs (incl. whether the PyTorch build supports the card);
  `--models` also loads every pretrained model once.
- Pretrained weights download on first use into the Hugging Face cache; `scripts/download_models.py` fetches them up
  front (`--local` keeps a copy in `pretrained/`, which the code then prefers). `run_all.py` prefetches automatically.

## Directory layout

```
AGENTS.md / CLAUDE.md     this file (CLAUDE.md just imports it for Claude Code)
README.md                 setup + usage for humans
requirements.txt          pip deps; environment.yml creates the cs-aml conda env
train.py                  CV + test predictions + submission for ONE model config (entry point)
run_all.py                runs every job of an experiment profile in parallel (GPUs + CPU) and builds a report
predict.py                re-predict any CSV from a saved run
make_submission.py        Kaggle file from one or more saved runs (re-predicts test, OOF-CDF blend, validates)
configs/
  default.yaml            shared settings: data, device, preprocessing, CV, semi-supervised, final strategy, outputs
  models/<config>.yaml    one file per model config (see table below); --model <config> selects it
  experiments/<profile>.yaml  job lists for run_all.py: blends (the members of the uploaded blends),
                          fixed (every blend member on the team's fixed 8k/2k split, for the report)
src/
  config.py               loads default.yaml <- models/<config>.yaml <- --set key=value overrides
  data.py                 CSV loading (external labelled data refused: rule 4(b)), CV splits (stratified / cluster / holdout), subsampling
  preprocessing.py        optional text-cleaning steps (each toggled in config)
  features.py             TfidfFeaturizer (word/char/subword) and stylometric features (base + extended)
  models/
    base.py               BaseModel interface: fit(texts, y, ...), predict_proba(texts), save/load
    __init__.py           REGISTRY mapping model class names -> classes
    tfidf_logreg.py  tfidf_nb.py  tfidf_svm.py      TF-IDF + linear models / NB / NB-SVM
    stylometric_gbm.py    hand-crafted style features -> LightGBM / random forest / hist GB
    lm_perplexity.py      causal-LM token statistics (+ Binoculars with an observer/performer pair) -> classifier
    embedding_logreg.py   frozen sentence-embedding model -> logistic regression
    transformer.py        fine-tuned pretrained encoder (bf16 on CUDA, gradient checkpointing, length-sorted inference)
  ensemble.py             rank / OOF-CDF averaging, greedy (Caruana) selection, logistic stacking on OOF predictions
  inference.py            predict with a saved run (refit model or fold models), submission validator
  report.py               run_all report: summary table, sanity checks, ensembles, figures
  leaderboard.py          rewrites the Results table below from experiments/results.csv
  utils.py                seeding, metrics (AUROC, TPR@1%FPR, ...), results log + file lock, device resolver
scripts/
  check_setup.py          environment / GPU / data / model-loading check
  download_models.py      pre-download Hugging Face weights
  blend.py                blend chosen runs (rank / greedy / stack)
  record_kaggle_score.py  store a Kaggle public score for a run or submission
  report_experiments.py   the report's fixed-split experiments (collect, hpsearch, curve, semisup, errors, debias)
  report_tables.py        results/ -> the report's LaTeX tables and figures
notebooks/01_eda.ipynb    exploratory data analysis
experiments/results.csv   one row per training run (tracked in git)
experiments/sessions/<session>/   run_all reports: report.md, summary.csv, ensembles.csv, sanity.csv, figures/ (tracked)
data/  outputs/  pretrained/      git-ignored: data, run artefacts/logs/submissions/caches, local model copies
```

## Model configs (`configs/models/`)

| config | what | hardware |
|---|---|---|
| `tfidf_logreg`, `tfidf_svm`, `tfidf_nb` | word+char TF-IDF → LR / linear SVM / NB-SVM | CPU |
| `stylometric_gbm` | 43 style/formatting features → LightGBM | CPU |
| `stylometric_gbm_ext` | + function words, approximate part-of-speech rates, readability, coherence (Ma et al. 2023) | CPU |
| `lm_gpt2`, `lm_gpt2_medium` | GPT-2 token log-prob/rank/entropy stats + style → LightGBM | GPU or MPS |
| `binoculars_qwen05b`, `binoculars_qwen15b` | Binoculars (Qwen2.5 base + instruct) + token stats + style → LightGBM | MPS ok for 0.5B; 1.5B GPU |
| `binoculars_qwen15b_zeroshot` | raw Binoculars score, no training (shares the feature cache) | GPU |
| `embed_bge_base`, `embed_e5_base` | frozen sentence embeddings → LR | GPU or MPS |
| `ft_distilroberta`, `ft_roberta_base`, `ft_electra_base`, `ft_modernbert_base`, `ft_deberta_v3_small`, `ft_deberta_v3_base` | fine-tuned encoders | GPU; small ones possible on MPS |
| `ft_openai_detector`, `ft_openai_detector_large` | fine-tune OpenAI's RoBERTa GPT-2-output detector (as in Ma et al.) | GPU (large: 12 GB) |
| `ft_deberta_v3_large` | DeBERTa-v3-large, bf16 + gradient checkpointing | GPU 12 GB |

Feature-extractor models (LM, embeddings) cache features in `outputs/cache/` keyed by text hash.
Fine-tunes use `final.strategy: fold_average` (test prediction = mean of the CV fold models).

## How a single run works

```
python train.py --model <config> [--set key=value ...] [--device auto|cuda|mps|cpu] [--quick] [--cv-only] [--notes "..."]
```

1. Config = `configs/default.yaml` ← `configs/models/<config>.yaml` ← `--set` overrides (values parsed as YAML).
2. All splits are preprocessed with the same `preprocessing:` steps.
3. Cross-validation on `train.csv` per `cv.scheme`: `stratified` (k-fold), `cluster` (k-means topic clusters held
   out as whole groups — approximates unseen domains, lower scores, not comparable to stratified), `holdout`.
   Pseudo-labels only ever enter the training side of a fold; `data.train_frac` subsamples it. Supervised labels
   come only from `train.csv`: `data.external_files` must stay empty (competition rule 4(b)).
4. Test predictions: `final.strategy` = `refit` (fit on all labeled data) or `fold_average`.
5. Outputs: `outputs/runs/<run_name>/` (config.yaml, oof.csv, metrics.json, test_pred.csv, unlabeled_pred.csv,
   status.json, model), `outputs/submissions/<run_name>.csv`; a row in `experiments/results.csv`; the table below.
6. Crash safety: each finished fold's predictions go to `outputs/runs/<run_name>/folds/` (keyed by a config + split
   fingerprint), and fine-tunes (`checkpoint_every_epoch: true`) save `outputs/checkpoints/<run_name>/fold_<k>/last.pt`
   after every epoch (model, optimizer, scheduler, scaler, RNG). Re-running with the same run name
   (`run_all --resume`, or `train.py --run-name <same>`) skips finished folds and continues the current fold after its
   last finished epoch. Checkpoints are deleted once a fold's predictions are saved (`output.keep_checkpoints=true`
   keeps them). `metrics.json` has `epoch_history` (train loss / fold-val AUROC per epoch, logged only — epochs stay
   fixed) and `epoch_val_auc_mean`.

Adding a model = a class in `src/models/` subclassing `BaseModel` + one line in `REGISTRY`, and/or a new
YAML in `configs/models/` (e.g. a different Hugging Face checkpoint for an existing class).

## Running everything (`run_all.py`)

```
python run_all.py --profile smoke    # every config on 2k texts / 2 folds / 1 epoch: catches setup errors
python run_all.py --profile pc       # full benchmark, both GPUs + CPU in parallel
python run_all.py --profile mac
python run_all.py --profile pc --only <job>,<job> | --skip <job> | --set cv.n_folds=3 | --gpus 0
python run_all.py --resume outputs/sessions/<session>     # continue after a crash / Ctrl+C
python run_all.py --report outputs/sessions/<session>     # rebuild the report
```

Each job is a `train.py` subprocess (log: `outputs/sessions/<session>/logs/<job>.log`), pinned to one GPU with
`CUDA_VISIBLE_DEVICES`; heaviest jobs start first; idle GPU workers also take CPU jobs; `after:` orders dependent jobs.
The report (`experiments/sessions/<session>/report.md`) has the results table, ensembles (rank mean, greedy
selection and LR stacking — the latter two cross-validated), learning-curve / semi-supervised / robustness
comparisons, per-topic AUROC, ROC curves (incl. low-FPR), score distributions, test-prediction correlations,
calibration, cost vs accuracy, and sanity checks (PASS/WARN/FAIL). The recommended upload is written to
`outputs/submissions/<session>__best.csv`.

## Observations so far

**Final submissions (uploaded 2026-10-31; selection revised 2026-11-10).** Three OOF-CDF blends were uploaded
(`make_submission.py`): A `final_A_greedy@v1` (7 runs, weights 3 3 2 1 1 1 1), B `final_B_lm4` (lm_gpt2,
lm_gpt2_medium, binoculars_qwen05b, binoculars_qwen15b, equal weights) and C `final_C_lm4_roberta` (B +
ft_roberta_base). Kaggle public: A 0.9916, B 0.9885, C 0.9885; train CV (≈ 0.9995) cannot separate them.
- B: the four best runs by 5-fold CV in the pc benchmark below (any family), equal weights.
  **A and C were rebuilt on 2026-11-10**: `run_all.py --profile topic_cv` reruns 9
  candidates with topic-held-out CV (whole k-means clusters of train held out), then `scripts/select_blend.py`
  (rule pre-registered in `configs/selection/blends.yaml`): A = greedy selection on the topic-held-out OOF
  (qwen05b ×4, distilroberta ×2, modernbert ×2, gpt2-medium, gpt2-nostyle, roberta-base, gpt2 ×1); C = B +
  ft_distilroberta (best 5-fold CV of the fine-tunes). Files `kaggle_submissions/final_{A,C}_*.csv`.
- **Test is ~15% verse** (short lines, no digits) vs 0.3% of train. Line structure is a train shortcut (human
  median 80 words/line, AI 28); `preprocessing.join_lines` exists to test it (off in every submitted model).

Data:
- Formatting differs systematically by class in train: blank lines (`\n\n`) appear in ~66% of AI passages and almost no
  human ones; human recipes often have fraction slashes stripped ("14 cup" for "1/4 cup"); double spaces are more common in
  human text. Test's blank-line rate (~32%) is consistent with the same pattern, but whether these cues hold for the unseen
  test domains/generators is unknown. The `preprocessing:` steps (and the `debiased_*` jobs) exist to test this —
  GPTZero (Adam et al. 2026) report that such formatting shortcuts in naive datasets hurt real-world robustness.
- **Topic shift:** with k-means (k=10) on TF-IDF over train+unlabeled+test, one cluster of creative writing
  (poems/stories: "love, life, world, day…") is ~41% of test but only ~6.5% of train. Unlabeled.csv matches train's topic
  mix, not test's. Adversarial validation (train vs test) AUROC ≈ 0.74.
- Some AI passages contain zero-width characters / Cyrillic homoglyphs, suggesting adversarial perturbations.
- AI-side cues: assistant openers ("Sure, here's…"), "Title:/Ingredients:" headers, heavy repetition (typical of small base
  models), all at different rates in test than in train → generator mix differs.

Full PC benchmark (session `20261028-145019_pc`, 35 jobs, 5-fold stratified CV unless noted; report in
`experiments/sessions/20261028-145019_pc/report.md`):
- **Train is close to saturated.** Every reasonable model is ≥ 0.990 CV AUROC, so differences between the top ~10 are
  within noise (fold std ≈ 0.0005). The real question is generalisation to unseen test domains/generators, which train
  CV can't measure (`cluster` CV, whole topics held out, is the label-safe estimate of that).
- **Winning families:** LM token statistics + style features → LightGBM (`lm_gpt2` 0.9993, `lm_gpt2_medium` /
  `binoculars_qwen05b` / `binoculars_qwen15b` 0.9992, 2–20 min); compact fine-tunes (`ft_distilroberta` 0.9991,
  `ft_modernbert_base` 0.9990, `ft_roberta_base` 0.9987, 12–33 min); stylometric GBM ext 0.9984 in < 1 min.
  Bigger LMs for features give nothing over GPT-2 small on train.
- **Underperformers:** the DeBERTa-v3 family (small/base/large 0.990/0.992/0.995) and ELECTRA (0.990) are worse *and*
  badly calibrated (log loss 0.13–0.46 vs 0.03–0.05) despite costing the most (large: 87 min) — likely under-tuned
  at lr 2e-5 / 3 epochs. `ft_openai_detector` (base) has 0.990 AUROC but TPR@1%FPR only 0.63. TF-IDF ≈ 0.95–0.96,
  frozen embeddings + LR 0.87 (BGE) / 0.92 (E5). Zero-shot Binoculars is only 0.71 — the train generators/attacks
  defeat it, so it is only useful as one feature among many (which is how `binoculars_*` use it).
- **Per-topic (train k-means clusters):** strong models are ≥ 0.996 in every cluster. Weak models fail on news
  ("said, year, film"), bios ("born, university") and the "new, said, used" cluster (TF-IDF ≈ 0.89–0.93, embeddings
  down to 0.78) and are near-perfect on recipes. Train has almost no poetry, so per-topic AUROC can't tell us how
  any model does on test's poems/stories.
- **Topic generalisation (cluster CV):** stylometric ext drops only 0.9984 → 0.9946, TF-IDF LR 0.9576 → 0.9136
  (fold std 0.033): content features don't transfer across topics, style features do.
- **Debiasing (strip blank lines / double spaces / quotes / zero-width / homoglyphs):** costs stylometric ext 0.0023 AUROC
  and TPR@1%FPR 0.981 → 0.942, but leaves DeBERTa-base unchanged (0.9919 → 0.9925) — the fine-tune doesn't lean on
  formatting. Debiased stylometric's mean test score rises 0.458 → 0.522, i.e. formatting cues push many test passages
  towards "human"; whether that is right is exactly the unknown (test blank-line rate 32% suggests the cue still holds).
- **Training-set size:** stylometric is saturated (10% of train → 0.9944, 50% → 0.9977); TF-IDF keeps improving
  (0.906 → 0.958 from 10% to 100%); DeBERTa-small loses 0.008 at 25%. More *in-domain* data won't help the strong
  models, but more *diverse* data (new domains/generators) might.
- **Unlabeled data:** pseudo-labelling (unlabeled, or unlabeled + test) changes nothing (Δ ≤ 0.0007) for TF-IDF and
  stylometric — the confident pseudo-labels are the easy cases the model already gets right.
- **Model agreement on test:** LM-feature and stylometric models are highly correlated (they share the style features);
  the RoBERTa-type fine-tunes form a second group, DeBERTa/ELECTRA a third, TF-IDF a fourth → ensembles should mix
  families, not stack near-duplicates.
- **Ensembles (nested CV):** LR stacking 0.9999 (TPR@1%FPR 0.998), greedy 0.9997, rank-mean top-5 0.9996 vs best single
  0.9993. The stacker uses large negative weights (zero-shot Binoculars −0.43, stylometric_gbm −0.38) that exploit
  train-specific correlations and could flip under shift; the greedy (non-negative) blend of lm_gpt2 +
  stylometric_gbm_ext + fine-tunes is the more robust candidate.
- Run time of `ft_deberta_v3_base` shows as 0 because its folds were reused after a scheduler crash.

Models (baselines run on the Mac sandbox, default configs):
- Stylometric features + LightGBM are much stronger than TF-IDF n-grams (0.998 vs ~0.96 CV) and barely drop under
  `cluster` CV (0.994), whereas TF-IDF LR drops to ~0.91. Top features by gain: blank-line count, type-token ratio,
  burstiness, hapax ratio. Without any length/newline/spacing features stylometric GBM still reaches ~0.974.
- On the train passages in the creative-writing cluster, OOF AUROC is a bit lower for every model (TF-IDF LR 0.938, stylometric 0.993).
- Pseudo-labelling unlabeled.csv (1 round, 40% most confident) did not change TF-IDF LR `cluster` CV.
- Rank-blending TF-IDF LR + stylometric GBM (1:3) gives 0.998 OOF.

Related work notes (summaries from the team): GPTZero (Adam et al. 2026) — supervised detector, debiasing of formatting
cues, adversarial/paraphrase augmentation, reports recall at 1% FPR; weak on old LLMs (GPT-2). Ma et al. 2023 — syntax
features (POS, function words, punctuation, sentence length, perplexity) give F1 0.97 with LR; fine-tuned RoBERTa
detector relies on function words. Ponde et al. 2026 (theory) — guided paraphrasing can make AI text provably close to
human text; per-passage detection has a ceiling.

## Results

The table is regenerated automatically by `train.py`, `scripts/blend.py` and
`scripts/record_kaggle_score.py` from `experiments/results.csv`. Record Kaggle
scores with `python scripts/record_kaggle_score.py --run <run_name or submission file> --score <x>`.

<!-- LEADERBOARD:START -->
**Best Kaggle public score:** 0.9916 — `final_A_greedy@v1` (final_A_greedy@v1)

| rank | run | config | CV scheme | CV AUROC | TPR@1%FPR | Kaggle public | train_frac | semi-sup | device | time (s) | author | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `20261030-135404_ood__lm_gpt2` | lm_gpt2 | stratified | 0.9993 ± 0.0004 | 0.994 |  | 1.0 | False | NVIDIA GeForce RTX 3060 | 76 | Max Shen | run_all:ood:lm_gpt2 |
| 2 | `20261028-145019_pc__lm_gpt2` | lm_gpt2 | stratified | 0.9993 ± 0.0004 | 0.994 |  | 1.0 | False | NVIDIA GeForce RTX 5070 | 142 | Max Shen | run_all:pc:lm_gpt2 |
| 3 | `20261028-145019_pc__lm_gpt2_medium` | lm_gpt2_medium | stratified | 0.9992 ± 0.0005 | 0.993 |  | 1.0 | False | NVIDIA GeForce RTX 5070 | 238 | Max Shen | run_all:pc:lm_gpt2_medium |
| 4 | `20261030-135404_ood__binoculars_qwen05b` | binoculars_qwen05b | stratified | 0.9992 ± 0.0005 | 0.994 |  | 1.0 | False | NVIDIA GeForce RTX 3060 | 1348 | Max Shen | run_all:ood:binoculars_qwen05b |
| 5 | `20261028-145019_pc__binoculars_qwen05b` | binoculars_qwen05b | stratified | 0.9992 ± 0.0005 | 0.994 |  | 1.0 | False | NVIDIA GeForce RTX 5070 | 510 | Max Shen | run_all:pc:binoculars_qwen05b |
| 6 | `20261030-135404_ood__lm_gpt2_medium` | lm_gpt2_medium | stratified | 0.9992 ± 0.0005 | 0.993 |  | 1.0 | False | NVIDIA GeForce RTX 3060 | 104 | Max Shen | run_all:ood:lm_gpt2_medium |
| 7 | `20261030-135404_ood__binoculars_qwen15b` | binoculars_qwen15b | stratified | 0.9992 ± 0.0005 | 0.993 |  | 1.0 | False | NVIDIA GeForce RTX 3060 | 1783 | Max Shen | run_all:ood:binoculars_qwen15b |
| 8 | `20261028-145019_pc__binoculars_qwen15b` | binoculars_qwen15b | stratified | 0.9992 ± 0.0005 | 0.993 |  | 1.0 | False | NVIDIA GeForce RTX 5070 | 1192 | Max Shen | run_all:pc:binoculars_qwen15b |
| 9 | `20261028-145019_pc__ft_distilroberta` | ft_distilroberta | stratified | 0.9991 ± 0.0006 | 0.992 |  | 1.0 | False | NVIDIA GeForce RTX 5070 | 735 | Max Shen | run_all:pc:ft_distilroberta |
| 10 | `20261028-145019_pc__ft_modernbert_base` | ft_modernbert_base | stratified | 0.9990 ± 0.0004 | 0.993 |  | 1.0 | False | NVIDIA GeForce RTX 5070 | 1974 | Max Shen | run_all:pc:ft_modernbert_base |
| 11 | `20261030-135404_ood__ft_distilroberta` | ft_distilroberta | stratified | 0.9988 ± 0.0006 | 0.991 |  | 1.0 | False | NVIDIA GeForce RTX 3060 | 1646 | Max Shen | run_all:ood:ft_distilroberta |
| 12 | `20261028-145019_pc__ft_roberta_base` | ft_roberta_base | stratified | 0.9988 ± 0.0006 | 0.989 |  | 1.0 | False | NVIDIA GeForce RTX 5070 | 866 | Max Shen | run_all:pc:ft_roberta_base |

_Top 12 of 72 logged runs by CV AUROC; full log in `experiments/results.csv`. CV scores from different schemes are not directly comparable (`cluster` CV holds out whole topics and is harder than `stratified`)._
<!-- LEADERBOARD:END -->

## Keeping this file current

When you change the directory structure, add a model/script/profile, change how
training or evaluation works, or learn something notable about the data or models,
update the relevant section above and add a line to the changelog below.
New best scores appear in the Results table automatically when runs are logged;
after a run_all session, add its headline findings to "Observations so far".

## Changelog

- 2026-11-10 — Blends A and C rebuilt with topic-held-out CV (`topic_cv` profile, `scripts/select_blend.py`,
  rule in `configs/selection/blends.yaml`); B unchanged. The `blends` profile lists the members of A–C.
  `data.external_files` is refused (rule 4(b)). Run in the `cs-aml` env: base Anaconda has no LightGBM and silently
  falls back to sklearn's HistGradientBoosting.

- 2026-10-31 — Final submissions A, B, C. `preprocessing.join_lines` (tested, off), `make_submission.py`
  (saved-model re-prediction, verified identical to training-time predictions; blends via each model's OOF-score
  CDF), `src/inference.py`, fine-tunes keep fold models (`output.save_fold_models`).

- 2026-10-29 — Full PC benchmark finished (35/35 jobs; analysis under "Observations so far"). Fixed `run_all.py`
  crashing when it printed a failed job's error tail (tqdm bar characters) to a cp1252 Windows console — stdout/stderr
  are now UTF-8 with `errors="replace"`. That crash had orphaned a just-started job, and `--resume` then started a
  second copy of it on the same GPU and both stalled, so after a scheduler crash, make sure no orphaned `train.py`
  is still running before resuming. `debiased_deberta_v3_base` spilled out of the 3060's 12 GB at batch 16 (Windows
  falls back to shared memory → ~20× slower instead of OOM) and was rerun with `gradient_checkpointing=true`.

- 2026-10-28 — Per-epoch checkpointing + fold-level resume for long fine-tunes (see "How a single run works" step 6).
  Tested by killing a smoke `ft_deberta_v3_small` session mid-epoch (fold 2, epoch 2) and `--resume`-ing: fold 1 was
  skipped, fold 2 continued from epoch 1; fold AUROCs 0.9771/0.9811 vs 0.9760/0.9809 for an uninterrupted run
  (GPU nondeterminism only). Each fold is now seeded with `seed + fold`, so a fold's result doesn't depend on which
  earlier folds were skipped.

- 2026-10-28 — PC smoke run (23/23 jobs ok). Fixed: fine-tunes loaded DeBERTa-v3 in its stored fp16 dtype
  (transformers 5 default), so AdamW updated fp16 master weights and produced NaN after step 1 — weights are now
  always loaded in fp32 (autocast still runs bf16). Report shows the real exception for failed jobs; smoke profile
  no longer logs to `results.csv` (`log_results: false`); `run_all --resume` appends to job logs and accepts
  `--set` / `--only` / `--skip` for the re-run jobs; HF symlink warnings silenced on Windows.

- 2026-10-28 — Pretrained models: 9 fine-tune configs (DistilRoBERTa, RoBERTa, ELECTRA, ModernBERT, DeBERTa-v3
  small/base/large, OpenAI detector base/large), GPT-2 / GPT-2-medium perplexity features, Binoculars (Qwen2.5
  0.5B/1.5B pairs, incl. zero-shot), frozen BGE / E5 embeddings; extended stylometric features (Ma et al.).
  `run_all.py` multi-GPU scheduler with smoke/pc/mac profiles and an automatic report (tables, figures, sanity checks,
  ensembles); bf16 + gradient checkpointing; fold-averaged test predictions; TPR@1%FPR metric; results-log file lock;
  UTF-8 everywhere (Windows); `check_setup.py`, `download_models.py`, `CLAUDE.md`.
- 2026-10-28 — Configurable torch device (`device:` config key, `--device` on train/predict/generate, `AIVH_DEVICE` env var);
  fuller `.gitignore` (all of `data/` except README, model/cache files, credentials, editor files).
- 2026-10-28 — Initial baselines logged (TF-IDF LR/NB/SVM, stylometric GBM, blends); EDA findings added above.
- 2026-10-28 — Initial structure: EDA notebook, preprocessing pipeline, six models, train/predict entry points,
  CV schemes, pseudo-labelling, external-data and generation scripts, blending, auto-updated results table.
