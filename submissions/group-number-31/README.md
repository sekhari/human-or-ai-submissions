# Group Number 31 — Human or AI?

Code for the CS5785 "Human or AI?" Kaggle competition. Six models were submitted, three from each of two
tracks, and every model is scored on the same fixed 2,000-row validation split.

## Start here

| I want to... | Go to |
|---|---|
| Understand how each model is built | [`MODELS.md`](MODELS.md): pipelines of all six submitted models |
| Read the report | [`reports/overleaf/kaggle_report.tex`](reports/overleaf/kaggle_report.tex) (the one report we submit; PDF preview: `reports/kaggle_report_preview.pdf`) |
| See every Kaggle upload and its score | [`SUBMISSIONS.md`](SUBMISSIONS.md) |
| See the six models side by side | [`common/results/compare_tracks.csv`](common/results/compare_tracks.csv), [`common/figures/six_models.pdf`](common/figures/six_models.pdf) |
| Run a classical model on a laptop | [`classical/README.md`](classical/README.md) |
| Add a new model | [`classical/CONTRIBUTING.md`](classical/CONTRIBUTING.md) |
| Run the pretrained (GPU) pipeline | [`pretrained/README.md`](pretrained/README.md) |
| See what is still to do | search the repo for `[TODO` |
| Brief an AI assistant (Claude Code, Codex, ...) | [`AGENTS.md`](AGENTS.md): task, layout, models, rules, TODOs in one page |

## Folder map

```
submissions/group-number-31/
├── README.md              this file: setup, commands, seeds, AI-use disclosure
├── MODELS.md              how each of the six submitted models is built
├── AGENTS.md              one-page briefing for AI assistants and new teammates (CLAUDE.md loads it)
├── train.py predict.py    team entry points (like the course's _example_team); they call the track scripts
├── requirements.txt       both tracks (classical/requirements.txt alone for a laptop)
├── SUBMISSIONS.md         every Kaggle upload and its score (generated from submission_log.csv)
├── submission_log.csv
├── reports/               THE REPORT: overleaf/kaggle_report.tex (self-contained), build_tables.py, inline_tables.py
├── common/                shared by both tracks
│   ├── val_ids.csv          the fixed validation split (2,000 ids, seed 20261023)
│   ├── data.py metrics.py   data loading, AUROC, length/verse subgroups
│   ├── compare_tracks.py    the six-model comparison -> results/ tables/ figures/
│   └── update_submission_log.py, validate_submission.py, make_split.py, describe_splits.py
├── classical/             from-scratch models, no pretrained weights (CPU-friendly)
│   ├── models/              one file per model: tfidf_lr (C1), stylo_gbm (C2), transformer_scratch (C3), + extras
│   ├── configs/             one YAML per model, its hyperparameter grid; selection/: pre-registered design rules
│   ├── train.py predict.py select_design.py experiments.py report_tables.py
│   ├── results/             metrics + predictions per model (all report numbers come from here)
│   ├── kaggle_submissions/  one CSV per model: final_D_stylo_gbm, final_E_tfidf_lr, final_F_transformer_scratch
│   └── reports/             tables/figures used by the report (+ a longer classical-only draft)
└── pretrained/            frozen-LM features + fine-tuned encoders, blended (needs a GPU)
    ├── src/ configs/        the pipeline, imported with its history from the research repo
    ├── train.py run_all.py make_submission.py scripts/
    ├── results/             fixed-split metrics, curves, error analysis (all report numbers come from here)
    ├── kaggle_submissions/  the uploaded CSVs (blends A, B, C)
    ├── experiments/         log of every research run (72 runs, 5-fold CV, seed 42)
    └── reports/             tables/figures used by the report (+ a longer pretrained-only draft)
```

The six submitted models:

| Model (`--model` name) | Track | Code |
|---|---|---|
| Model (version) | Track | Code |
|---|---|---|
| Blend A `blend_a` (v2: weights 4 2 2 1 1 1 1 by greedy selection on topic-held-out CV) | pretrained | `pretrained/make_submission.py` |
| Blend B `blend_b` (equal weights, unchanged), Blend C `blend_c` (v2: B + DistilRoBERTa) | pretrained | same script |
| D: C2 stylometric LightGBM `stylo_gbm` (unchanged) | classical | [`classical/models/stylo_gbm.py`](classical/models/stylo_gbm.py) |
| E: C1 TF-IDF + logistic regression `tfidf_lr` (v3) | classical | [`classical/models/tfidf_lr.py`](classical/models/tfidf_lr.py) |
| F: C3 small Transformer from scratch `transformer_scratch` (v3) | classical | [`classical/models/transformer_scratch.py`](classical/models/transformer_scratch.py) |

See [`MODELS.md`](MODELS.md) for each model's pipeline and [`SUBMISSIONS.md`](SUBMISSIONS.md) for every version's
scores. A, C, E and F were rebuilt on 2026-11-10 with topic-held-out CV (§5); their earlier uploads stay in the
log as history.

Notes:

- **The report** is `reports/overleaf/kaggle_report.tex`, one self-contained file (tables inlined, the plot in
  pgfplots, references inline, no `.bib`). After any new result run `python reports/refresh.py` (cs-aml env): it
  regenerates every table, inlines them, checks the file and compiles `reports/kaggle_report_preview.pdf`. Then
  upload the `reports/overleaf/` folder to Overleaf as is; it needs nothing else.
- **AI-assistant files.**
  - [`AGENTS.md`](AGENTS.md) in this folder is the team-wide briefing.
  - `pretrained/AGENTS.md` is the research repo's log of every experiment, carried over with the pretrained track.
    `pretrained/scripts/` still update its results table, so leave it where it is.
- **Why the deep path.** The course only accepts PRs that touch `submissions/<team-name>/`, so all our files
  live in this folder, named after our Kaggle team "Group Number 31" (lowercase, hyphens for spaces, as the
  course asks).
- **`_example_team/`** next to this folder belongs to the course (its README says not to modify it), so it stays.
  Our layout follows it: `README.md`, `requirements.txt`, `train.py` and `predict.py` at the top of the team folder.
- **Branches.** There is one branch, `main`. The course PR comes from `submission/group-number-31`, a copy of
  `main`. To update it, run `git push origin main:submission/group-number-31`.

## 1. Team

- Kaggle team name: `Group Number 31` (folder and PR branch: `group-number-31`)
- Members (name, NetID): Zhenman Shen (zs476), Giorgi Samushia (gs799), Aynan Kareem (ak2896)

## 2. Environment

Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # both tracks
pip install -r classical/requirements.txt             # or: classical track only (laptop CPU is enough)
```

The pretrained track needs a 12 GB NVIDIA GPU. An RTX 50-series card needs a CUDA 12.8+ PyTorch build, installed
before the requirements; see `pretrained/README.md`.

Competition data is never committed. Put `train.csv`, `test.csv`, `unlabeled.csv` and `sample_submission.csv` in
`data/` inside this folder (git-ignored), or pass `--data-dir PATH`, or set `HOA_DATA_DIR`.

## 3. Validation protocol and train/evaluate commands

All commands run from this folder.

```bash
python common/make_split.py          # must print "OK: val_ids.csv reproduced exactly"
```

`common/val_ids.csv` holds the ids and labels of a fixed stratified 2,000-row validation set from `train.csv`:
1,000 human and 1,000 AI, made with `train_test_split` and seed `20261023`.

- Every model in both tracks trains on the other 8,000 rows and is scored on these 2,000.
- Classical hyperparameters and designs are chosen by 5-fold CV inside the 8,000 rows, never on the validation set
  or on Kaggle. Pretrained members use fixed settings.
- Classical Kaggle models are refit on all 10,000 rows. The blends' Kaggle files come from 5-fold runs over all
  10,000 rows (seed 42); blend A's weights and blend C's added encoder were chosen on those runs' out-of-fold
  predictions, which include the validation rows.

```bash
python train.py --model stylo_gbm --seed 20261023    # train on 8k, score the fixed 2k -> classical/results/stylo_gbm/
                                                     # also tfidf_lr, transformer_scratch, linear_svm, nbsvm, rf_stylo, ensemble_c123
python train.py --model blend_b                      # pretrained: retrain a blend's members, 5-fold CV (GPU, hours)
```

Report experiments:

```bash
cd classical  && python experiments.py curve | semisup | errors && python report_tables.py && cd ..
cd pretrained && python run_all.py --profile fixed && python scripts/report_experiments.py collect --session outputs/sessions/<session>
              # then: hpsearch, curve, semisup, errors, debias; then python scripts/report_tables.py
python reports/refresh.py          # all tables -> reports/overleaf/kaggle_report.tex -> preview PDF
```

## 4. Command that writes `submission.csv`

Final Kaggle selections: pretrained **blend A** (`final_A_topic_greedy.csv`, public 0.98810) and classical **C2**
(`stylo_gbm`, file D, public 0.95995). Every uploaded
file is committed (`*/kaggle_submissions/`), and its row in `SUBMISSIONS.md` links the commit that added it.

```bash
python predict.py --model blend_a --committed --out submission.csv   # the exact uploaded file, no compute
python train.py --model stylo_gbm --final                            # classical: refit on 10k (~10 s CPU)
                                                                     #   -> classical/kaggle_submissions/final_D_stylo_gbm.csv
python predict.py --model stylo_gbm --out submission.csv             #   then predict from the saved model
                                                                     #   (same two commands for tfidf_lr, transformer_scratch)
python predict.py --model blend_a --out submission.csv               # pretrained: re-predict from the saved runs
```

- Pretrained blends re-predict test with the saved 5-fold runs, which are too large for git (0.3–1.7 GB per
  fold). They live in the research repo's `outputs/runs/`, or pass `--runs-dir`.
- Run on those runs, LM-feature members reproduce exactly; fine-tuned members re-predict with bf16 GPU noise
  (Spearman ≥ 0.9999 to the committed files).
- Retraining them from scratch takes about 4 h on two 12 GB GPUs; see `MODELS.md`.

## 5. Seeds and important config

- Split, CV folds and every new experiment: `20261023`.
- The pretrained research runs behind the uploaded files used seed `42`, with fold *k* seeded `42 + k`.
- Fine-tunes and `transformer_scratch` train in bf16 on GPU, so reruns can differ in the last digits.
- **No external labelled data** (competition rule 4(b)). Supervised labels come only from `train.csv`, and nothing
  external validates or chooses a submission.
- C1 and C3 designs are chosen by topic-held-out inner CV on the 8,000 training rows (`classical/select_design.py`,
  rule pre-registered in `classical/configs/selection/`).

## 6. AI-tool disclosure

Matches Q6.3 in the report.

- **Our decisions.** We decided the overall structure (two tracks on one fixed validation split), the model
  architectures, the candidate models and the analyses to run.
- **Claude Code's role.** Claude Code (Anthropic) mainly wrote the code to those specifications and ran the
  experiments on our machines. It also helped draft report text, which we edited.
- **What we verified and changed.**
  - We read the data ourselves and changed the pipelines along the way to match what we saw.
  - We compared every local score with its Kaggle score. When they disagreed (C1 and C3 v1), we had the cause
    investigated and changed the models accordingly.
  - We made every Kaggle upload.
  - Every reported number is generated from committed results files by scripts in this folder.

Pretrained weights are used only in `pretrained/`. Appendix A of the report lists every checkpoint, how it was
used and its citation:
- submitted blends (current versions): GPT-2 and GPT-2-medium, Qwen2.5-0.5B/1.5B (base and instruct),
  DistilRoBERTa, ModernBERT-base and RoBERTa-base;
- earlier blend A (v1) only: DeBERTa-v3-large and the OpenAI RoBERTa-large GPT-2-output detector;
- explored only: DeBERTa-v3-base/small, ELECTRA, the base-size OpenAI detector, BGE and E5 embeddings.

`classical/` uses no pretrained weights. No external LLM API labelled or scored test passages, and no hidden labels
were retrieved.
