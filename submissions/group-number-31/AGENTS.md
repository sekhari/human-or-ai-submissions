# AGENTS.md — team folder for the CS5785 "Human or AI?" Kaggle project

Read this first. It is the context file for AI coding assistants (Claude Code, Codex, Cursor, ...) and for
teammates. Human-facing setup and commands are in [README.md](README.md).

## The task

- Kaggle competition `cornell-tech-cs-5785-combined-2026-ai-vs-human-text`: predict P(AI-written) for 3,000
  test passages. Metric: AUROC. Public leaderboard = 30% of test, private = 70%. 5 uploads per day per team.
- Data (never committed): `train.csv` (10,000 labelled, 50/50, `label` 1 = AI), `unlabeled.csv` (3,000),
  `test.csv` (3,000), `sample_submission.csv`. Test contains unseen domains and generators: about 17% of test is
  verse against 1.6% of train (`common/results/split_profile.csv`).
- Grading: a LaTeX report from the course template (`reports/overleaf/kaggle_report.tex`), the private Kaggle score, and
  the code plus its commit history, all submitted as **one pull request** from this fork to
  `sekhari/human-or-ai-submissions`.

## Layout

```
human-or-ai-submissions/            the team's fork of the course repo
├── README.md, STAFF.md, .github/   course files: never edit
└── submissions/
    ├── _example_team/              course example: never edit
    └── group-number-31/            EVERYTHING of ours (Kaggle team name "Group Number 31", lowercased, hyphens)
        ├── AGENTS.md               this file
        ├── README.md               setup, commands, seeds, AI-use disclosure (required by the course)
        ├── MODELS.md               how each of the six submitted models is built (pipelines, settings)
        ├── train.py, predict.py    team entry points, shaped like _example_team; they call the track scripts
        ├── requirements.txt        both tracks
        ├── SUBMISSIONS.md          every model and Kaggle upload; generated from submission_log.csv
        ├── reports/                overleaf/kaggle_report.tex, the one report we submit; build_tables.py
        ├── common/                 shared by both tracks: val split, data loading, metrics, log, comparison
        ├── classical/              from-scratch models, no pretrained weights, CPU-friendly
        └── pretrained/             pretrained-LM pipeline imported with history from the research repo
                                    (has its own AGENTS.md: the research log of 72 runs)
```

Why the deep path: the course README only allows a team's PR to touch `submissions/<team-name>/`.
Don't move our files out of the team folder.

## Two tracks, one validation protocol

All numbers in the report are measured the same way. The 2,000 ids in `common/val_ids.csv` are a stratified
split with seed 20261023. Models train on the other 8,000 rows and are scored on those 2,000. Hyperparameters are
picked by 5-fold CV inside the 8,000 rows, never on the validation rows. Final Kaggle models are refit on all
10,000 rows. `python common/make_split.py` checks that the split reproduces exactly.

| Track | Submitted models | Val AUROC | Public AUROC |
|---|---|---|---|
| pretrained | **Blend A v2 (final)**: 7 members, weights 4 2 2 1 1 1 1, greedy on topic-held-out CV | 0.99987 | 0.98810 |
| pretrained | Blend B (4 LM-feature models, equal weights) | 0.99985 | 0.98848 |
| pretrained | Blend C v2 (B + DistilRoBERTa, equal weights) | 0.99994 | 0.98724 |
| classical | **D (final)**: C2 `stylo_gbm`: 101 stylometric features → LightGBM | 0.99831 | 0.95995 |
| classical | E: C1 `tfidf_lr` v3: word + raw char + distorted-text (k=100) TF-IDF → logistic regression | 0.99670 | 0.95849 |
| classical | F: C3 `transformer_scratch` v3: lines joined, BPE + 4-layer Transformer, random init | 0.98590 | 0.90826 |

Source of these numbers: `submission_log.csv` / `SUBMISSIONS.md`. Always re-read them there; this table is a
snapshot.

One Kaggle file per model, letters A–F only (no new letters: a new version replaces its model's file). Pretrained:
`pretrained/kaggle_submissions/final_{A,B,C}_*.csv`, no date; don't rewrite an uploaded file. Classical:
`classical/kaggle_submissions/final_{D,E,F}_<model>.csv`, no date. Earlier versions stay in git history and in the
log: `final_A_greedy@v1`, `final_C_lm4_roberta`, `final_E_tfidf_lr@v1/@v2`, `final_F_transformer_scratch@v1/@v2`.

**Validation does not predict Kaggle here.** Test has unseen generators and far more verse. To compare designs
under shift, use topic-held-out inner CV (`classical/train.py --cv topic`, `classical/select_design.py`): it
holds out whole k-means topics of the 8,000 training rows and needs nothing beyond `train.csv`. See MODELS.md,
"Why C1 and C3 v1 fell on Kaggle".

**No external labelled data, for anything** (competition rule 4(b): not to train, validate or choose models).

### pretrained/ — how the blends work

There are two kinds of member models.

1. **Frozen LM as a feature extractor, LightGBM as the classifier**
   (`src/models/lm_perplexity.py`, configs `lm_gpt2*`, `binoculars_qwen*`).
   - GPT-2, GPT-2-medium or Qwen2.5 (base + instruct) score each token of the text; their weights are never updated.
   - The per-token log-probability, rank and entropy are summarised into 15 numbers per text. Binoculars models
     add 3 more: the performer's mean log-probability, the observer–performer cross-perplexity and the Binoculars score.
   - Most of these models also add 43 hand-written style features.
   - A LightGBM model trained on these features outputs P(AI).
2. **Fine-tuned encoder classifiers** (`src/models/transformer.py`, configs `ft_*`).
   - RoBERTa-base, DeBERTa-v3-large and the RoBERTa-large OpenAI detector, fine-tuned end to end with a
     classification head.
   - Test predictions average the 5 cross-validation fold models.

A blend maps each member's test score to a percentile using that member's out-of-fold scores, then takes a
weighted average (`src/ensemble.py: cdf_mean`). Every member scores every text: there is no gating and no voting.

- Blend A v2: greedy forward selection on the members' topic-held-out out-of-fold predictions
  (`run_all.py --profile topic_cv`, `scripts/select_blend.py`, rule in `configs/selection/blends.yaml`). Blend B:
  the four best runs by 5-fold CV in the 35-job benchmark, equal weights. Blend C v2: B + the best-CV fine-tune.
  A v1 and C v1 (earlier uploads) are in git history and the log.
- **Run everything in the `cs-aml` conda env** (`pretrained/environment.yml`). Base Anaconda lacks LightGBM and
  silently falls back to sklearn's HistGradientBoosting, and its sklearn version changes C1's output slightly.
- The uploaded runs used seed 42 and 5-fold CV. The weights live in the research repo's `outputs/runs/`, not
  in git.
- `make_submission.py` builds the CSVs from those runs (fine-tune members re-predict with bf16 noise).
  The full retrain takes about 4 h on two 12 GB GPUs. Don't retrain to "check" anything.

### classical/ — plug-in models

- One model = `models/<name>.py` (subclass of `BaseModel`: `fit`, `predict_proba`) + `configs/<name>.yaml`
  (params and an optional `search:` grid) + one line in `models/__init__.py` `REGISTRY`.
- `python train.py --model <name>` runs the search, fits on the 8k rows and scores the fixed 2k. It writes
  `results/<name>/` and updates `submission_log.csv`.
- `--final` refits on 10k rows and writes `kaggle_submissions/final_<X>_<name>.csv` (X: D, E, F; a new model
  passes `--letter`). It replaces
  that model's file and resets its log row to "pending upload".
- `python train.py --model <name> --cv topic` runs the search with topic-held-out CV.
  `python select_design.py --model <name>` applies a pre-registered design rule (`configs/selection/<name>.yaml`)
  and writes `results/selection/<name>/`.

### Entry points (team folder)

- `python train.py --model <classical model> [--final]` forwards to `classical/train.py`.
- `python train.py --model blend_b|blend_c|blend_a` retrains that blend's members (`pretrained/run_all.py --profile blends`, GPU, hours).
- `python predict.py --model <name> --out submission.csv` predicts from saved models or runs. Add `--committed` to
  copy the uploaded file instead.
- The full pipelines of all six submitted models are in `MODELS.md`.
- Full guide: `classical/CONTRIBUTING.md`.

## Rules for any change

1. Only add or modify files inside `submissions/group-number-31/`.
2. Never commit competition CSVs (`data/`), Kaggle credentials, `.env`, model weights, checkpoints, `outputs/`,
   `artifacts/` or caches. Run `git status` and check the staged list before every commit. The course root
   `.gitignore` ignores every `*.csv`; our `.gitignore` re-includes only results, logs, the val split and
   the uploaded submissions.
3. Supervised labels come only from `train.csv`, and no externally sourced labelled data may train, validate or
   choose anything (rule 4(b)). Don't use external LLM APIs to label or score test passages, and don't look for
   hidden labels (RAID or anywhere else).
4. Never type a number into a report or README by hand. Every number comes from a results file written by code:
   - `classical/report_tables.py`, `pretrained/scripts/report_tables.py`, `common/compare_tracks.py` and
     `reports/build_tables.py` generate the LaTeX tables; `reports/refresh.py` runs them all and inlines the
     result into `reports/overleaf/kaggle_report.tex` (see "The report and its Overleaf folder");
   - unknown values stay as visible `[TODO: ...]`.
5. Never tune anything on the 2,000 validation rows or on the Kaggle leaderboard.
6. Small commits, one idea each. Never squash, rebase or force-push pushed history: graders read it. The
   classical track uses the prefixes `feat(<model>)`, `exp(<model>)`, `sub(<model>)` and `log(<model>)`.
7. Ask a human before: any Kaggle upload, opening or updating the course PR, force-pushing, deleting branches,
   or a run longer than about 2 hours.
8. After a Kaggle upload, log the score with `python common/update_submission_log.py ... --public <score>`.
   It rewrites `SUBMISSIONS.md`. Then regenerate the tables.

## Branches

There is one branch, `main`; work there, or on a short-lived branch merged back into `main` and then deleted. The
older per-track and per-model branches (`pretrained`, `classical`, `model/*`) were merged and deleted on
2026-11-04; their commits are all in `main`'s history.

The course PR comes from `submission/group-number-31`, a copy of `main`. To update it, run
`git push origin main:submission/group-number-31`.

## The report and its Overleaf folder

The report we submit is **`reports/overleaf/kaggle_report.tex`**, the only file in `reports/overleaf/`. The team
uploads that folder to Overleaf as is, so it must always be current and self-contained:

- **Whenever a result, table, model, score or the log changes, update the report in the same piece of work**:
  run `python reports/refresh.py` (cs-aml env). It regenerates every table, inlines them into
  `reports/overleaf/kaggle_report.tex`, checks the file and compiles it to `reports/kaggle_report_preview.pdf`.
  Then fix any prose number that the change made stale (prose is not generated; search the `.tex` for the old
  value), run `refresh.py` again, and commit the `.tex` and the preview PDF together.
- Never edit between `% <<gen ...` and `% gen>>` by hand: those blocks are overwritten by `inline_tables.py`.
- Keep `reports/overleaf/` to exactly what Overleaf needs: no `\input`, no image files, no `.bib` (references
  are an inline `thebibliography`; the Q2.2 plot is pgfplots inside the `.tex`). Keep the file pure ASCII
  (Overleaf compiles with pdfLaTeX); `refresh.py` fails if either rule breaks.
- Matplotlib figures elsewhere (`*/reports/figures/`, `common/figures/`) are for the repo, not the report;
  if a regeneration changes only their timestamps, don't commit them.
- Tell the team after a report change, so they re-upload `reports/overleaf/kaggle_report.tex`.

## Regenerating everything (after new results)

```bash
cd submissions/group-number-31
python reports/refresh.py     # cs-aml env: all tables -> inline into reports/overleaf/ -> checks -> preview PDF
```

`refresh.py` runs, in order: `classical/report_tables.py`, `pretrained/scripts/report_tables.py`,
`common/compare_tracks.py`, `reports/build_tables.py`, `reports/inline_tables.py`, then a tectonic compile of
the Overleaf folder alone (tectonic is in the `tex` conda env; `--no-pdf` skips it).

## Open TODOs (search for `[TODO`)

- The PR URL (title page and Q6.1).
- Private scores, after the reveal.
