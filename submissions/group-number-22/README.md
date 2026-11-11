# Human or AI? Group 22

**Team:** Gaurav Patel (gp438), Jalil Jabbarli (jj864), Rahul Devasia (rd668)
**Kaggle team name:** Group Number 22
**Development repo:** https://github.com/Jalil-g/Cornell-Text-Authorship (full history is included in this folder)

We predict the probability that a passage is AI-written. Our metric is AUROC.

## Final model

Three models, each trained only on labeled training text:

- **Style:** HistGradientBoosting on 147 writing-style features (lengths, vocabulary variety, case, punctuation, blank lines, common words). No word identity. Also trained on copies with the spacing squashed, so it does not rely only on layout.
- **Text:** word 1-3 + case-preserving character 2-5 TF-IDF logistic regression (C=16), averaged with a character 2-5 linear SVM (C=1) trained on passages without blank lines. Passages with a blank line get +2, but only if no human training passage has one.
- **ELECTRA:** `google/electra-small-discriminator`, fine-tuned locally for 3 epochs, 256 tokens.

Each model's score is standardized on `unlabeled.csv`. Final score: `P(AI) = sigmoid(0.75 * (0.5 * style + 0.5 * text) + 0.25 * ELECTRA)`.

| Result | AUROC |
|---|---:|
| Frozen validation (2,000 rows, models fit on 8,000) | 0.9991 |
| Kaggle public | 0.98553 |

## Setup

Python 3.12. Dependencies are pinned with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked                          # core environment (notebooks 01-15, 17)
uv sync --project envs/electra --locked   # adds torch 2.15.0 and transformers 5.22.2 (notebooks 16, 18-20)
```

**Data:** put the five competition CSVs in `data/raw/`. Notebook 01 can also download them with KaggleHub: copy `.env.example` to `.env` and add your Kaggle token. Data, models and predictions are never committed.

## Make `submission.csv`

One standalone notebook. It reads the raw CSVs, trains all three models on the 10,000 labeled rows, and writes the submission. It takes about 8 minutes on an Apple M-series laptop, mostly for ELECTRA.

```bash
uv run --project envs/electra --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=7200 notebooks/20_final_submission.ipynb
```

Output: `submissions/20_final_submission/<run-id>/submission.csv` (columns `id,label`, label = P(AI)).

## Reproduce the validation experiments

Each notebook is one experiment. Run them in order; later notebooks use files saved by earlier ones. The split (8,000 train / 2,000 validation, stratified, seed 42) is fixed in `configs/split.json`.

```bash
uv run --locked jupyter nbconvert --execute --to notebook --inplace notebooks/01_data_setup.ipynb
for nb in 03_word_baseline 04_word_comparison 05_char_comparison 06_learning_curve \
          07_pseudo_labels 08_error_analysis 09_linear_style 10_expanded_text \
          11_formatting_rule 12_seed_stress 13_rich_style 14_style_augmentation \
          15_normalized_transfer; do
  uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=3600 notebooks/$nb.ipynb
done
uv run --project envs/electra --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=7200 notebooks/16_local_electra.ipynb
uv run --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=3600 notebooks/17_final_blend.ipynb
for nb in 18_final_checks 19_internal_neural_pick; do
  uv run --project envs/electra --locked jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=7200 notebooks/$nb.ipynb
done
```

Each run writes a JSON record to `results/<notebook>/` with the Git commit, data and split hashes, package versions, settings, seed, runtime and metrics. Each experiment is summarized in `docs/`.

| Notebooks | What they test |
|---|---|
| 01-02 | Data checks, duplicate audit, fixed split and metric helpers |
| 03-06 | Word baseline, model comparison, character features, learning curve |
| 07 | Self-training with `unlabeled.csv` (no gain) |
| 08-09 | Error analysis, small style feature test (not adopted) |
| 10-11 | Larger text model grid, blank-line rule |
| 12 | Seed, split and formatting stress tests |
| 13-15 | Style features, formatting augmentation, normalized features |
| 16-19 | ELECTRA, final blend, final refit and checks, blend weight check |
| 20 | Final submission |

## Seeds and settings

- Seed 42 for the split, folds, models and ELECTRA.
- scikit-learn runs on one thread.
- Every vectorizer, scaler and model is fit on training rows only. Choices were made on training data, and validation was used to confirm them.
- ELECTRA on Apple MPS is not bit-for-bit repeatable. A rerun matched the submitted file with Spearman correlation 0.9969.

## AI-use disclosure

We used OpenAI Codex and Anthropic Claude Code for implementation, debugging, and running experiments. We also used agents to simplify and reorganize the notebooks, and Claude helped draft the report from the saved results. All code and results were reviewed, modified, and verified by us.