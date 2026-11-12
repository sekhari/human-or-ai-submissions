# Group Number 2: Human or AI?

**Kaggle team name:** `Group Number 2`
**Members:** Usman Ali Arshi (ua49), Jerry Zhou (zz977),
Varun Nautiyal (vn239)

## Final submission

A rank average of DeBERTa-v3-base (`base512`) and a
gradient-boosting model on language-model likelihood features,
with weights 0.8 and 0.2. Validation AUROC 0.9961 (base512 alone:
0.9943). Kaggle public AUROC 0.95636. Our second selected
submission is base512 alone (public 0.94871).

## Repository map

| Approach | Where |
|---|---|
| DeBERTa-v3 small and base, TF-IDF + LR, weird-character model, learning curve, pseudo-labels, analysis | this folder |
| Word and character TF-IDF + linear SVM | `tfidf_svm/` |
| LM likelihood features (Qwen2.5-0.5B) and the final blend | `lm_stack/` (see its README) |

## Setup

Python 3.10+ (`lm_stack` was tested on 3.12.10).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put the Kaggle files (`train.csv`, `unlabeled.csv`, `test.csv`)
in `./data/` (gitignored, never committed). Override with
`AML_DATA_DIR=/path/to/data`. `lm_stack/` has its own
`requirements.txt` and reads its own `lm_stack/data/`.

## Validation protocol

Fixed stratified 2,000-row holdout from `train.csv` (seed 42).
This folder and `lm_stack/` use the same split (we checked that
the 2,000 ids match); `tfidf_svm/` uses the same seed and
settings (stratified, 20%). Training rows whose normalized text
exactly matches a validation row are dropped (none were found).
Classical hyperparameters are chosen by 3-fold CV on the training
partition only. The validation set is used for reporting and
final selection, never for fitting.

## Commands

Run from this folder unless noted.

```bash
python baselines.py                     # word/char TF-IDF + weird-char
python train_deberta.py --mode all      # DeBERTa-v3-small, all steps
python train_deberta.py --mode main --tag base512 \
  --model-name microsoft/deberta-v3-base \
  --max-len 512 --batch-size 8 --grad-accum 2
python train_deberta.py --mode final --tag base512 \
  --model-name microsoft/deberta-v3-base \
  --max-len 512 --batch-size 8 --grad-accum 2
python analyze.py                       # tables and plots
```

On Colab, open `colab_runner.ipynb` and Run all (T4 GPU, about
1 hour for the small model).

`tfidf_svm/`: run `zz977_submission.ipynb` with `tfidf_svm/` as the
working directory and `train.csv`, `test.csv` in `tfidf_svm/data/`.
It writes `tfidf_svm_submission.csv`.

`lm_stack/`: see `lm_stack/README.md` for feature extraction and
training (`extract_features.py`, `train.py --seed 42`).

## Commands that write the submission files

```bash
# final submission (blend); needs results/preds/ from the runs above
cd lm_stack
python ensemble.py --npz-dir ../results/preds
# writes lm_stack/blend/submission_w0.2.csv

# second selection: base512 alone
python blend.py --submit base512 --out results/submission_base512.csv
```

`python predict.py --out submission.csv` writes the submission for
the `final_config` in `results/results.json` (currently
DeBERTa-v3-small).

## Seeds and config

- Seed 42 for the split, CV, subsampling, training and the
  bootstrap.
- DeBERTa-v3-small: `microsoft/deberta-v3-small`, max_len 384,
  batch 16, 2 epochs, lr 3e-5, weight decay 0.01, 10% linear
  warmup, fp16 on CUDA (`--no-amp` to disable).
- base512: `microsoft/deberta-v3-base`, max_len 512, batch 8,
  gradient accumulation 2.
- Pseudo-labeling: unlabeled passages with P(AI) >= 0.95 or
  <= 0.05 from the main model.
- Blend: LM weight 0.2, chosen from 0.1, 0.2 and 0.3 before any
  results were seen.
- Pretrained weights: `microsoft/deberta-v3-small`,
  `microsoft/deberta-v3-base`, `Qwen/Qwen2.5-0.5B`,
  `Qwen/Qwen2.5-0.5B-Instruct`. No external LLM API was used on
  test passages.

## AI assistance

- **Usman:** Claude (Anthropic) drafted the scripts in this folder
  from our experiment plan. We reviewed the code, ran every
  experiment ourselves, and checked reported numbers against the
  saved prediction files.
- **Jerry:** Claude for ideation and Gemini to curate the code in
  `tfidf_svm/`.
- **Varun:** Claude (chat) for planning, explanations and report
  drafting, and Claude Code for the code in `lm_stack/`. I read the
  code, ran the tests, extraction, training and Kaggle submissions
  myself, and checked Usman's predictions against his published
  AUROCs before blending.