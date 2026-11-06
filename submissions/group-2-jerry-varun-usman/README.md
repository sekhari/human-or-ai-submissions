# Group 2 — Jerry, Varun, Usman · Human or AI?

**Kaggle team name:** `Group 2 - Jerry, Varun, Usman`
**Members:** Usman Ali Arshi (NetID: TODO), Jerry (NetID: TODO), Varun (NetID: TODO)

> Branch `usman/deberta-pipeline` is Usman's individual approach. The team will submit
> whichever member's pipeline scores best on the shared validation protocol.

## Approach (this branch)

Fine-tuned **DeBERTa-v3-small** classifier (learned subword representation), compared against
word TF-IDF + LR, character n-gram TF-IDF + LR, and a small "weird character" / stylometric
feature model. The final submission is the best of DeBERTa alone or an equal-weight rank
average of DeBERTa with the classical models, chosen on validation AUROC.

## Setup

Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put the Kaggle files (`train.csv`, `unlabeled.csv`, `test.csv`, `sample_submission.csv`) in
`./data/` (gitignored, never committed). Override with `AML_DATA_DIR=/path/to/data`.

## Validation protocol

Fixed stratified 2,000-row holdout from `train.csv` (seed **42**), identical for every model
(`src/common.py`). Training rows whose normalized text exactly matches a validation row are
dropped. Classical hyperparameters are selected by 3-fold CV on the training partition only;
the validation set is used for reporting and final-config selection, never for fitting.

## Commands

```bash
python baselines.py                    # Q1: word / char TF-IDF + weird-char models (CPU, minutes)
python train_deberta.py --mode all     # main, learning curve, pseudo-labels, final (GPU recommended)
python analyze.py                      # tables -> results/report_tables.md, plot, errors.csv
python predict.py --out submission.csv # Kaggle submission from cached final predictions
```

On Colab, open `colab_runner.ipynb` and Run all (T4 GPU, ~1 hour).

## Seeds and config

- Split / CV / subsampling / training seed: `42`
- DeBERTa: `microsoft/deberta-v3-small`, max_len 384, batch 16, 2 epochs, lr 3e-5,
  weight decay 0.01, 10% linear warmup, fp16 on CUDA (`--no-amp` to disable)
- Pseudo-labeling: unlabeled passages with P(AI) ≥ 0.95 or ≤ 0.05 from the main model

## AI assistance

Code was developed with Claude (Anthropic), which drafted the scripts from our experiment
plan. We reviewed the code, ran every experiment ourselves, and checked reported numbers
against the saved prediction files. Pretrained weights: `microsoft/deberta-v3-small` (Hugging Face).
