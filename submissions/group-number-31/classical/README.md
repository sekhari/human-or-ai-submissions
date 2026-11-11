# Classical track: models trained from scratch

No pretrained weights, embeddings or tokenizers anywhere in this folder. Every model trains on the team's
fixed 8,000 rows and is scored on the same 2,000 validation rows as the pretrained track
(`../common/val_ids.csv`, seed 20261023). Runs on a laptop CPU; `transformer_scratch` uses a GPU if present.

## Setup

```bash
pip install -r requirements.txt          # Python 3.11
python ../common/make_split.py           # verifies the shared validation split
```

Kaggle CSVs go in `submissions/<team>/data/` (git-ignored), or pass `--data-dir PATH` / set `HOA_DATA_DIR`.

## Train / evaluate on the fixed split

```bash
python train.py --model <name>           # inner-CV search on the 8,000 rows, fit, score the 2,000
```

Writes `results/<name>/metrics.json` (chosen params, train/val AUROC, fit time, hardware), `val_pred.csv`,
`search.csv`, and the model's row in the team's testing history (`../SUBMISSIONS.md`).

## Kaggle submission

```bash
python train.py --model <name> --final                # refit on all 10,000 rows (--letter <X> for a new model)
                                                       # -> kaggle_submissions/final_<X>_<name>.csv
python predict.py --model <name> --out submission.csv   # re-create the file from artifacts/<name>/
```

One Kaggle file per model, lettered after the pretrained track's A-C: D = `stylo_gbm`, E = `tfidf_lr`,
F = `transformer_scratch`. An improved version replaces its model's file under the same letter (E and F now
hold v2); the v1 scores stay in `../SUBMISSIONS.md` as `final_E_tfidf_lr@v1` / `final_F_transformer_scratch@v1`,
linked to the commits that hold the v1 files.

## Checking a design against topic shift (train labels only)

```bash
python train.py --model <name> --cv topic                    # the config's search, scored with topic-held-out CV
python select_design.py --model <name> --jobs 8              # pre-registered design selection (configs/selection/)
```

The fixed validation split comes from train's distribution, so it cannot show the drop to Kaggle (test has topics,
generators and far more verse than train). `--cv topic` clusters the 8,000 training rows into 10 topics (k-means on
word TF-IDF of those rows only, seed 20261023) and each of the 5 folds holds out whole topics
(`results/topic_cv/clusters.csv`). It needs no data beyond `train.csv`. `select_design.py` scores the candidates
listed in `configs/selection/<name>.yaml`, applies the rule written there (committed before the run) and writes
`results/selection/<name>/`.

## Models

| name | what |
|---|---|
| `template` | character-count logistic regression; the starting point for new models |
| `tfidf_lr` (C1) | word 1-2-gram + raw char 1-5-gram + word 1-3-grams over text-distorted input (features/distortion.py) TF-IDF -> logistic regression |
| `stylo_gbm` (C2) | 101 hand-crafted stylometric features (features/stylometry.py) -> LightGBM |
| `transformer_scratch` (C3) | 4-layer Transformer encoder (d=256) from random init over a 12k byte-level BPE trained on train + unlabeled text; input has single line breaks joined and is text-distorted |
| `linear_svm` | C1 v1's TF-IDF features (word + char_wb) -> linear SVM (comparison row) |
| `nbsvm` | NB-SVM: NB log-count-ratio scaled binary n-grams -> LR (comparison row) |
| `rf_stylo` | C2's stylometric features -> random forest (comparison row) |
| `ensemble_c123` | C1+C2+C3 rank/CDF blend, weights by greedy selection on inner-CV OOF predictions |

Adding a model: see [CONTRIBUTING.md](CONTRIBUTING.md).
