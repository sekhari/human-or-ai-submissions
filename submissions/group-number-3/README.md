# Group Number 3 — Human or AI?

**Team:** Aarushi Singh (`as4639`), Yuchen Feng (`yf467`), Vivian (Yan Ran) Li (`yl4546`)
**Kaggle team:** Group Number 3
**PR:** https://github.com/sekhari/human-or-ai-submissions/pull/23

**Final submission:** rank-average ensemble of five models: fine-tuned RoBERTa-base (0.5),
fine-tuned DistilRoBERTa (0.1), GPT-2 perplexity GBM (0.2), GPT-2 + GPT-2 Medium perplexity/Binoculars
GBM (0.1) and word+char TF-IDF logistic regression (0.1). All whitespace is collapsed before the
transformer and perplexity models (blank lines leaked the label in training but not in test).

| | Validation AUROC (fixed 2,000 rows) | Kaggle public AUROC |
|---|---|---|
| TF-IDF logistic regression baseline | 0.9506 | 0.864 |
| DistilRoBERTa + TF-IDF + GPT-2 perplexity | 0.9940 | 0.921 |
| **Final 5-model ensemble** | **0.9960** | **0.936** |

## Setup

Core experiments tested with Python 3.13/3.14 on macOS arm64 (Apple M4; transformers run on the
GPU via PyTorch MPS, CUDA or CPU also work). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r submissions/group-number-3/requirements.txt
pip install -r submissions/group-number-3/requirements-transformer.txt   # torch + transformers
cd submissions/group-number-3
```

Download the five official [competition CSVs](https://www.kaggle.com/competitions/cornell-tech-cs-5785-combined-2026-ai-vs-human-text/data)
into `data/`. Use the committed `splits/splits.csv` (stratified 2,000-row validation split, seed 42);
do not regenerate it. Run every command below from `submissions/group-number-3/`.

### Pretrained models (downloaded automatically from Hugging Face)

| Model | Used for | License |
|---|---|---|
| `distilroberta-base` | fine-tuned classifier | Apache-2.0 |
| `roberta-base` | fine-tuned classifier | MIT |
| `gpt2` | perplexity features | MIT |
| `gpt2-medium` | perplexity / Binoculars features | MIT |

No external labeled data and no external LLM APIs were used; all predictions come from models
trained locally on the provided `train.csv`.

## Reproduce the final Kaggle submission

Each model is retrained on all 10,000 labeled rows, then the predictions are rank-averaged.

```bash
python src/predict.py --model tfidf_logreg --out submission.csv                      # ~20 s
python src/finetune_transformer.py --mode full --model distilroberta-base             # ~14 min (M4)
python src/finetune_transformer.py --mode full --model roberta-base --max_len 256 --batch 8   # ~45 min
python src/perplexity.py  --mode full                                                 # seconds with cache
python src/perplexity2.py --mode full                                                 # seconds with cache
python src/ensemble_n.py --mode test \
  --models distilroberta-base roberta-base ppl2_gbm ppl_gbm tfidf --w 0.1 0.5 0.1 0.2 0.1
```

Output: `sub_ensemble_final.csv` (3,000 rows, `id,label`), the file submitted to Kaggle.

Perplexity features are cached in `results/ppl_features.csv` and `results/ppl2_features.csv`.
To recompute them, delete the cache (~9 and ~32 min on an M4). `perplexity2.py` scores and caches
on its first run and exits; run the same command again to train. If `perplexity.py` crashes right
after saving features on macOS (an OpenMP clash between torch and scikit-learn), just rerun it.

## Validation experiments (fixed 2,000-row validation set)

```bash
python src/evaluate.py --model tfidf_logreg          # also: word_logreg, char_logreg, naive_bayes, linear_svm, style_gbm
python src/model_table.py                            # train/val AUROC + time for all classical models
python src/tune.py                                   # LogReg C grid
python src/tune_gbm.py                               # style GBM grid
python src/finetune_transformer.py --mode val --model distilroberta-base
python src/finetune_transformer.py --mode val --model roberta-base --max_len 256 --batch 8
python src/perplexity.py  --mode val
python src/perplexity2.py --mode val
python src/ensemble_n.py --mode val --models distilroberta-base roberta-base ppl2_gbm ppl_gbm tfidf
```

Analysis scripts: `src/newline_check.py`, `src/style_importance.py`, `src/robustness.py`,
`src/error_analysis.py`, `src/error_analysis_probs.py`, `src/subgroup_analysis.py`,
`src/pseudo_label.py`, `src/svm_ngram.py`.

### Training-size and pseudo-labeling study (Q2/Q3)

```bash
python src/complete_person_b.py
python -m unittest discover -s tests
```

`complete_person_b.py` runs fresh Q2 TF-IDF curves and Q3 full-pipeline pseudo-labeling.
Add `--rerun-style` for style-model curves. Q2 uses subset seeds 0/1/2 and one full-pool fit;
split/style seed is 42. Parameters are in `src/models.py` and `config/person_b.json`.
The earlier frozen-feature study remains in `src/person_b.py`; run stages
`verify`, `baseline`, `q2`, `q3` in that order.

## Seeds and configuration

- Validation split: stratified, seed 42 (`splits/splits.csv`).
- Transformers: seed 42, 2 epochs, AdamW, learning rate 2e-5, linear schedule with 6% warmup,
  max length 256; batch 16 (DistilRoBERTa) / 8 (RoBERTa-base).
- Gradient boosting (style and perplexity): `HistGradientBoostingClassifier`, 300 iterations,
  learning rate 0.05, `random_state=42`. Logistic regression: C=4, deterministic.
- GPU training on MPS is not bit-for-bit deterministic, so re-runs can differ slightly.

## AI-tool disclosure

We used Claude (Anthropic) and Codex (OpenAI) to write most of the Python scripts (data splits, model definitions,
robustness and error-analysis scripts, transformer fine-tuning, perplexity features, ensembling),
to debug git and environment problems, to explain results and to draft report text. Team members
ran every experiment, checked the outputs against the logged results, read the misclassified
passages, and made the modeling decisions (hyperparameters, rejecting the style model after finding
the newline artifact, which models to submit and select as final). See report Q6.3 for details.
