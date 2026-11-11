**Team members:** Rachel Liu (rl996), Henry Lam (hgl32), Yang Xiao (yx752)  
**Kaggle team name:** `team-27`

Final model: `svm_frozen_lm_yx752_v6.ipynb`, with `RUN_PARAMETER_SEARCH=False` (the default). It blends the v5 SVM / gradient-boosting fusion of hand-built style features with a LightGBM model trained on frozen GPT-2 features (75% LM branch, 25% v5).

## Setup

### 1. Python version

Use **Python 3.10 or newer**. Check your version:

```bash
python3 --version
```

### 2. Create an environment and install dependencies

Run every command below from this folder (`submissions/team_27/`).

```bash
cd submissions/team_27
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_v6.txt
pip install jupyter                # runs the notebooks from the command line
```

`requirements_v6.txt` pins the versions used for the final model (`lightgbm==4.6.0`, `transformers==4.44.2`, `tokenizers==0.19.1`, `huggingface-hub==0.25.2`, `safetensors==0.4.5`). `requirements.txt` holds the looser dependencies for the earlier notebooks (`neuralnet_rl998_v*.ipynb`, `logistic_hgl32.ipynb`, `svm_yx752*.ipynb`).

**GPU (optional):** a CUDA build of PyTorch speeds up feature extraction. To get one, install PyTorch first, using the command for your platform from <https://pytorch.org/get-started/locally/>, then run `pip install -r requirements_v6.txt`. Everything also runs on CPU.

**macOS + LightGBM:** if `import lightgbm` fails with `Library not loaded: libomp.dylib`, install the OpenMP runtime:

```bash
brew install libomp
```

### 3. Add the competition data (not committed)

Download the Kaggle files and put them in `data/`:

```text
submissions/team_27/data/
  train.csv
  test.csv
  unlabeled.csv
  sample_submission.csv
```

`data/` is in `.gitignore`. The notebook also checks the current folder and `../cornell-tech-cs-5785-combined-2026-ai-vs-human-text/`.

### 4. Download the frozen GPT-2 checkpoint (one time)

```bash
python frozen_lm_features.py --download --data data
```

This saves the `openai-community/gpt2` model and tokenizer files to `model_cache/`, which is gitignored. It is the only step that needs network access. No passage text is sent to any external service.

## Train / validate

**Step 1: extract the frozen LM features.** This needs no labels and reads only from the local cache.

```bash
python frozen_lm_features.py --data data --output lm_features/gpt2 --batch-size 8
# CPU-only machine:
python frozen_lm_features.py --data data --output lm_features/gpt2 --device cpu --batch-size 1
```

It writes feature CSVs for train, unlabeled and test into `lm_features/gpt2/`. Each CSV has a `.json` sidecar that records the model name, the feature list and a SHA-256 hash of the input file. The notebook checks these before it uses the features. Each passage is cut to 511 tokens plus BOS, and the model's weights are never updated.

**Step 2: run the model selection and development evaluation.**

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 svm_frozen_lm_yx752_v6.ipynb
```

The notebook:

- holds out a stratified 2,000-row development split from `train.csv` (seed 27, saved to `v6_results/val_ids.csv`)
- reproduces the recorded final LM branch: 31 leaves, 400 trees, learning rate 0.03, L2=2, LM-branch weight 0.75
- computes three-fold training OOF scores for calibration, without selecting another model
- scores the chosen blend once on the development split (`v6_results/development_comparison.csv`, `v6_results/summary.json`)

Once the features from Step 1 exist, the notebook runs on CPU. Optional environment variables `V6_DATA_DIR` and `V6_LM_FEATURE_DIR` can point to another existing data/feature directory; otherwise the documented relative directories are used. Baseline OOF scores are recomputed. To repeat the earlier search as a separate experiment, set `RUN_PARAMETER_SEARCH=True`; its outputs go to `v6_search_results/`, rather than replacing the default final-run outputs.

Original submitted-configuration reference results: v5 baseline 0.998104, frozen branch 0.999448, selected blend **0.999590** development AUROC. The original public AUROC was **0.98872**. These reference values identify the recorded run; rerunning the same parameters on another hardware/software stack may produce small numerical differences.

**Optional report analyses** (training AUROC, learning curve, length bins, error examples, unlabeled-data experiment). These don't overwrite the submission:

```bash
python v6_report_analysis.py      # writes to v6_results/report/
```

Reproducibility depends on both model settings and the frozen feature tables. The original run used CUDA/float16 extraction with torch 2.5.1; a CPU/float32 extraction or different package versions can change feature values slightly. Keep the feature sidecars and input hashes when comparing runs.

## Predict (writes Kaggle submission)

The last cell of `svm_frozen_lm_yx752_v6.ipynb` refits every component on all labeled rows and writes the submission. That cell runs as part of the `nbconvert` command above. The output file is:

```text
v6_results/submission_yx752_v6.csv
```

It has columns `id,label` (P(AI)) and 3,000 rows, ordered like `sample_submission.csv`. The notebook asserts this format before it saves the file.

To get the file as `submission.csv` for Kaggle:

```bash
cp v6_results/submission_yx752_v6.csv submission.csv
```

> `train.py` and `predict.py` are still the course template stubs and do nothing. Use the commands above.

## Seeds and notes

- Final model seed: `SEED = 27`. It's used for the dev split, every `StratifiedKFold(n_splits=3)`, LightGBM / HistGradientBoosting and logistic calibration.
- Validation: stratified 2,000-row holdout from `train.csv`. It's the same split across all model-comparison notebooks.
- Chosen LM-branch config: LightGBM on GPT-2 + 73 style features, `num_leaves=31`, `n_estimators=400`, `learning_rate=0.03`, `reg_lambda=2`, `colsample_bytree=0.8`, `subsample=0.8`.
- Blend: `0.25 × v5 logit + 0.75 × LM-branch logit`, then logistic calibration.
- Frozen checkpoint: `openai-community/gpt2` (recorded revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`). All supervised labels come from the provided `train.csv`. No external labeled data and no LLM APIs are used to label or predict test passages.

## AI assistance

- **Henry Lam (hgl32):** Used Claude to explain concepts (TF-IDF, AUROC, regularization), to draft and review notebook code (the evaluation helper, C and α grids, error-analysis and feature-comparison cells, and the test-prediction cell), and to draft report text from our notebooks. I ran every experiment, chose the preprocessing and final settings from the results, read the misclassified passages to propose the 6 added features, and checked reported numbers against notebook output.
- **Rachel Liu (rl996):** Used Claude and Copilot to brainstorm code and debug my three neural-network models (NN v1, v2 and v3). I read through many of the passages labeled as AI, identified stylization features, and had Claude help me write regex functions for them to feed into the preprocessing steps. I also used Claude to validate the data for these written portions. I ran every neural-network experiment myself, chose the training settings (such as the number of epochs, picked by lowest development loss) from the validation results, and decided which features to add in v2 and v3. I reviewed and edited all AI-suggested code and text before using it, and I did not use any AI tool to label or classify the competition passages.
- **Yang Xiao (yx752):** Used ChatGPT/Codex to assist with developing, debugging, and optimizing SVM models, feature extraction, and ensemble methods, as well as analyzing experimental results and drafting report text. I independently compared validation and Kaggle scores across multiple versions, evaluated the effectiveness of AI-assisted code improvements, and selected v6 based on experimental performance. I also verified the shared validation split, prediction-to-ID alignment, probability outputs, and reproducibility, and reviewed a shuffled-label control to ensure the reliability of the results.

No external LLM API was used to label or classify competition passages. GPT-2 ran locally as a frozen feature extractor, and all labels came from `train.csv`.
