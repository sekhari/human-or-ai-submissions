# Group_Number_33 — Human or AI?

Cornell Tech · CS5785 Applied Machine Learning

**Kaggle team name:** `Group_Number_33`

| Team member | NetID |
|---|---|
| Yuqi Wang | yw2897 |
| Lemuel Sumardy | ls2332 |
| William Quinn | wcq3 |

This is the team entrypoint for reproducing our selected model: **145 handcrafted writing-style and structural features followed by scikit-learn's 800-tree ExtraTreesClassifier**. The expected AUROC on our fixed 8,000/2,000 validation protocol is **0.9956835**. This is a local validation result, not a Kaggle public or private score.

## 1. Files and responsibilities

```text
submissions/group_number_33/
  README.md                  setup, method, commands, and submission requirements
  requirements.txt           pinned dependencies for the selected pipeline
  train.py                   validation OR final training, selected by --mode
  predict.py                 load the final model and write submission.csv
  test_entrypoints.py        checks for the command-line entrypoints
  code/                      preserved team implementation and experiment history
    models/dave_extratrees/   selected model, frozen recipe, and error analysis
    models/text_generalization/style_search.py  the 145 feature calculations
  data/                      original Kaggle CSVs, supplied locally and ignored
  artifacts/                 generated models/metrics, local and ignored
  submission.csv             generated Kaggle probabilities, local and ignored
```

The top-level scripts reuse the existing model and feature extractor. They do not replace scikit-learn's tree implementation or change the selected recipe. Other team approaches remain under `code/models/`.

## 2. Install the environment

Use **Python 3.10**; the retained pipeline and these entrypoints were tested with **Python 3.10.8** on macOS (Apple Silicon). A GPU, pretrained model download, and external API key are not needed for the selected model.

From a fresh checkout:

```bash
git clone --branch submission/Group_Number_33 https://github.com/TheLemperor/human-or-ai-submissions.git
cd human-or-ai-submissions/submissions/group_number_33
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

If your Python 3.10 executable is named `python3`, use that when creating the environment. On Windows, activate with `.venv\Scripts\activate` instead.

The dependency file pins NumPy, pandas, SciPy, scikit-learn, joblib, and threadpoolctl. It covers final-model training, prediction, entrypoint/model tests, and error analysis. Archived CNN, embedding, and other comparison experiments can require additional dependencies; they are not imported by the final pipeline.

**Run the remaining commands from this team directory**, with the environment activated. The scripts also work from other directories when given explicit paths.

## 3. Supply the competition data locally

Download the original files from the [Kaggle competition](https://www.kaggle.com/competitions/cornell-tech-cs-5785-combined-2026-ai-vs-human-text/data) and place them here:

```text
data/train.csv               10000 rows: id, text, label
data/test.csv                 3000 rows: id, text
data/sample_submission.csv    3000 rows: id, label
```

Labels are **0 = human** and **1 = AI**. `id` identifies a passage and is never a model feature. Use `--data-dir /absolute/path/to/data` if the files are elsewhere.

Use the original CSV bytes and row order. The frozen [recipe](code/models/dave_extratrees/recipe.json) contains input checksums and ordered split digests; the scripts reject altered files rather than silently change the benchmark. Do not preprocess, rewrite, or sort the input CSVs before running these commands.

`unlabeled.csv` and `train_starter_2000.csv` are not used by the selected final model. The optional unlabeled-data experiment below uses `unlabeled.csv` separately. The starter file is a subset, not extra labeled data.

Original competition CSVs and generated MiniLM caches were removed from this submission branch's reachable history on November 11, 2026. Supply the CSVs locally as above. Archived LR/RF experiments regenerate missing MiniLM embeddings using their additional dependencies and pretrained weights; `--skip-minilm` omits those embedding experiments. The final ExtraTrees pipeline does not use these caches.

The cleanup retained all 34 existing commits, their authors, dates, messages, and merge structure; affected commit IDs changed. If you cloned the branch before this cleanup, save any local work and clone it into a fresh directory before continuing. Do not merge the old history back into the cleaned branch.

## 4. Train and evaluate on the fixed validation split

```bash
python train.py --mode validate --data-dir data --output-dir artifacts/validation
```

`validate` is the default mode, so omitting `--mode validate` has the same effect. This command:

1. Reads **only `train.csv`** and checks its checksum and schema.
2. Recreates the fixed, stratified split with `test_size=0.2`, `random_state=12`, and `stratify=label`: 8,000 training rows (4,000 per class) and 2,000 validation rows (1,000 per class).
3. Independently converts each passage into 145 numbers, fits the selected model on the 8,000 training rows, and computes training and validation AUROC.
4. Saves metrics, validation probabilities, exact training/validation IDs, the recipe, and ordered feature names under `artifacts/validation/`.

Expected validation AUROC: **0.9956835**. Runtime depends on hardware; feature extraction is included in the reported fitting time. This mode does not fit all 10,000 rows, create a final model artifact, read test passages, or generate Kaggle predictions.

The validation partition was reused during earlier model development. It is a fixed development holdout, not an untouched independent test. The stateless feature extractor learns no vocabulary or corpus statistics from validation passages, and their labels are not used to fit the 8,000-row model.

## 5. Refit the selected model for the final submission

After selecting the configuration, train a separate model on all 10,000 labeled rows:

```bash
python train.py --mode final --data-dir data --output-dir artifacts/final
```

This command reads only `train.csv`, uses the unchanged recipe, and saves `artifacts/final/model.joblib`, `metrics.json`, `recipe.json`, and `feature_names.json`. The model file includes provenance metadata and is verified after reloading. This final fit has no held-out validation score; the report's validation AUROC comes from Step 4.

Training modes refuse to overwrite an existing run directory. To repeat an experiment, choose another directory, such as `--output-dir artifacts/validation-repeat`.

## 6. Generate submission.csv

```bash
python predict.py --data-dir data --model artifacts/final/model.joblib --out submission.csv
```

This command **does not train or tune a model**. It loads the trusted local artifact produced in Step 5, verifies the selected recipe and 10,000-row training provenance, reads only `test.csv` and `sample_submission.csv`, and computes `P(AI)` for all test passages. `train.csv` is not required for prediction.

The output is checked for:

- Exactly 3,000 rows and the columns `id,label`.
- Every test ID exactly once, in `sample_submission.csv` order.
- Finite numeric probabilities between 0 and 1; here `label` is a probability, not a hard class.

The script refuses to overwrite existing files. For another prediction run, use another `--out` path. It never uploads automatically. Submit the generated CSV manually through Kaggle. Different platforms or parallel summation order can change the last floating-point digits; use an absolute tolerance of **1e-12** when comparing predictions, not CSV byte equality.

## 7. How raw text becomes a probability

```text
Raw text, preserving punctuation, case, spacing, and line breaks
    -> fixed word/sentence/paragraph parsing rules
    -> 145 numeric style and structure measurements
    -> sklearn ExtraTreesClassifier (800 trees)
    -> averaged AI probability, clipped to [1e-7, 1 - 1e-7]
```

Words are identified with a Unicode regular expression that keeps internal straight/curly apostrophes. Sentences use a punctuation/newline heuristic; paragraphs are separated by blank lines. Tokens are case-folded for repetition/diversity statistics, while the original text remains available for capitalization and layout features. We do not globally lowercase, strip punctuation, remove stopwords, stem, or lemmatize the text.

Feature groups include length, vocabulary diversity/repetition, word/sentence length distributions, character proportions, punctuation, and formatting/layout. Each passage produces one fixed-length row. The final method uses **no TF-IDF, learned embedding, corpus vocabulary, or feature scaler**. The number 145 comes from explicitly defined measurements; it is not a learned embedding dimension or a claim of optimal feature count.

See the [feature names](code/models/dave_extratrees/feature_names.json), [feature calculations](code/models/text_generalization/style_search.py), and [full method explanation](code/models/dave_extratrees/README.md).

| Setting | Selected value |
|---|---|
| Split seed / model seed | 12 / 12 |
| Validation fraction | 0.2, stratified by label |
| n_estimators | 800 |
| max_features | 0.8 |
| min_samples_leaf | 2 |
| max_depth | None |
| criterion | gini |
| bootstrap | True |
| max_samples | 0.8 |
| class_weight | balanced_subsample |
| n_jobs / thread-pool limit | 2 / 2 |

Bootstrap sampling occurs **inside the training partition** and is separate from the validation split. The prediction code uses the AI class (`1`) probability. Threshold `0.5` is used only when selecting classification errors for analysis, not for AUROC scoring or the Kaggle CSV.

## 8. Verification and report experiments

The [entrypoint reproduction record](entrypoint_reproduction.json) records a fresh-environment run on 2026-11-11: 9 lightweight checks and 1 real-data integration test passed. All 2,000 validation probabilities matched the retained version exactly; the largest difference across 3,000 Kaggle probabilities was approximately `1.11e-16`, below the `1e-12` comparison tolerance. Q4 error counts were also reproduced. Optional archived experiments were not rerun as part of this entrypoint update.

Run lightweight CLI and model checks without competition data:

```bash
python -m unittest test_entrypoints.EntrypointTests -v
PYTHONPATH=code python -m unittest models.dave_extratrees.test_model -v
```

Run the optional real-data integration test. It uses temporary directories, verifies validation with only `train.csv` available, and verifies prediction with only the two test-side CSVs available:

```bash
COMPETITION_DATA_DIR="$PWD/data" python -m unittest test_entrypoints.ReproductionTests.test_separate_validation_training_and_prediction -v
```

For generalization/error analysis, first complete Step 4, then run:

```bash
PYTHONPATH=code python -m models.dave_extratrees.analyze_errors \
  --run-dir artifacts/validation --data-dir data --output-dir artifacts/error-analysis
```

It verifies the held-out predictions and writes a timestamped `analysis.json`. It does not load the final 10,000-row model or fit anything. Expected results include 19 false positives and 35 false negatives at threshold 0.5. Strong correlations between formatting and labels limit what the high random-holdout score establishes about new domains.

The retained [training-size and unlabeled-data experiments](code/models/dave_extratrees/q4_q5/README.md) use a separate, existing `code/data/` location. To reproduce them, also download `unlabeled.csv`, then run:

```bash
python -m pip install -r code/models/dave_extratrees/q4_q5/requirements.txt
mkdir -p code/data
cp data/train.csv data/unlabeled.csv code/data/
PYTHONPATH=code python -m models.dave_extratrees.q4_q5.learning_curve
PYTHONPATH=code python -m models.dave_extratrees.q4_q5.unlabeled
```

These write `code/outputs/q4_q5/`. The learning curve uses 500/1,000/2,000/5,000/8,000 training examples; the pseudo-labeling experiment uses confidence 0.95 and the same fixed validation partition. They are separate report experiments and do not alter the final model generated above. Other archived model instructions live in [`code/`](code/README.md); their own dependencies and experiment settings apply.

## 9. AI-tool disclosure

We used OpenAI Codex to assist with coding, debugging, running experiments, checking reproducibility, and improving documentation. We directed the experiments, reviewed validation and Kaggle results, and selected the final approach. AI also helped explain the implementation and its limitations. The final model runs locally and does not use external LLM APIs to classify competition passages.


