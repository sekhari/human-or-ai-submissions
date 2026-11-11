# Group Number 1 — Human or AI?

## Team

- Alina Patel — ap2599
- Haoxiang Zhang — hz795
- Tim Eilers - te228

## Entrypoint

All feature extraction, training, validation experiments and prediction
code is contained in `Human_or_AI_Full_Colab_Training.ipynb`.

The notebook follows report questions Q0–Q6. It compares Logistic
Regression, Linear SVM and CatBoost, evaluates training-set size and
pseudo-labeling, and analyzes the final ensemble.

The final approach combines frozen GPT-2 and Pythia features,
formatting/repetition measurements, contextual vectors and sparse-text
features. Supervised classifiers convert these features into P(AI).

## Data

Download the five supplied competition files from Kaggle:

- train.csv
- test.csv
- unlabeled.csv
- train_starter_2000.csv
- sample_submission.csv

Place them in a local `data/` folder. The starter file is a subset of
train.csv, not additional labeled data.

Competition data, credentials, feature caches and model checkpoints
are not included in this repository.

## Environment

Use Python 3.12 or 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name human-ai-group-1
```

The reference training run used a Colab Tesla T4. Frozen language-model
feature extraction uses CUDA when available; classifiers and PCA use
CPU. A full CPU run can be substantially slower.

## Train, evaluate and generate predictions

Run these commands from this team directory:

```bash
mkdir -p outputs

HUMAN_AI_DATA_DIR="$PWD/data" \
HUMAN_AI_OUTPUT_DIR="$PWD/outputs" \
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.kernel_name=human-ai-group-1 \
  --ExecutePreprocessor.timeout=-1 \
  --output Human_or_AI.executed.ipynb \
  --output-dir outputs \
  Human_or_AI_Full_Colab_Training.ipynb
```

This executes the controlled experiments, refits the final classifiers
on all 10,000 labeled examples and generates test probabilities.

Copy the generated file to `./submission.csv`:

```bash
python -c "import json,pathlib,shutil; r=json.loads(pathlib.Path('outputs/latest_run.json').read_text()); shutil.copyfile(r['submission'],'submission.csv')"
```

Outputs include experiment tables, figures, fitted models and a
prediction manifest. Matching caches can be reused. Set
`HUMAN_AI_REUSE=0` to recompute them.

## Google Colab

Open the notebook, select a T4 GPU runtime, and place the competition
files in `MyDrive/human_or_ai/data/`. Run the notebook from top to bottom.

The setup cell also allows custom data and output paths.

## Validation and reproducibility

- Random seed: 42.
- Fixed stratified split: 8,000 training and 2,000 validation examples.
- Hyperparameter selection: five-fold CV within the training partition.
- Learned transformations are fitted on the relevant training rows.
- Final Kaggle classifiers are refitted on all 10,000 true labels.
- Q3 pseudo-labels are used only in the separate unlabeled-data experiment.
- Data hashes, model revisions, configuration and package versions are recorded.

The reference validation AUROC is 0.999391. The team-reported public
Kaggle AUROC is 0.98819 for the reference submission. Newly generated
files do not automatically inherit that score. Numerical results may
vary with the environment.

## Pretrained models and AI assistance

Public GPT-2 weights (MIT) and Pythia-410m weights (Apache-2.0) are
downloaded and evaluated inside the notebook runtime. Their parameters
remain frozen. Exact model revisions are recorded in the notebook.

No external LLM API is used to label or predict competition test
passages. No external labeled authorship dataset or hidden test labels
are used.

ChatGPT/Codex assisted with implementation, debugging, model ideas,
documentation and draft interpretation. 
